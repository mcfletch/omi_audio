# omi_audio — code review, 2026-07-31

Reviewer: Claude Opus 5 · Scope: the whole `omi_audio` package at commit-time state
(1,806 statements of source, 1,732 lines of tests) · Target: fitness for a public release.

> **Status: acted on before the first release.** Every finding below except
> **Q10** (version control, which is now in hand) was addressed. The headline
> changes are the `maxDistance` conformance fix, the non-finite-gain guard, the
> correctly-sized oversized block, and moving `uri` resolution out of the
> library entirely into [`AudioLibrary`](../../src/omi_audio/library.py).
>
> **This document has deliberately not been edited to match.** It records what
> the package looked like on the day it was reviewed, and rewriting it would
> destroy the only thing it is now good for. Line numbers and quoted code refer
> to the state before the fixes and will not match the current source; the code
> and its tests are where the present tense lives.

---

## Summary

`omi_audio` is a well-architected package. The two-thread split is real and
consistently honoured, the module boundaries are clean, and the docstrings are the
best-written prose I have reviewed in this workspace — they explain *why*, not
*what*, which is what makes a 28-year-old codebase survivable. It passes
`mypy --strict` and `ruff` cleanly, holds 95% branch coverage, and 9 of the 12
behavioural mutations I injected were caught by the tests.

It is **not ready for a public release**, for six reasons, in order:

1. `distance_gain` diverges from the `KHR_audio_emitter` formulas by up to 50× at
   long range (**C1**).
2. A single NaN position silences *every* sound in the mix, for as long as it
   persists (**C2**).
3. The documented "oversized block plays silence" behaviour produces garbage at a
   real device, not silence (**C3**).
4. An untrusted glTF `uri` reaches `os.path.abspath` with no confinement (**S1**).
5. Emitters cannot be located: node and scene emitter references, the part of the
   spec that says *where a positional emitter is*, are not modelled at all (**F1**).
6. `bufferView` audio — i.e. every `.glb` file — is silently unplayable (**F2**).

Items 1–4 are bugs in shipped behaviour. Items 5–6 mean the library cannot yet do
the job its README claims for real-world glTF content.

Separately, and directly answering the questions that prompted this review: the
**VRML97 field mapping is undocumented** (**V1**), and there is **not one diagram of
emitter geometry anywhere** (**D1**) — no cone, no distance curves, no ellipsoids,
no pan arc. Those are the two documentation gaps that matter most.

### What is genuinely good

Stated plainly, so the criticism below is read in proportion:

- **Docstrings.** `mixer.py`'s module docstring explains the audio-thread contract
  better than most commercial audio engines' design documents. The habit of naming
  the failure a design avoids ("steering somebody else's explosion") is excellent.
- **`tests/test_output_level.py`.** Asserting a *level in dBFS at the device
  hand-off*, for an emitter at a stated position, is an unusually good idea and
  catches a class of failure that gain-arithmetic tests cannot.
- **The tox `playback`/`nobackend` split.** Exercising the silent path as itself
  rather than through a monkeypatch is the right call and rare.
- **`TestAllocationDiscipline`.** Pinning the no-allocation rule under `tracemalloc`
  turns an aspiration into a test.
- **`mypy --strict` clean**, `py.typed` shipped, `license-files` declared, lazy
  optional backend, `NullDevice` as a first-class backend.
- **Licensing discipline.** The `miniaudio`-only choice, with the reasoning recorded
  in `pyproject.toml`, is correct and well documented.

### Measurements

| Check | Result |
|---|---|
| `pytest` | 234 passed, 1.7 s |
| `coverage --branch` | 95% (783 stmts, 33 missed; 168 branches, 12 partial) |
| `ruff check` (E,W,F,B) | clean |
| `mypy` (configured) | clean |
| `mypy --strict` | clean |
| Mutation sweep (12 injected) | 9 killed, **3 survived** |

Per-module coverage: `model` 99%, `spatial` 98%, `clip` 96%, `mixer` 96%,
`engine` 91%, `synth` 94%, `device` **84%**.

---

## Correctness

### C1 — `distance_gain` clamps to `maxDistance` for all three models; the spec clamps for none · `spatial.py:112-113` · **blocker**

The extension gives these formulas verbatim:

```
inverse     : refDistance / (refDistance + rolloffFactor * (max(d, refDistance) - refDistance))
exponential : pow(max(d, refDistance) / refDistance, -rolloffFactor)
linear      : 1.0 - rolloffFactor * (d - refDistance) / (maxDistance - refDistance)
```

Neither the inverse nor the exponential formula mentions `maxDistance`.
`spatial.py:112-113` clamps `distance` to `maxDistance` before *every* model:

```python
if max_distance > 0.0:
    distance = min(distance, max_distance)
```

Measured, at `refDistance=1.0`, `maxDistance=10.0`, `inverse`:

| distance | spec | omi_audio | error |
|---|---|---|---|
| 5 m | 0.200 | 0.200 | — |
| 10 m | 0.100 | 0.100 | — |
| 50 m | 0.020 | 0.100 | **5× too loud** |
| 500 m | 0.002 | 0.100 | **50× too loud** |

The spec's *prose* is different again: `maxDistance` is "the maximum distance
between the emitter and listener, beyond which the audio cannot be heard".
`omi_audio` implements neither reading — it makes the emitter audible **forever**,
at a fixed −20 dB plateau, when the author asked for a 10 m cap. A scene with
capped ambient emitters will have every one of them audible across the whole map.

This is pinned as correct by `test_spatial.py:61`
(`test_distance_is_clamped_to_the_maximum_before_the_curve_is_applied`) and
restated in `SPATIALISATION.md:20-33` and the `distance_gain` docstring, so the
divergence is currently protected on three sides.

**Fix.** Implement the three formulas verbatim; use `max_distance` only in the
`linear` denominator. That matches Web Audio, which is what Blender, Godot and
three.js exporters target, so it is also the interoperable choice. The prose/formula
conflict in the extension is worth a short note in `SPATIALISATION.md` recording
which reading was taken and why — and worth raising upstream with OMI. Update the
test to assert the spec values above.

### C2 — one NaN position poisons the entire mix · `engine.py:180-190`, `mixer.py:446-461` · **blocker**

A non-finite position — a degenerate scenegraph transform, an uninitialised bone, a
physics blow-up, a division by a zero scale — produces NaN gains from `gains_for`.
`VoiceHandle.set_gain` stores them unchecked, `_ramp_channel` multiplies them into
the shared `self._out` buffer, and `np.clip` does **not** remove NaN. Because every
voice accumulates into that one buffer, one bad emitter destroys *all* audio.

Verified:

```
clean block max: 0.704
after one NaN position  -> whole-mix NaN: True
... stays NaN for every block while the bad position persists
after aiming back at a valid position -> recovers one block later
```

The innocent second voice in that test is silenced too. On a real device this is
full-scale noise or a dead output, depending on the driver.

**Fix.** Sanitise at the control-thread boundary, where it is free — in
`VoiceHandle.set_gain` and `Mixer.play_gains`:

```python
def _finite(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0
```

Do **not** guard on the audio thread; the boundary is the right place and costs two
comparisons per frame per sound. Add a test that aims an emitter at a NaN position
and asserts the block is finite and the *other* voice is untouched.

### C3 — the oversized-block path does not produce silence at a real device · `mixer.py:477-490` · **blocker**

When a device asks for more than `max_block`, `blocks()` yields
`self._silence[:min(frames, self.max_block)]` — *fewer frames than were asked for*.
miniaudio's playback callback does:

```python
samples_bytes = _bytes_from_generator_samples(samples)
ffi.memmove(output, samples_bytes, len(samples_bytes))
```

It copies only what it was given and **leaves the rest of the output buffer
unwritten** — stale audio from the previous callback, or uninitialised memory.
Measured with the real `miniaudio` 1.71:

```
oversized ask: 128 frames requested -> 512 bytes yielded (needs 1024)
  -> 512 bytes of the device buffer left UNWRITTEN
```

So the outcome is a repeating fragment or a buzz, not the silence promised by
`mixer.py:469-476` and `MIXING.md:100-106`. `test_a_request_larger_than_the_maximum_is_served_as_silence`
inspects only the array the generator yields, so it confirms a claim that does not
hold where it matters.

**Fix.** Yield a correctly-sized zero buffer. Keep `_silence` sized to the largest
request ever seen and grow it on the (rare, already-logged) mismatch path — that
allocation happens once, on a path that is already broken, and is far better than
emitting garbage. Then extend the test to assert the yielded block is
`frames` long, not merely all-zero.

### C4 — `to_gltf` aliases mutable model state into the JSON it returns · `model.py:195-199`

`_write` copies field values by reference. `AudioEmitter.sources` is a `list`, so
the returned block shares it with the model:

```
block['emitters'][0]['sources'] is doc.emitters[0].sources  -> True
after mutating the JSON, the model says: [0, 1, 99]
```

Anyone who post-processes an exported block — remapping indices during a merge, for
example, which is the normal reason to touch it — silently corrupts the document
they exported from.

**Fix.** In `_write`, copy container values: `list(value) if isinstance(value, list) else value`.
Add a round-trip test that mutates the block and asserts the model is unchanged.

### C5 — `from_gltf` raises on malformed content, contradicting its own contract · `model.py:168-192`

`model.py:148-150` and `DATA-MODEL.md:52-54` both promise that bad content "should
cost a sound, not a traceback". Measured:

| input | result |
|---|---|
| `{'emitters': {'a': 1}}` | `AttributeError: 'str' object has no attribute 'items'` |
| `{'emitters': ['nonsense']}` | `AttributeError` |
| `{'audio': [None]}` | `AttributeError` |
| `{'sources': [{'gain': 'loud'}]}` | loads happily, `gain='loud'` |
| `{'emitters': [{'sources': 'abc'}]}` | loads happily, `sources='abc'` |

The last two are worse than the crashes: a string `gain` flows through the model and
detonates later, in the mixer, far from the document that caused it —
`TypeError: can't multiply sequence by non-int of type 'float'`.

**Fix.** Two changes in `_read`: skip entries that are not mappings (and non-list
arrays), and coerce each value to its declared field type, falling back to the
default and logging once on failure. `dataclasses.fields()` already carries the
types. Add a `TestMalformedDocuments` class covering the table above.

---

## Security

### S1 — path traversal / arbitrary local file access from an untrusted glTF `uri` · `clip.py:185-197`, `engine.py:235`

`ClipCache.key` calls `os.path.abspath` on whatever name it is handed, and
`play_source` hands it `audio.uri` straight from the document. Verified end to end:

```
play_source(uri='../../../../etc/passwd') -> decoder handed '/etc/passwd' (exists=True)
uri='/etc/shadow'                          -> decoder handed '/etc/shadow'
uri='http://evil.example/x.mp3'            -> treated as a relative file path
uri='data:audio/mpeg;base64,AAAA'          -> treated as a relative file path
uri='sounds%2Fshot.wav'                    -> not percent-decoded, fails to resolve
```

There is no base-directory confinement, no scheme rejection, and no percent-decoding
(glTF URIs *are* percent-encoded, so that last line is also a plain correctness bug).

**Impact, stated honestly:** this is not a direct file-read primitive — `miniaudio`
will refuse to decode `/etc/passwd`, and the bytes never reach the caller. It is
(a) a file-existence oracle, since a resolvable path is decoded and an unresolvable
one is not, and the two log differently; (b) an arbitrary-audio-file read from
anywhere on the host; and (c) on Windows, a UNC path such as `\\attacker\share\x.wav`
becomes an outbound SMB connection and an NTLM credential leak. For a library whose
entire purpose is consuming third-party scene files, that is worth closing.

**Fix.** Give `ClipCache` a `base_dir`, `unquote()` the URI, reject any name with a
scheme, and reject any resolved path that escapes the base via `os.path.commonpath`.
Document `uri` as untrusted input in `DATA-MODEL.md`. Handle `data:` URIs explicitly
(**F3**) rather than letting them fall through to the filesystem.

### S2 — `filterwarnings = ["ignore::DeprecationWarning"]` · `pyproject.toml:77`

This suppresses, globally, exactly the NumPy 2.x and Python 3.14/3.15 deprecations
that a matrix spanning 3.10–3.15 exists to surface. Replace with
`error::DeprecationWarning:omi_audio` or delete the line.

---

## API and functional gaps

### F1 — node and scene emitter references are not modelled · `model.py` · **blocker**

The extension attaches emitters to the scene graph:

```json
"nodes":  [{ "extensions": { "KHR_audio_emitter": { "emitters": [0] } } }],
"scenes": [{ "extensions": { "KHR_audio_emitter": { "emitters": [1, 3] } } }]
```

with the rule that scenes may carry only `global` emitters, nodes either kind.
`omi_audio.model` reads only the three document-level arrays. A consumer that loads
a glTF with this library therefore **cannot find out which node a positional emitter
is attached to** — which is the entire content of "positional". The reference
consumer, OpenGLContext, has to re-parse the glTF itself to recover it.

**Fix.** Add `emitters_for_node(node)` and `emitters_for_scene(scene)` returning
resolved `AudioEmitter` lists, validate the global-only-on-scenes rule (warn, do not
raise), and document the full `node → emitter → source → audio` chain in
`DATA-MODEL.md` — the existing mermaid diagram stops one link short of where it
needs to start.

### F2 — `bufferView` audio is silently unplayable · `clip.py`, `engine.py:232-240` · **blocker**

`Audio.bufferView` is modelled, but there is no bytes→`Clip` path anywhere:
`decode_file` takes a path and nothing else exists. Verified — `play_source` on a
`bufferView`-backed source returns `None`, with no warning. Since `.glb` embeds its
audio in buffer views, **the dominant shipping format for glTF is silently silent.**

**Fix.** Add `decode_bytes(data, sample_rate, mime_type=None) -> Clip` over
`miniaudio.decode()`, and `ClipCache.put_bytes(key, data)`. Give `play_source` an
optional `resolve_buffer_view: Callable[[int], bytes]` so the caller supplies the
buffer without `omi_audio` learning what a glTF buffer is. Until it exists, say so
in `DATA-MODEL.md` rather than leaving a modelled field that does nothing.

### F3 — `data:` URIs unhandled

glTF permits `data:audio/mpeg;base64,…`. Currently treated as a file path and
silently failed. Once **F2** lands this is four lines.

### F4 — `play_source` falls back to `audio.name` as a filename · `engine.py:235`

```python
name = audio.uri or audio.name
```

In glTF, `name` is a human-readable label, not a locator. This will either produce a
nonsense path or — worse — resolve a label that happens to collide with a real file.
Use `uri` only; if it is empty, that is a document with no locator and should warn.

### F5 — `autoplay` is modelled and never acted on

Nothing in the engine consults it and no document tells the consumer they must.
Either implement it in `play_source`/a scene-start helper, or state the division of
responsibility in `DATA-MODEL.md`.

### F6 — elevation is computed and discarded · `engine.py:188`

`azimuth, _ = listener.azimuth_elevation(position)`. Correct for a stereo
equal-power panner, but it means a sound directly overhead and one dead ahead are
indistinguishable, and that limitation is written down nowhere.
`SPATIALISATION.md` should say it in one sentence.

### F7 — `Mixer.max_block` is writable but its buffers are sized once · `mixer.py:189-226`

Raising it produces exactly the silent truncation the docstring says it refuses:

```
Mixer(max_block=64); m.max_block = 4096
m.mix(4096) -> shape (64, 2)      # 64 frames returned for a 4096-frame request
```

It passes the `frames > self.max_block` guard, then `self._out[:4096]` silently
yields 64 rows. The project's own test at `test_output_level.py:119` mutates this
attribute, which legitimises the pattern.

**Fix.** Make `max_block` a read-only property fixed at construction. Rework that
test to construct a small-`max_block` mixer instead.

### F8 — `master_gain` has two sources of truth at construction · `engine.py:58-61`

It is passed to `Mixer(master_gain=…)` *and* stored in `self._master_gain`. Harmless
today only because `_volume` starts at `1.0`. Construct the mixer at unity and let
`_applyGain()` be the single writer.

---

## The VRML97 mapping

### V1 — the geometry is implemented and well tested; the *mapping* is missing entirely

`ellipsoid_reach`/`ellipsoid_gain` are correct and genuinely well covered — my
mutation replacing the linear-in-dB ramp with a linear-in-amplitude one was killed
by two tests. The focal-conic derivation in the docstring is a pleasure to read.

But the delivery stops there, and this is what the review was asked about:

1. **Nothing inside `omi_audio` uses them.** The only consumer is
   `openglcontext/OpenGLContext/scenegraph/audio.py:401`.
2. **The caller must compute `cos_theta` itself, and no helper is provided** — while
   the glTF path gets `_cone_angle` done for it inside the engine (`engine.py:259`).
   The asymmetry means every consumer re-derives the same eight lines; OpenGLContext's
   `_ellipsoidGain` is exactly that re-derivation.
3. **No document maps a VRML97 `Sound` node onto this API.** `SPATIALISATION.md`
   explains the mathematics and never says which field goes where. A reader holding a
   `Sound` node has no way to get from it to a call.

**Fix — add `docs/VRML97.md`** containing:

- A field-by-field table: `minFront`→`min_front`, `minBack`→`min_back`,
  `maxFront`→`max_front`, `maxBack`→`max_back`, `intensity`→a gain multiplier,
  `location`/`direction`→world-space vectors the caller must transform,
  `priority`→`Mixer.play(priority=)`, `spatialize`→whether to pan at all,
  `AudioClip.pitch`→`rate`, `startTime`/`stopTime`→the caller's scheduling.
- A worked example, ten lines, from node to `play_gains`.
- **What is not implemented**, stated in `omi_audio`'s own docs rather than only in
  the consumer's docstring: fractional seeking into a clip whose `startTime` predates
  the scene.
- A diagram of the two ellipsoids (see **D1**).

**And add `spatial.ellipsoid_gain_at(location, direction, listener_position, …)`**
that does the `cos_theta` derivation, mirroring what `_cone_angle` already does for
cones. That is the function consumers actually want, and it removes the one piece of
geometry every one of them currently duplicates.

---

## Documentation

### D1 — there is not one diagram of emitter geometry · **the largest documentation gap**

Two mermaid diagrams exist and both are good: the thread split in
`ARCHITECTURE.md:8-20` and the three-array indirection in `DATA-MODEL.md:16-20`.

`SPATIALISATION.md` — the document whose entire subject is *how each emitter type
shapes the sound* — has **no diagram at all**. Missing, specifically:

- **The cone.** Inner and outer angles drawn as *diameters* (the single most
  misread thing in this API — the code takes pains over it at `spatial.py:143-144`
  and the prose has to say "angular diameters" twice because there is no picture),
  the unattenuated interior, the interpolation band, and where `coneOuterGain` takes
  over.
- **The three distance curves on one set of axes**, gain against distance, with
  `refDistance` and `maxDistance` marked — which makes "only `linear` reaches
  silence" obvious at a glance instead of a sentence to be trusted.
- **The two VRML97 ellipsoids**, sharing a focus at the sound, `front`/`back`
  asymmetry visible, with the dB ramp between them shaded. The focal-conic
  explanation is currently three paragraphs of prose doing a diagram's job.
- **The equal-power pan arc**, including the behind-the-listener fold, which is
  surprising behaviour that a picture makes self-evident.
- **A global vs positional emitter** sketch — one listener, two emitters, showing
  which one moves.

These can be inline SVG, mermaid, or a small committed matplotlib script with its
PNG output (the last has the advantage that the curves are then generated from
`spatial.py` itself and cannot drift from it — which I would recommend).

### D2 — no game/engine integration guide

The README has a quick start and one paragraph pointing at OpenGLContext. For a
library whose stated audience is games, the missing chapter is:

- the per-frame loop shape — `listen()` → `aim()` every live handle → cull;
- handle lifetime: when to keep one, when to drop it, why `None` is normal;
- choosing a voice budget, and what actually happens at the ceiling;
- priority conventions (the spec says 1.0 is most important — say what 0.5 means);
- the three patterns that cover most games: one-shots, looping positional ambience,
  and non-spatial music, each with code;
- what an application must do itself: `autoplay`, repeat intervals and variance
  (`DATA-MODEL.md:66-78` correctly identifies these as absent from the extension,
  then leaves the reader there);
- threading rules for the *application* — the docs describe the library's two
  threads thoroughly and never say "call all of this from one thread", which is the
  rule a consumer actually needs. `ClipCache` is explicitly not thread-safe
  (`clip.py:166-168`) and that constraint appears nowhere in `docs/`.

### D3 — documented guarantees the code does not meet

**C1**, **C3** and **C5** are each stated as a guarantee in the docs. Documentation
that overstates is worse than absent documentation, because it stops the reader
checking. Whichever way each is resolved, the prose must move with the code.

### D4 — no examples directory, no runnable demo

`synth` exists precisely so that a demo needs no assets and no licences — and there
is no demo. A 30-line "sound orbits the listener" script would exercise the real
device path (**T3**), give **D2** its backbone, and be the first thing most
evaluators run.

### D5 — no rendered API reference

The docstrings are the best asset this package has, and nothing renders them. They
are written in Sphinx style (`:class:`, `:func:`, `:mod:`, `Raises:` blocks) which
renders as noise on GitHub, and `pyproject.toml`'s `Documentation` URL points at the
docs *folder*. Add a minimal Sphinx or mkdocstrings build and publish it; this is a
few hours' work for a disproportionate gain.

### D6 — missing project files

No `CHANGELOG.md`, `CONTRIBUTING.md`, or `SECURITY.md`. For a public release
carrying a "review it before relying on it" warning, `SECURITY.md` and a changelog
are the two that matter.

---

## Test suite

The suite is better than most. Coverage is 95% with branch coverage on, tests are
named as behaviours, and the structure (one class per behaviour cluster) reads well.
The criticisms below are specific, not general.

### T1 — three surviving mutations = three untested claims

I injected twelve single-line behavioural changes and re-ran the suite. Nine were
caught. These three were not:

| Mutation | Result |
|---|---|
| `play_source` ignores the source's `gain` entirely | **234/234 still pass** |
| The engine ignores `volume` (uses `master_gain` alone) | **234/234 still pass** |
| Any `muffle > 0` treated as fully wet | **234/234 still pass** |

Each is a documented headline feature:

- **`test_the_sources_gain_is_applied` (`test_engine.py:168`) does not test that the
  source's gain is applied.** It asserts `handle is not None`. The `play_source`
  docstring calls honouring these settings "the whole point of reading the extension
  rather than inventing a format" — and it is unverified. *Fix:* play the same clip
  with `gain=1.0` and `gain=0.25`, mix both, assert the ratio.
- **`volume` is never tested at all** — not its clamping, not its existence, and not
  its product with `master_gain`, to which `ARCHITECTURE.md:69-82` devotes an entire
  section explaining that conflating them produces "a volume control which prints a
  new number and changes nothing". *Fix:* assert
  `engine.master_gain = 0.5; engine.volume = 0.5` reaches the mixer as `0.25`, and
  that each clamps independently.
- **The muffle *blend* is untested.** Every test uses `muffle=0.0` or `1.0`. The
  blend is the documented reason it is a float and not a bool ("so an application can
  fade it in as a listener submerges"). *Fix:* assert `muffle=0.5` lands between the
  dry and wet levels.

### T2 — assertions that cannot fail

`test_voices_report_themselves_for_a_debug_overlay` (`test_mixer.py:248`) asserts
`active_voices == 1` and `len(mixer.voices) == 4`, and never calls `__repr__` — which
is what "report themselves" means. No `__repr__` in the package is tested
(`Clip`, `Voice`, `VoiceHandle` all define one).

### T3 — the real device is never driven

Every `MiniaudioDevice` test uses `FakeBackend`/`FakePlaybackDevice`. The one
real-hardware test opens a device and reads `sample_rate` back. **Nothing verifies
that what `Mixer.blocks()` yields is a shape `miniaudio` accepts** — the single most
important integration in the package rests on an untested assumption.

I checked it by hand: a 2-D float32 memoryview does cast correctly, 32 frames → 256
bytes. It works. But it works by luck of `_bytes_from_generator_samples` handling
`itemsize != 1`, and a `miniaudio` release that tightened that would break playback
with a green suite.

**Fix.** Add a contract test against the real dependency, skipped when absent:
assert `miniaudio._bytes_from_generator_samples(block)` has length
`frames * channels * 4`. Better still, drive a real `PlaybackDevice` on
`miniaudio.Backend.NULL` — miniaudio's null backend runs a genuine audio thread with
no hardware, which would exercise the whole hand-off in CI.

### T4 — fixture hygiene

- `tmp_path` is requested and never used in `test_clearing_drops_the_decoded_samples`,
  and requested-but-pointless in `test_a_clip_decodes_once_and_is_then_returned_from_the_cache`
  and `test_the_cache_decodes_at_its_own_rate` (the decoder is a stub; the file on
  disk is irrelevant).
- The `engine` fixture is defined **twice**, with different bodies —
  `test_engine.py:19` and `test_output_level.py:79`.
- `constant`/`ramp`/`beep` helpers and the `needs_miniaudio` skip marker are
  re-rolled per module. `conftest.py` holds only `pose`, and is the right home for
  all of them.
- No registered markers and no `--strict-markers`/`--strict-config` in `addopts`.

### T5 — untested public surface

`device.describe()` (a public exported function, 0% covered); `engine.sample_rate`;
`engine.volume`; all three `__repr__`s; `VoiceHandle.set_gain_pan` on a *live* voice
(it is only ever called on a dead handle, so it is smoke-tested, not tested);
`Mixer.play` with an out-of-range `pan`; `distance_gain`'s `linear` /
`max<=ref` branch (`spatial.py:122` — the one uncovered statement); both degenerate
guards in `_cone_angle` (`engine.py:269,273`); `Clip.resampled` on an empty clip;
and the `EXTENSION`, `MIME_MPEG`, `GLOBAL` constants.

### T6 — no lint, type or coverage gate in CI

`.github/workflows/test.yml` runs `tox`; `tox` runs `pytest` only. `ruff` and `mypy`
are declared in the `dev` extra and **run nowhere**. The clean lint/strict-type state
this package currently enjoys is therefore unenforced and will rot on the first
contribution. There is no coverage measurement or threshold at all.

**Fix.** Add `lint` and `typecheck` tox envs, add them to `[gh]`, and add
`--cov=omi_audio --cov-branch --cov-fail-under=90` to the test env.

---

## Code quality and maintainability

### Q1 — `_applyGain` is camelCase · `engine.py:104,114,116`

The only camelCase name in an otherwise uniformly snake_case codebase. Rename to
`_apply_gain`. (`ruff` does not catch this; adding the `N` ruleset would.)

### Q2 — the lazy-import block is duplicated verbatim · `clip.py:38-57`, `device.py:40-63`

Two module globals, two `_import_attempted` flags, two `_backend()` functions, and
two public predicates for one fact — `decoder_available()` and
`miniaudio_available()`. `DEFAULT_SAMPLE_RATE = 44100` is likewise defined in both
`clip.py:34` and `device.py:28`. Two copies drift; the tests already have to
monkeypatch each module separately, which is the smell showing through.

**Fix.** One `_backend.py` with `backend()` and `available()`; re-export both
historical names for compatibility. One `DEFAULT_SAMPLE_RATE`.

### Q3 — the numpy typing is `Any`, which is why `--strict` passes

The package ships `py.typed`, so its annotations are a promise to downstream type
checkers. It passes `mypy --strict` partly because the interesting values are
untyped:

- `Clip.__init__(samples: Any)`, and `Clip.samples` has no annotation at all;
- `blocks() -> Generator[Any, int, None]` — should be `Generator[memoryview, int, None]`;
- `AudioDevice.start(source: Any)`;
- `Listener.from_view_platform(platform: Any)`.

With `numpy>=2.0` required, `numpy.typing.NDArray[np.float32]` is available and
free. `platform: Any` should be a `Protocol` with `position` and `quaternion` — that
*documents in the type system* the duck-typing the docs make a selling point of,
which is strictly better than a prose paragraph.

### Q4 — pre-3.9 typing style

`Dict`/`List`/`Optional`/`Tuple` from `typing` throughout, with
`requires-python = ">=3.10"` and `from __future__ import annotations` already in
every module. `dict[str, Any]`, `list[int]` and `X | None` are available, shorter,
and what a reader in 2026 expects. `ruff`'s `UP` ruleset automates the whole change.

### Q5 — `mypy` pinned to one version · `pyproject.toml:87`

`python_version = "3.12"` while the package supports 3.10–3.15. Combined with **T6**
(never run in CI), the type checking is weaker than it looks.

### Q6 — a `positional` block on a `global` emitter is silently discarded · `model.py:131-135`

Correct per spec, but such a document is malformed and the author gets no signal.
Log once at debug or info.

### Q7 — `_read` puts a raw `dict` into a typed field · `model.py:183-186`

`_read(AudioEmitter, entry)` passes `positional={...}` — a `dict` — into a field
annotated `Optional[PositionalProperties]`, and `from_gltf` replaces it two lines
later. It works, but the object is in a lying state in between, the annotation is
defeated, and `_read(AudioEmitter, …)` called alone (as a future maintainer
reasonably might) returns a broken object. Exclude `positional` in `_read` and set
it explicitly.

### Q8 — release posture

`0.1.0` with `Development Status :: 3 - Alpha` and a prominent "largely LLM-written,
no guarantees" warning is honest — and I would keep the warning; it is the right call
and rarely made. But if this is going to PyPI, publish it as `0.1.0a1` so the index
itself reflects that status, and add the `CHANGELOG` (**D6**) before the first
release rather than after.

### Q9 — Python 3.15 has no trove classifier

It is in the tox envlist and the CI matrix but missing from `classifiers`.

### Q10 — the package is entirely untracked in git

`git ls-files` inside `omi_audio/` returns **zero files**. The whole package is
untracked in the parent workspace repository. `pyproject.toml` already points at
`github.com/mcfletch/omi_audio`, so the intent is a standalone repo — but until that
exists, every file here, including this review and all of `docs/`, is a single copy
with no history and no undo.

**This should be the first action taken**, ahead of any fix in this document.

### Provenance note

The workspace's `CLAUDE.md` asks that new provenance be recorded as a spec file
under the project's own `specs/` directory, cited from the code. `omi_audio` cites
`KHR_audio_emitter`, the Web Audio API and ISO/IEC 14772-1 by URL from its
docstrings, which is good practice and is very likely sufficient — all three are
published specifications, not copyleft source. But there is no `specs/` directory
and no clean-room record, so if any behaviour (the ellipsoid derivation, the
equal-power fold) was established by reading an *implementation* rather than a
specification, that is currently unrecorded. A short `specs/README.md` naming the
sources consulted would close the question permanently.

---

## Recommended order of work

**Before any public release**

1. Put the package under version control in its own repository (**Q10**).
2. Fix the three shipped-behaviour bugs: `maxDistance` conformance (**C1**), NaN
   poisoning (**C2**), oversized-block garbage (**C3**).
3. Confine `uri` resolution (**S1**), and stop treating schemes and percent-encoding
   as filenames (**F3**).
4. Kill the three surviving mutations with real assertions (**T1**), and add the
   `miniaudio` contract test (**T3**).
5. Correct the three docs claims that no longer match the code (**D3**).

**Before calling the glTF support complete**

6. Node and scene emitter references (**F1**).
7. `bufferView`/`data:` audio (**F2**, **F3**), and drop the `audio.name` fallback (**F4**).
8. Fix `to_gltf` aliasing (**C4**) and `from_gltf` robustness (**C5**).

**Documentation, which is what will decide whether anyone adopts this**

9. Diagrams for every emitter type in `SPATIALISATION.md` (**D1**), generated from
   `spatial.py` so they cannot drift.
10. `docs/VRML97.md` with the field mapping, plus `ellipsoid_gain_at()` (**V1**).
11. The game-integration guide (**D2**) and a runnable demo (**D4**).
12. A rendered API reference (**D5**).

**Maintainability, worth doing while the code is still small**

13. CI gates for `ruff`, `mypy` and coverage (**T6**) — this one protects all the
    others.
14. De-duplicate the backend import (**Q2**); real numpy types and a `Protocol`
    (**Q3**); modern typing syntax (**Q4**); `_apply_gain` (**Q1**);
    `max_block` read-only (**F7**).
15. Consolidate test fixtures into `conftest.py` (**T4**) and cover the public
    surface listed in **T5**.
