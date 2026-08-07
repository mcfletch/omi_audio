# Changelog

Notable changes to `omi_audio`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/) — with the usual caveat that
`0.x` makes no compatibility promise, and this is an alpha.

## [Unreleased]

### Added

- `synth.rumble` — the low end of the range: noise with the top rolled away
  over a tone that falls as it goes, optionally saturated. `synth.impact` is
  white noise and is therefore bright however long it decays, so a detonation
  or a motor built from it comes out as a crack; this is the generator those
  want. `cutoff`, `pitch`/`pitch_end`, `tone`, `drive`, `tilt` and `attack`
  are its numbers, and the level is normalised afterwards so driving a sound
  makes it dirtier and never louder.

  **`tilt` is where weight should come from**, in decibels per octave: it
  leans the noise toward the bottom, which is what a blast, a motor and
  thunder are. Weight from `tone` is a different sound and usually the wrong
  one — a low sine under a hard attack is a *drum*, and one that falls as it
  goes is a drum being tuned, which is what it reads as however small a share
  of the mix it has. Reach for `tone` when something is meant to have a note
  in it.

  `cutoff` and `floor` are the top and the bottom edges of its noise, and the
  two together are a **band** — which is what anything hollow is. A tube rings
  around a pitch and has almost nothing underneath it, and that is the
  difference between the pop of a mortar and a thump.

- `synth.reverberated` — the tail of a room, baked into a clip when it is
  made. **Not a stronger `echoed`**: discrete returns are heard as returns —
  a clap, and then another clap — where a room is heard as one sound going
  on, and a short bright report with no tail reads as a *drum* however its
  spectrum is arranged. Three bands decaying at their own rates, with the
  middle outlasting both the bottom and the top, because that is what
  distance does to a sound. Nothing here runs on the audio thread and the
  mixer still has no reverb bus; what it has is clips that already sound like
  they happened somewhere.

- `synth.echoed` — a clip with quieter copies of itself behind it. A
  slap-back rather than reverb: there is no room modelled here, and what a
  hard sound in a large place gives back is a handful of discrete returns.
  `damping` and `thinning` take the top and the bottom off each repeat in
  turn, which is what stops a return sounding like the same event happening
  again: air takes the high end as the sound travels, and a near-field thump
  never comes back off anything at all. The result is never louder than what
  it was given.

## [0.1.0a1] — 2026-07-31

**Initial public release.** An alpha: the API may still move

### The package

Renderer-agnostic spatial audio built natively on glTF's `KHR_audio_emitter`
model — which is the Web Audio `PannerNode` model — and mixed in NumPy. NumPy is
the only hard dependency; `miniaudio` is an optional extra for decoding files
and reaching a sound card, and without it the library still mixes and stays
silent.

### What is in it

- **The data model** (`model`) — `KHR_audio_emitter` as plain dataclasses, field
  for field, with the extension's own names and defaults. `from_gltf` /
  `to_gltf` round-trip, and `from_gltf` never raises whatever a third-party
  document contains. Node and scene emitter references are read, so a positional
  emitter can be located.
- **Resolving a document's audio** (`library`) — `AudioLibrary` holds what one
  document's `audio` array has resolved to. **A `uri` is never resolved, opened
  or interpreted by this package**: the application supplies a `fetch` callback,
  because only it knows where its content lives and what a third-party document
  may reach. Audio arrives as a local file, as bytes, or as an already-decoded
  clip — so `bufferView` audio (every `.glb`) and `data:` URIs play, and a
  download that lands three frames later is an ordinary silence until it does.
- **Spatialisation** (`spatial`) — the extension's three distance models
  implemented as written, the Web Audio cone, equal-power stereo panning with
  the behind-the-listener fold, and VRML97's two ellipsoids for `Sound` nodes,
  which nothing else can express. Every curve is a pure function of geometry.
- **Clips** (`clip`) — encoded audio decoded to mono float32 at one rate, from a
  file or from bytes, decoded once and cached by name.
- **The mixer** (`mixer`) — a fixed voice pool summed into stereo blocks:
  allocation-free, lock-free on the audio thread, priority-based voice stealing,
  per-block gain ramping, and an underwater low-pass.
- **The device seam** (`device`) — `miniaudio`, or silence. A missing package, a
  device that will not open and a machine with no audio hardware all end in one
  warning and a `NullDevice`; `open_device()` cannot raise.
- **The engine** (`engine`) — the one object an application holds, keeping
  decoding and path resolution off the audio thread.
- **Synthesised sounds** (`synth`) — tones, chirps, noise and impacts made out of
  arithmetic, so a demo or a test needs no assets and no licences.

### Known limitations

Stated here because finding them out later is worse:

- **Stereo only**, and the pan carries azimuth alone — a sound overhead and one
  dead ahead are indistinguishable. Height needs an HRTF, surround needs more
  than two channels, and neither is implemented.
- **No reverb, occlusion or doppler.** `muffle` is the only effect and it is a
  master-bus low-pass.
- **No streaming**: clips are decoded whole, into memory.
- **No scheduling**: nothing here has a clock, so `autoplay` starts when the
  application says its scene has begun.
- **`maxDistance` follows the extension's formulas, not its prose** — the two
  disagree, and only the `linear` model uses it. `PositionalProperties.in_range()`
  is the other reading, kept explicit. See
  [SPATIALISATION.md](docs/SPATIALISATION.md#maxdistance-means-two-different-things-and-this-library-picks-one).

### Quality of the release

- 435 tests, 99% branch coverage, `ruff` and `mypy --strict` clean, all gated in
  CI across Python 3.10–3.15 with and without the optional backend.
- `py.typed` shipped, so the annotations are a promise downstream type checkers
  can use.
- **Largely LLM-written**, and the README, the package docstring and
  [SECURITY.md](SECURITY.md) all say so. Review it before relying on it for
  anything that matters.

[Unreleased]: https://github.com/mcfletch/omi_audio/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/mcfletch/omi_audio/releases/tag/v0.1.0a1
