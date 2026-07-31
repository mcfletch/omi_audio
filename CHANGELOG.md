# Changelog

Notable changes to `omi_audio`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/) — with the usual caveat that
`0.x` makes no compatibility promise, and this is an alpha.

## [Unreleased]

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
