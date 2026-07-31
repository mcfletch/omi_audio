# omi_audio

Renderer-agnostic **spatial audio** for Python, built natively on the glTF
[`KHR_audio_emitter`](https://github.com/omigroup/gltf-extensions/tree/main/extensions/2.0/KHR_audio_emitter)
data model — which is the Web Audio `PannerNode` model. Emitters have a distance
curve, an optional directional cone and a gain; a listener hears them panned
about its own forward axis. The mixing is a fixed voice pool summed in NumPy.

- **NumPy is the only hard dependency.** No graphics library, no scenegraph, no
  audio library is required to mix.
- **`miniaudio` is optional** — it decodes files and reaches the sound card.
  Without it the library still mixes and simply stays silent.
- **Nothing here is copyleft**, and neither is anything it depends on:
  `miniaudio` and every decoder it bundles are MIT or public domain.
- **Silence is a backend.** A missing package, a device that will not open, or a
  machine with no audio hardware all end in one warning and a `NullDevice`.
  `open_device()` cannot raise.
- **The audio thread never allocates, blocks, decodes or logs.** Pre-allocated
  buffers, NumPy `out=` everywhere, and a lock the mixing never takes.
- **A document's `uri` is never resolved here.** A glTF file comes from
  somebody you have never met; deciding what one of its strings is allowed to
  mean is the application's call, not a library's. See
  [`AudioLibrary`](docs/DATA-MODEL.md#getting-the-actual-audio-audiolibrary).

> ⚠️ **This code is largely LLM-written.** It has a test suite, but it comes with
> **no guarantees** of correctness, accuracy, or fitness for any purpose (see
> [`LICENSE`](LICENSE), MIT). Review it before relying on it for anything that
> matters.

## Install

```bash
pip install omi_audio                # mixing only (NumPy)
pip install "omi_audio[playback]"    # + miniaudio, for files and a sound card
```

## Quick start

```python
from omi_audio import AudioEngine, model, synth

engine = AudioEngine()                      # opens a device, or silence

# A sound with no file behind it, so this runs anywhere.
engine.clips.put('ping', synth.impact(0.4, seed=1))

emitter = model.AudioEmitter(
    gain=0.8,
    positional=model.PositionalProperties(refDistance=2.0, rolloffFactor=1.0),
)

handle = engine.play('ping', emitter=emitter, position=(3.0, 0.0, -5.0))
```

Run it for real, with no assets and no sound card required:

```bash
python examples/orbit.py            # a sound orbits the listener for ten seconds
python examples/orbit.py --silent   # never opens a device, still prints levels
```

Each frame, move the listener and re-aim whatever is still playing. A moving
sound is *re-aimed*, never restarted: `aim()` writes two floats and the mixer
ramps to them across the next block.

```python
engine.listen(camera)                       # anything with .position/.quaternion
engine.aim(handle, emitter, position=emitter_world_position)
```

## Testing without a sound card

Everything below the device is arithmetic over arrays, so build an engine on a
`NullDevice` and assert on the mix:

```python
from omi_audio import AudioEngine, NullDevice, synth

engine = AudioEngine(device=NullDevice(sample_rate=8000), voices=8)
engine.mixer.play(synth.tone(440.0, 1.0, sample_rate=8000), pan=1.0)
block = engine.mixer.mix(64)                # (64, 2) float32
assert block[:, 0].max() < 1e-9             # nothing in the left ear
assert block[:, 1].max() > 0.1              # and plenty in the right
```

## The pieces

In the order sound travels through them:

| Module | Answers |
|---|---|
| `model` | `KHR_audio_emitter` as typed records, with the extension's own field names and defaults; `from_gltf`/`to_gltf` round-trip, and the node/scene references that say where an emitter *is* |
| `library` | What a document's audio references have resolved to — the seam where **your** resolver, not this library, decides what a `uri` means |
| `spatial` | Every gain curve — three glTF distance models, the Web Audio cone, VRML97's two ellipsoids, equal-power panning — and the listener's pose |
| `clip` | Encoded audio → mono float32 at one rate, from a file or from bytes (`.glb` buffer views, `data:` URIs, downloads), decoded once |
| `synth` | Tones, chirps, noise and impacts made out of arithmetic, so demos and tests need no assets and no licences |
| `mixer` | A fixed voice pool summed into stereo blocks: allocation-free, lock-free on the audio thread, priority stealing, gain ramping, an underwater low-pass |
| `device` | Where blocks go — `miniaudio`, or silence |
| `engine` | The one object an application holds |

Deeper documentation is in [`docs/`](docs/README.md) — start with
[**GAME-INTEGRATION.md**](docs/GAME-INTEGRATION.md) if you are building
something, or [SPATIALISATION.md](docs/SPATIALISATION.md) for every gain curve
with a diagram of each, generated from the code that implements it.

## What this does not do

Stated up front, because finding out later is worse:

- **Stereo only, and the pan carries azimuth alone.** A sound directly overhead
  and one dead ahead are indistinguishable. Height needs an HRTF and surround
  needs more than two channels; neither is here.
- **No reverb, occlusion or doppler.** `muffle` is the only effect, and it is a
  master-bus low-pass.
- **No streaming.** Clips are decoded whole into memory.
- **No scheduling.** Nothing here has a clock; `autoplay` starts when your
  application says the scene has begun.

## Using it from a scenegraph

`omi_audio` is deliberately ignorant of scenegraphs: it is handed world
positions, a listener pose and clips. [OpenGLContext](https://github.com/mcfletch/openglcontext)
is the reference integration — `AudioEmitter`, `AudioSource` and VRML97's
`Sound` nodes, a per-context engine driven once a frame from the render pass,
and a glTF loader that reads `KHR_audio_emitter` blocks straight into this
model.

## Contributing, changes, security

- [CONTRIBUTING.md](CONTRIBUTING.md) — how to run the gates, and the two rules
  with teeth (the audio thread, and the untrusted document).
- [CHANGELOG.md](CHANGELOG.md)
- [SECURITY.md](SECURITY.md) — the threat model, and why your resolver is the
  boundary.

## Licence

MIT. Note that some jurisdictions do not allow for LLM generated code to have
a copyright, as such this library may be in the Public Domain in your 
jurisdiction. There is **NO** warranty of any kind on the software.

See [`LICENSE`](LICENSE).
