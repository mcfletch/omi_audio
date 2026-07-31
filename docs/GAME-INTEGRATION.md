# Using this in a game

`omi_audio` knows nothing about your engine. It is handed world positions, a
listener pose and clips, and it produces stereo blocks. This page is the other
half of that bargain: what the application has to do, and in what order.

There is a runnable version of everything below in
[`examples/orbit.py`](../examples/orbit.py), which needs no assets and no sound
card.

## The one rule that is not about audio

**Call all of it from one thread** — the one that draws the frame.

`AudioEngine`, `ClipCache` and `AudioLibrary` are control-thread state with no
locks on them. The only threading in the design is the device's own, and it is
entirely on the far side of `Mixer.blocks()`; `Mixer.play()` takes a lock so two
*control* threads cannot claim one voice slot, but nothing else here is safe to
call concurrently.

If a download lands on a worker thread, hand the bytes to the frame thread and
call `library.supply_bytes()` there. You are already doing that for meshes.

## The shape of a frame

```python
engine.listen(camera)                     # 1. where the ears are
for sounding in live_sounds:              # 2. re-aim everything still playing
    engine.aim(sounding.handle, sounding.emitter,
               position=sounding.node.world_position,
               forward=sounding.node.world_forward)
live_sounds = [s for s in live_sounds     # 3. drop what has finished
               if s.handle is not None and s.handle.playing]
```

Three things, once a frame, and none of them touches the audio thread. `aim()`
writes two floats onto a playing voice and the mixer ramps to them across the
next block, so following a hundred moving emitters costs a hundred pairs of
floats. Nothing is restarted, re-resolved or re-decoded.

Order matters only between 1 and 2: aim after the listener has moved, or every
sound is one frame behind the camera.

## Handles

`play()` returns a `VoiceHandle`, or `None`.

**`None` is normal.** The clip did not resolve, or the voice pool refused the
sound because everything was busy with something more important. It is an
outcome, not an error — `aim()` accepts `None`, so a caller need never test
first.

**A handle is only ever about the sound it was made for.** Once that sound
finishes or is stolen, the handle goes inert: `playing` is False and everything
else does nothing. So:

- **Keep a handle** for anything you will steer or stop: a looping ambience, an
  engine note, a sound attached to a moving object.
- **Drop it** for a one-shot you will never touch again — a footstep, a gunshot.
  `engine.play(...)` and discard the result.
- **Never keep one past the object it belongs to.** It will not do damage if you
  do, but it will keep an object alive that should have gone.

```python
handle = engine.play('engine-loop', emitter=emitter, position=car.position,
                     loop=True, priority=0.8)
...
handle.stop()                             # harmless if it already finished
```

## Choosing a voice budget

`AudioEngine(voices=N)` fixes the pool at construction. Slots are made once and
reused, so `N` is a memory and CPU ceiling, not a target — a scene that fires a
thousand sounds a second costs what a scene firing ten does.

The default is 32. Thirty-two simultaneous sounds is already past what a listener
can pick apart; the reason to raise it is a scene with a lot of quiet continuous
ambience competing with the loud transient things, and the reason to lower it is
a very small machine.

**At the ceiling nothing fails.** A new sound is weighed against the weakest one
playing — by `priority` first, then by how audible it currently is — and either
steals that slot or is refused. Stealing the quietest sound of the lowest
priority is the least audible theft available.

## Priority

`priority` follows `KHR_audio_emitter` and VRML97: **1.0 is the most important**,
0.0 the least, and the default is 0.0. It is only ever compared against other
sounds, so what matters is the ordering you choose, not the numbers. A workable
scale:

| | |
|---|---|
| `1.0` | dialogue, and anything the player must hear to keep playing |
| `0.8` | the player's own actions — their weapon, their footsteps, their vehicle |
| `0.5` | other actors nearby |
| `0.2` | ambience and scenery |
| `0.0` | decoration; first to be dropped |

Give everything the same priority and the pool falls back to stealing the
quietest, which is a reasonable second choice but a worse one than saying what
matters.

## Three patterns that cover most games

### A one-shot

```python
engine.clips.put('shot', synth.impact(0.3, seed=1))        # once, at load

emitter = model.AudioEmitter(gain=1.0, positional=model.PositionalProperties(
    refDistance=4.0, rolloffFactor=1.0))
engine.play('shot', emitter=emitter, position=muzzle.world_position,
            priority=0.8)                                   # result discarded
```

Fired sixty times a second, this costs one decode and one voice claim per shot.
The clip cache is what makes that true.

### Looping positional ambience

```python
self.handle = engine.play('river', emitter=self.emitter,
                          position=self.node.world_position,
                          loop=True, priority=0.2)
# and each frame:
engine.aim(self.handle, self.emitter, position=self.node.world_position)
```

Cull it yourself when it is far away — an inaudible emitter still costs a voice:

```python
if not self.emitter.positional.in_range(engine.listener.distance_to(position)):
    self.handle.stop()
```

`in_range()` is `maxDistance` read as a hard cutoff, which is what the extension's
prose says and what its formulas do not; see
[SPATIALISATION.md](SPATIALISATION.md#maxdistance-means-two-different-things-and-this-library-picks-one).

### Non-spatial music

```python
music = model.AudioEmitter(type='global', gain=0.6)
engine.play('theme', emitter=music, loop=True, priority=1.0)
```

A `global` emitter is heard the same wherever the listener stands, so there is
nothing to aim and nothing to update. Passing no emitter at all does the same
thing.

## Two volumes, and they multiply

| | Who owns it | Written by |
|---|---|---|
| `engine.volume` | the **player** | a settings screen, a volume key — safe to re-read every frame |
| `engine.master_gain` | the **application** | you, once: how loud this content is authored to be |

The mixer sees their product. Keeping them apart is what stops a per-frame
refresh of the player's setting from silently undoing the application's mix.

```python
engine.master_gain = 0.8                  # this scene is authored a bit hot
engine.volume = settings.audio_volume     # 0..1, re-read whenever you like
```

`engine.muffle` runs from 0 to 1 and blends the whole mix towards a low-passed
copy of itself — it is a float rather than a switch so you can fade it in:

```python
engine.muffle = min(1.0, depth_below_surface / 0.5)
```

## Loading a glTF scene

```python
document = model.from_gltf(gltf['extensions'][model.EXTENSION])
library = engine.library(document, fetch=my_fetch)          # see DATA-MODEL.md

for node in scene_nodes:
    for emitter in document.emitters_for_node(node):
        placed[emitter] = node
for emitter in document.emitters_for_scene(scene):
    placed[emitter] = None                                  # global: no transform

handles = engine.start_autoplay(
    library, place=lambda e: None if placed[e] is None
    else (placed[e].world_position, placed[e].world_forward))
```

`fetch` is where your resolver decides what a document's `uri` is allowed to
mean. `omi_audio` never resolves one — see
[DATA-MODEL.md](DATA-MODEL.md#getting-the-actual-audio-audiolibrary).

## What the application has to do itself

Not oversights: `KHR_audio_emitter` has nowhere to say these, and inventing
fields for them would be inventing a format.

- **When "loaded" happens.** `autoplay` means "when the glTF is loaded", and only
  you know whether that is the moment the file parsed or the moment the player
  walked into the room. Call `start_autoplay()` at the second one.
- **A repeat interval.** Ambience that should be *occasional* — a distant rumble,
  a bird — is not the same as ambience that loops. Looping gives a continuous
  noise where a sparse one was wanted. Keep a timer and fire a one-shot.
- **Variance on that interval.** Two speakers of the same sound started together
  will beat together for ever. Jitter each interval by ±20% or so.
- **Culling.** See `in_range()` above, and stop what the player cannot hear.
- **Scheduling of any kind.** Nothing here has a clock.

## Testing your integration without a sound card

Everything below the device is arithmetic over arrays:

```python
engine = AudioEngine(device=NullDevice(sample_rate=8000), voices=8)
engine.play(synth.tone(440.0, 1.0, sample_rate=8000), gain=1.0)
block = engine.mixer.mix(64)               # (64, 2) float32
assert abs(block).max() > 0.1
```

A `NullDevice` never pulls, so `mix()` is yours to call and to assert on. That
is how this package's own suite tests levels at particular positions — see
`tests/test_output_level.py` for the pattern.

`omi_audio.synth` exists so that a demo or a test needs no assets and no
licences: tones, chirps, noise and impacts, made out of arithmetic.

## What is not here

- **Elevation, HRTF, surround.** Output is stereo and the pan carries azimuth
  only; see [SPATIALISATION.md](SPATIALISATION.md#what-the-output-does-not-do).
- **Reverb, occlusion, doppler.** `muffle` is the only effect, and it is a
  master-bus low-pass rather than a per-source one.
- **Streaming.** Clips are decoded whole, into memory. A twenty-minute music
  track is a hundred megabytes of float32, so keep music short or loop it.
