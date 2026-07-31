# VRML97 `Sound` nodes

VRML97 attenuates sound in a way nothing else can express, so `omi_audio`
implements it rather than approximating it with the glTF model. This page is the
map from a `Sound` node's fields to this API — everything a consumer holding a
parsed node needs in order to make a call.

The mathematics, and a diagram of the two ellipsoids, are in
[SPATIALISATION.md](SPATIALISATION.md#the-vrml97-ellipsoids). The reference
consumer is [OpenGLContext](https://github.com/mcfletch/openglcontext).

## Why not just map it onto the glTF model

A `Sound` node's attenuation is **two ellipsoids sharing a focus at the sound**,
with a ramp between them that is linear in decibels. `KHR_audio_emitter` offers
a distance curve plus a cone, and no combination of those produces an ellipsoid:
the glTF cone attenuates by *angle alone* and the VRML ellipsoid's reach *is* a
function of angle. Approximating one with the other would put the boundary in
the wrong place in every direction except two.

So both live in `omi_audio.spatial`, side by side, and a consumer picks the one
its content was authored for.

## The field map

| VRML97 `Sound` field | Goes to | Notes |
|---|---|---|
| `location` | `ellipsoid_gain_at(location=…)` | **World space.** The node's field is in its own coordinate system; apply the transform stack first. |
| `direction` | `ellipsoid_gain_at(direction=…)` | World space likewise. Need not be unit length. A zero vector means "no front and no back", and the `front` distances then apply in every direction. |
| `minFront` | `min_front` | Inner ellipsoid, along `direction`. |
| `minBack` | `min_back` | Inner ellipsoid, against it. |
| `maxFront` | `max_front` | Outer ellipsoid, along `direction`. |
| `maxBack` | `max_back` | Outer ellipsoid, against it. |
| `intensity` | a multiplier on the result | VRML97 defines it as a linear factor in `[0, 1]`, which is what `gain` means here too. |
| `priority` | `Mixer.play(priority=…)` / `AudioEngine.play(priority=…)` | Same convention in both specifications: **1.0 is the most important**. It decides what a new sound may steal and what it may be refused for. |
| `spatialize` | whether to pan at all | `TRUE` → pan by azimuth as below. `FALSE` → hand the same gain to both ears, as a global emitter does. |
| `source` | an `AudioClip` or `MovieTexture` | Only the audio matters here; resolve it to a `Clip`. |
| `AudioClip.pitch` | `rate=` | A resampling ratio: speed and pitch move together, as speeding up a record does. |
| `AudioClip.loop` | `loop=` | |
| `AudioClip.startTime` / `stopTime` | **the caller's scheduling** | See [what is not implemented](#what-is-not-implemented). |
| `AudioClip.description` | nothing | A caption, not a playback parameter. |

There is no equivalent of the glTF cone, and no equivalent of `distanceModel`:
the ellipsoids *are* the distance model.

## A worked example

Ten lines, from a parsed node to a sounding voice.

```python
from omi_audio import AudioEngine, spatial

engine = AudioEngine()
engine.clips.put('bell', decode_your_audioclip(sound.source))   # your resolver

def gain_for(sound, location, direction, listener):
    """`sound`'s linear gain at `listener`, with its fields already in world space."""
    return sound.intensity * spatial.ellipsoid_gain_at(
        location, direction, listener.position,
        min_front=sound.minFront, min_back=sound.minBack,
        max_front=sound.maxFront, max_back=sound.maxBack)
```

Starting it, and then following it every frame:

```python
level = gain_for(sound, location, direction, engine.listener)
if sound.spatialize:
    azimuth, _ = engine.listener.azimuth_elevation(location)
    left, right = spatial.equal_power_pan(azimuth)
else:
    left = right = 2.0 ** -0.5                       # the same in both ears
handle = engine.mixer.play_gains(engine.clip('bell'), level * left, level * right,
                                 priority=sound.priority, loop=clip.loop,
                                 rate=clip.pitch)

# ... each frame, with the node's world pose recomputed:
handle.set_gain(level * left, level * right)         # inert once the sound has gone
```

`set_gain` writes two floats onto a playing voice, so following every `Sound`
node in a scene costs a pair of floats each. Nothing is restarted, re-resolved or
re-decoded — see [MIXING.md](MIXING.md#gains-are-ramped-never-stepped).

## What `ellipsoid_gain_at` saves you

It computes the distance and the cosine of the angle between the sound's
direction and the direction to the listener, then calls `ellipsoid_gain`. Doing
that by hand is eight lines, and every consumer would write the same eight —
`omi_audio`'s glTF path already has the equivalent done for it inside the
engine, and the asymmetry was a real gap.

`ellipsoid_reach` and `ellipsoid_gain` remain available for a consumer that has
already worked the geometry out, or that wants to draw the surface.

## What is not implemented

Stated here rather than only in a consumer's docstring, because it is the
library's limitation and not theirs:

- **Fractional seeking.** A `Sound` whose `AudioClip.startTime` is in the past
  when the scene loads should begin part-way through the clip. Voices always
  start at sample zero, so such a sound starts at its beginning instead. There
  is no `Mixer.play(offset=…)`.
- **`startTime`/`stopTime` scheduling.** Nothing here has a clock. A consumer
  drives its own timeline and calls `play` and `stop` at the right moments,
  which is what a scenegraph is already doing for everything else in the scene.
- **`MovieTexture` audio.** The video half is somebody else's problem, and the
  audio half arrives here as ordinary samples once it has been separated.
- **Elevation.** Panning is by azimuth only; see
  [SPATIALISATION.md](SPATIALISATION.md#what-the-output-does-not-do).

## Reference

ISO/IEC 14772-1:1997 (VRML97), 6.42 `Sound` and 6.2 `AudioClip` —
<https://www.web3d.org/documents/specifications/14772/V2.0/part1/nodesRef.html#Sound>
