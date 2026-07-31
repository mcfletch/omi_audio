# Mixing, and the audio-thread contract

`omi_audio.mixer` is the part that runs on the **audio thread**, and everything
about its shape follows from that.

## The rules

- **The pool is fixed.** `Voice` slots are made once, at construction, and
  reused. Starting a sound configures a slot; it never allocates one. A scene
  that fires a thousand sounds a second costs what a scene firing ten does, and
  the cost is known before it runs.
- **The buffers are made once too.** `Mixer.mix()` writes into pre-allocated
  arrays with NumPy's `out=` parameters and returns a *view*. An allocation on
  the audio thread is a garbage collection on the audio thread, and that is an
  audible gap.
- **Nothing there blocks, decodes, resolves a path or logs.** Those all happen
  on the control thread, before a clip ever reaches a voice.
- **The lock is the control thread's alone.** `Mixer.play()` takes it so two
  *control* threads cannot claim one slot. The mixing never takes it.
- **Every number crossing from the control thread is checked there.** See
  [below](#gains-are-checked-at-the-boundary).

The no-allocation rule is asserted, not assumed: a test mixes a full pool for
twenty blocks under `tracemalloc` and holds the result to nothing measurable.

## Gains are ramped, never stepped

A gain that jumps from one block to the next is a discontinuity, and a
discontinuity is a click. Each voice carries the gain it *ended* the last block
at and the gain it is *aiming* for, and the mixer interpolates across the block.

This is also what smooths the hard edge at the outer boundary of a VRML97
sound's ellipsoid, which is why that curve is left as the specification writes
it rather than being fudged.

## Gains are checked at the boundary

`VoiceHandle.set_gain()` and `Mixer.play_gains()` replace any gain that is not a
finite number with zero, and refuse a rate that is not finite and positive.

That is not defensiveness for its own sake. Positions come out of a scenegraph,
and a scenegraph produces NaN for perfectly ordinary reasons: a transform with a
zero scale, an uninitialised bone, a physics step that blew up, a normalisation
of a zero vector. A NaN gain multiplied into the mix **stays** — `np.clip` does
not remove it — and because every voice accumulates into one shared buffer, a
single bad emitter would silence *every sound in the scene* for as long as the
bad position persisted.

The check is at the boundary rather than in the mixing, so it costs two
comparisons per aimed sound and nothing at all per frame. Silence is the right
substitute: it is what an emitter would sound like if it were nowhere, which is
what a NaN position says it is. The moment the position is good again, the sound
comes back.

`Mixer.max_block` is read-only for a related reason. Every buffer above was
sized from it, so raising it afterwards would pass `mix()`'s guard and then
silently return a short block — exactly the truncation that guard exists to
refuse. A device that needs a bigger period needs a new `Mixer`.

## Refusing, and taking back

The pool has to be able to say no, and it has to be able to *take back*. When
every voice is busy the newcomer is weighed against the weakest one playing — by
`priority` first, then by how audible it currently is — and either steals it or
is refused. Stealing the quietest sound of the lowest priority is the least
audible theft available.

Because a slot can be taken back, `play()` hands out a **`VoiceHandle`** rather
than the slot itself:

```python
handle = engine.play('explosion.wav', priority=0.9)
...
handle.set_gain(0.2, 0.4)   # does nothing at all once the sound has gone
handle.stop()               # likewise; no caller ever has to test first
```

A handle remembers *which* sound it was for, by generation. Once the slot holds
something else, the handle is inert. Without that, a caller still steering a
sound whose slot was recycled would be steering somebody else's explosion — a
mistake that is silent, intermittent and very hard to find, which is why it is
designed out rather than documented.

`play()` returning `None` — the clip is empty, the rate is not positive, or
every voice is busy with something more important — is an ordinary outcome, not
an error. `engine.aim()` accepts `None`, so a caller need not check.

## Resampling

Each voice holds a fractional cursor and a rate, and the mixer interpolates
linearly between neighbouring samples. That is what makes an arbitrary playback
rate possible, and it is also how a clip recorded at another sample rate is
played back correctly: the rate is scaled by `clip.sample_rate /
mixer.sample_rate`.

`playbackRate` changes speed and pitch together, as speeding up a record does.
It is a resampling ratio, not a pitch shift.

## Muffling

`mixer.muffle` runs from 0 (clear) to 1 (underwater) and blends the mix towards
a low-passed copy of itself. The filter is a cascade of four two-tap averages —
`y[n] = (x[n] + x[n−1]) / 2` — which is the cheapest thing that is genuinely a
low-pass rather than merely quieter. Each stage has the response `cos(ω/2)`, so
four of them are `cos⁴(ω/2)`: about −33 dB near Nyquist and under −0.03 dB at a
hundredth of it. That is the shape of "heard through water", reached with three
vector operations per stage and no recursion.

Each stage carries the sample that fell off the end of the previous block;
without it the filter would restart every block and tick at the seam.

It is a blend rather than a switch so an application can fade it in as a
listener submerges:

```python
engine.muffle = min(1.0, depth_below_surface / 0.5)
```

## The device hand-off

A device is handed the generator `Mixer.blocks()` returns and pulls from it on
its own thread: it `send()`s the number of frames it wants and receives a
`memoryview` of the mixed block. A view, not a copy, so no audio buffer is
allocated per callback — and therefore a block is only valid until the next
call.

A device asking for **more frames than the mixer prepared** gets silence, *of
the size it asked for*, rather than an exception. Raising inside a device
callback tears down playback on a thread nobody is watching, which is a worse
failure than a gap.

The size matters as much as the silence. A backend copies the bytes it is handed
and leaves the rest of its output buffer exactly as it was — holding the previous
callback's audio, or nothing at all. Half a block of zeroes followed by half a
block of last time is not a gap but a buzz, repeating for as long as the
mismatch lasts. So the block is grown to fit: one allocation, on a path that is
already broken, at most once per size, which is a far better trade than emitting
garbage. Build the `Mixer` with a larger `max_block` to fix the cause.

The mismatch is logged once and only once — the single deliberate exception to
the no-logging-on-the-audio-thread rule, because an unreported permanent silence
would be worse still.

`Mixer.mix()` called directly *does* raise for an oversized block: there is no
room to mix one, and quietly truncating would drop audio silently.

## Clipping

Summed voices can exceed full scale. The mix is clipped to `[-1, 1]`, which is
honest about it and keeps the device from wrapping the waveform round on
conversion.
