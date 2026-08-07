"""Sounds made out of arithmetic rather than out of files.

Demonstrations and tests need something audible.  Shipping audio files would
mean shipping their licences, so the sounds here are generated: a tone, a noise
burst, a swept chirp and a percussive impact are enough to hear a distance
curve, a pan and a voice pool working, and they cost nothing to redistribute.

They are also useful in their own right as placeholders while real content is
being made -- a game with a synthesised gunshot is a game that can be played and
tuned, which is the point of a placeholder.

Every function returns a :class:`~omi_audio.clip.Clip`, so a
synthesised sound is played by exactly the same path as a decoded one.
"""

from __future__ import annotations


from collections.abc import Callable
from typing import cast

import numpy as np

from omi_audio.clip import DEFAULT_SAMPLE_RATE, Clip

#: Seconds of fade at each end of a generated tone.  A waveform that starts at
#: full amplitude starts with a step, and a step is a click.
DEFAULT_FADE = 0.01


def _time_base(duration: float, sample_rate: int) -> np.ndarray:
    """Sample times in seconds for a clip of ``duration``."""
    return np.arange(max(0, int(duration * sample_rate)), dtype=np.float64) / sample_rate


def _fade_window(frames: int, sample_rate: int, fade: float) -> np.ndarray:
    """A 1.0 envelope with linear ramps of ``fade`` seconds at each end."""
    window = np.ones(frames, dtype=np.float32)
    ramp = min(int(fade * sample_rate), frames // 2)
    if ramp > 0:
        edge = np.linspace(0.0, 1.0, ramp, endpoint=False, dtype=np.float32)
        window[:ramp] = edge
        window[frames - ramp:] = edge[::-1]
    return window


def silence(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Clip:
    """A clip of nothing.  Useful as a placeholder that occupies real time."""
    return Clip(np.zeros(max(0, int(duration * sample_rate)), dtype=np.float32),
                sample_rate, name='silence')


def tone(frequency: float, duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE,
         amplitude: float = 0.5, fade: float = DEFAULT_FADE,
         harmonics: int = 1) -> Clip:
    """A sine wave, faded in and out so it starts and stops without clicking.

    ``harmonics`` adds that many partials in all, each at ``1/n`` of the
    fundamental's amplitude -- the series of a sawtooth, which is why more of
    them sound progressively brighter and reedier rather than merely louder.
    The default of 1 is a bare sine.

    Reach for more than one whenever the sound has to *show* something about
    timbre, a filter above all: a sine has nothing above its fundamental, so a
    low-pass can only change how loud it is and never how it sounds. Partials
    that would land above Nyquist are left out rather than allowed to alias back
    down as an out-of-tune whistle, and the amplitude is shared out across the
    partials so the sum still fits inside ``amplitude``.
    """
    times = _time_base(duration, sample_rate)
    partials = [n for n in range(1, max(1, harmonics) + 1)
                if frequency * n < sample_rate / 2.0] or [1]
    weights = np.array([1.0 / n for n in partials])
    weights /= weights.sum()
    samples = np.zeros(times.size, dtype=np.float64)
    for partial, weight in zip(partials, weights, strict=True):
        samples += weight * np.sin(2.0 * np.pi * frequency * partial * times)
    samples = (amplitude * samples).astype(np.float32)
    if samples.size:
        samples *= _fade_window(samples.size, sample_rate, fade)
    return Clip(samples, sample_rate, name='tone %gHz' % (frequency,))


def chirp(start: float, end: float, duration: float,
          sample_rate: int = DEFAULT_SAMPLE_RATE, amplitude: float = 0.5,
          fade: float = DEFAULT_FADE) -> Clip:
    """A sine sweeping linearly from ``start`` to ``end`` hertz.

    The phase is the integral of the instantaneous frequency, not the frequency
    times the time; getting that wrong produces a sweep that ends at the wrong
    pitch and is the classic mistake here.
    """
    times = _time_base(duration, sample_rate)
    if not times.size:
        return Clip(np.zeros(0, dtype=np.float32), sample_rate, name='chirp')
    rate = (end - start) / max(times[-1], 1e-9)
    phase = 2.0 * np.pi * (start * times + 0.5 * rate * times * times)
    samples = (amplitude * np.sin(phase)).astype(np.float32)
    samples *= _fade_window(samples.size, sample_rate, fade)
    return Clip(samples, sample_rate, name='chirp %g-%gHz' % (start, end))


def noise(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE,
          amplitude: float = 0.5, seed: int | None = None,
          fade: float = DEFAULT_FADE) -> Clip:
    """Uniform white noise.  Wind, static, the raw material of an impact."""
    generator = np.random.default_rng(seed)
    frames = max(0, int(duration * sample_rate))
    samples = generator.uniform(-amplitude, amplitude, frames).astype(np.float32)
    if samples.size:
        samples *= _fade_window(samples.size, sample_rate, fade)
    return Clip(samples, sample_rate, name='noise')


def impact(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE,
           amplitude: float = 0.9, seed: int | None = None,
           decay: float = 12.0) -> Clip:
    """A percussive hit: noise under a sharp exponential decay.

    ``decay`` is the exponent's rate -- larger is shorter and drier.  The attack
    is instantaneous on purpose; a transient is what makes a hit read as a hit.

    White noise is bright at every length, so this is a crack, a tap or a
    hiss.  For the bottom of the range -- a motor, a detonation -- see
    :func:`rumble`.
    """
    times = _time_base(duration, sample_rate)
    generator = np.random.default_rng(seed)
    samples = generator.uniform(-1.0, 1.0, times.size)
    samples *= amplitude * np.exp(-decay * times)
    return Clip(samples.astype(np.float32), sample_rate, name='impact')


def rumble(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE,
           amplitude: float = 0.9, seed: int | None = None,
           decay: float = 4.0, attack: float = 0.0, cutoff: float = 400.0,
           pitch: float = 70.0, pitch_end: float | None = None,
           tone: float = 0.5, drive: float = 1.0, tilt: float = 0.0,
           floor: float = 0.0) -> Clip:
    """The bottom of the range: noise with the top taken off, over a falling tone.

    What an explosion, a motor and distant thunder have in common, and what
    :func:`impact` cannot be: white noise is bright however long it decays, so
    a bang built from it is a crack rather than a boom.  Two sources make the
    difference here -- ``tone`` is the balance between them, from 0 for all
    noise to 1 for all body.

    ``cutoff`` is where the noise starts to roll away above, in **hertz**, and
    ``floor`` where it rolls away below; the slopes are gentle ones, so they
    shape the sound rather than muting it.  A ``floor`` of nought leaves the
    bottom alone, and the two together are a **band** -- which is what anything
    hollow is, a tube or a drum shell or a pipe: it rings around a pitch and
    has very little underneath it.
    ``tilt`` tips the noise the other way, in **decibels per octave**: 0 is
    white, and negative numbers raise the bottom relative to the top, which is
    what an explosion, a motor and thunder all are.

    **Weight from ``tilt`` and weight from ``tone`` are not the same sound.**
    A low sine under a hard attack is a *drum*, and one that falls as it goes
    is a drum being tuned; that is what a listener hears whenever the tone
    carries the bottom end, however quiet it is against the noise.  Weight from
    tilted noise is a thump with no pitch in it, which is what a blast is.
    Reach for ``tone`` when something is genuinely meant to have a note in it.

    ``pitch`` and ``pitch_end`` are that tone's frequency at the start and at
    the end, in hertz, and ``pitch_end`` defaults to half of ``pitch``, an
    octave down.

    ``decay`` is the envelope's rate as in :func:`impact`, and ``attack`` is
    seconds to reach full level: 0 is an instantaneous transient, which is a
    detonation, and a tenth of a second or so is something spooling up.

    ``drive`` saturates the result, which adds harmonics above the body tone
    and is what makes a boom throaty rather than round.  1 leaves it clean; the
    level is brought back to ``amplitude`` afterwards, so driving a sound makes
    it dirtier and never louder.
    """
    times = _time_base(duration, sample_rate)
    if not times.size:
        return Clip(np.zeros(0, dtype=np.float32), sample_rate, name='rumble')
    mix = min(1.0, max(0.0, float(tone)))
    end = float(pitch) * 0.5 if pitch_end is None else float(pitch_end)
    samples = ((1.0 - mix) * _dark_noise(times.size, sample_rate, cutoff, seed,
                                         tilt, floor)
               + mix * _falling(times, float(pitch), end))
    samples *= _envelope(times, decay, attack)
    if drive > 1.0:
        samples = np.tanh(samples * float(drive))
    peak = float(np.abs(samples).max())
    if peak > 0.0:
        samples *= float(amplitude) / peak
    return Clip(samples.astype(np.float32), sample_rate, name='rumble')


#: Where a reverberation tail's bands are divided, in hertz, and how fast each
#: dies relative to the length asked for.  **The middle is what comes back.**
#: Measured off a rifle recorded outdoors: through the whole of its tail the
#: band from one to four kilohertz carries half the energy, while the bottom
#: stays at three or four per cent and the top is gone inside a fifth of a
#: second.  That is what distance does -- air takes the top, and the low end of
#: a report is a near-field thump that never comes back off anything -- and a
#: tail that keeps all three bands equally reads as static rather than as a
#: place.
_TAIL_BANDS = ((500.0, 1.9), (3500.0, 1.0), (None, 2.8))


def reverberated(clip: Clip, seconds: float = 0.6, level: float = 0.5,
                 seed: int | None = None) -> Clip:
    """``clip`` with the tail of a room behind it, baked in.

    Where :func:`echoed` gives back a handful of *returns*, this gives back one
    sound going on: a dense decaying wash, darkening as it goes.  The
    difference is not a matter of degree.  Three discrete repeats are heard as
    repeats -- a clap, and then another clap -- and what a hard sound outdoors
    or in a hall actually does is keep within a few decibels of its peak for
    the better part of a second while the top drains out of it.  A gunshot
    without that reads as a drum however its spectrum is arranged.

    ``seconds`` is roughly how long the tail takes to fall away, and ``level``
    how loud it is against the sound itself; nought hands the clip straight
    back.  ``seed`` fixes the room, so a reference recording is reproducible.

    **This is baked into a clip when it is made, and it is not a bus effect.**
    Nothing here runs on the audio thread, nothing is shared between voices,
    and the mixer still has no reverb -- what it has is a clip that already
    sounds like it happened in a place.
    """
    if level <= 0.0 or not clip.frames:
        return clip
    frames = max(1, int(float(seconds) * clip.sample_rate))
    tail = _tail(frames, clip.sample_rate, float(seconds), seed)
    wet = np.convolve(np.asarray(clip.samples, dtype=np.float64), tail)
    peak = float(np.abs(wet).max())
    if peak > 0.0:
        wet *= float(np.abs(clip.samples).max()) / peak
    samples = np.zeros(wet.size, dtype=np.float64)
    samples[:clip.frames] = clip.samples
    samples += wet * float(level)
    loudest, before = float(np.abs(samples).max()), float(np.abs(clip.samples).max())
    if loudest > before > 0.0:
        samples *= before / loudest
    return Clip(samples.astype(np.float32), clip.sample_rate,
                name='%s in a room' % (clip.name,))


def _tail(frames: int, sample_rate: int, seconds: float,
          seed: int | None) -> np.ndarray:
    """One room's response: noise dying away, the top of it dying first.

    Three bands, each decaying at its own rate, which is the cheapest thing
    that is *shaped* like a room rather than merely long.  The alternative --
    one noise under one envelope -- stays as bright at the end as it was at
    the start, and a tail that never darkens reads as static rather than as a
    place.  Which band outlasts which is :data:`_TAIL_BANDS`.
    """
    times = np.arange(frames, dtype=np.float64) / sample_rate
    generator = np.random.default_rng(seed)
    noise = generator.uniform(-1.0, 1.0, frames)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(frames, 1.0 / sample_rate)
    built = np.zeros(frames, dtype=np.float64)
    low = 0.0
    for edge, rate in _TAIL_BANDS:
        within = (freqs >= low) if edge is None else ((freqs >= low)
                                                      & (freqs < edge))
        band = np.fft.irfft(np.where(within, spectrum, 0.0), n=frames)
        built += band * np.exp(-6.9 * rate * times / max(1e-6, seconds))
        low = edge if edge is not None else low
    return built


def echoed(clip: Clip, delay: float = 0.09, level: float = 0.4,
           taps: int = 3, damping: float = 0.0,
           thinning: float = 0.0) -> Clip:
    """``clip`` with quieter copies of itself behind it.

    A slap-back rather than reverb: there is no room here to model, and what a
    hard, sharp sound in a large place actually gives back is a handful of
    discrete returns.  It is also what *makes* a sound read as hard -- a crack
    with nothing behind it could have come from anywhere, and the returns are
    how a listener knows it did not.

    ``delay`` is seconds to the first repeat, ``level`` how loud it is relative
    to the direct sound, and each repeat after it is ``level`` of the one
    before.  A ``level`` of nought is no echo at all and the clip is handed
    straight back, so a voice that declares none pays nothing for the option.

    ``damping`` is where a repeat starts to lose its top and ``thinning``
    where it loses its bottom, both in **hertz**, and both applied again to
    each repeat in turn.  Between them they are why a return is heard as the
    sound coming back rather than as the sound happening twice: air and soft
    surfaces take the high end away as it travels, and the heavy near-field
    thump of something like a gunshot never comes back off anything at all.
    Zero for either leaves that end of the repeats alone.

    The result is long enough to hold the last repeat, and **never louder than
    what it was given**: repeats that land on top of one another are brought
    back down together, because an echo is a sound arriving late and not a
    sound arriving louder.
    """
    if level <= 0.0 or taps < 1 or not clip.frames:
        return clip
    step = max(1, int(float(delay) * clip.sample_rate))
    frames = clip.frames + step * int(taps)
    samples = np.zeros(frames, dtype=np.float64)
    samples[:clip.frames] = clip.samples
    coming_back = np.asarray(clip.samples, dtype=np.float64)
    weight = 1.0
    for repeat in range(1, int(taps) + 1):
        weight *= float(level)
        if damping > 0.0:
            coming_back = _darkened(coming_back, clip.sample_rate, damping)
        if thinning > 0.0:
            coming_back = _thinned(coming_back, clip.sample_rate, thinning)
        at = step * repeat
        samples[at:at + clip.frames] += coming_back * weight
    peak, before = float(np.abs(samples).max()), float(np.abs(clip.samples).max())
    if peak > before > 0.0:
        samples *= before / peak
    return Clip(samples.astype(np.float32), clip.sample_rate,
                name='%s echoed' % (clip.name,))


def _darkened(samples: np.ndarray, sample_rate: int,
              cutoff: float) -> np.ndarray:
    """One pass of the same roll-off :func:`_dark_noise` shapes noise with."""
    return _shaped(samples, sample_rate,
                   lambda ratio: 1.0 / (1.0 + ratio ** 2), cutoff)


def _thinned(samples: np.ndarray, sample_rate: int,
             cutoff: float) -> np.ndarray:
    """The same roll-off from the other end: the bottom taken away."""
    return _shaped(samples, sample_rate,
                   lambda ratio: ratio ** 2 / (1.0 + ratio ** 2), cutoff)


def _shaped(samples: np.ndarray, sample_rate: int,
            gain: Callable[[np.ndarray], np.ndarray],
            cutoff: float) -> np.ndarray:
    """``samples`` with each frequency scaled by ``gain(f / cutoff)``."""
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
    spectrum *= gain(freqs / max(1e-6, float(cutoff)))
    return np.fft.irfft(spectrum, n=samples.size)


#: Where a tilted noise stops rising and starts falling again, in hertz.  Near
#: the bottom of hearing, and it is a *floor* rather than a limit for a reason:
#: an untempered slope runs away as the frequency approaches nothing, and what
#: it produces is a clip that is mostly one enormous inaudible excursion with
#: the sound riding quietly on top of it.  Below this the same slope is turned
#: around, so a tilt spends its headroom on what can be heard.
_TILT_FLOOR = 30.0


def _dark_noise(frames: int, sample_rate: int, cutoff: float,
                seed: int | None, tilt: float = 0.0,
                floor: float = 0.0) -> np.ndarray:
    """White noise with everything far above ``cutoff`` rolled away.

    ``floor`` takes the bottom off in the same way, so the two are a band;
    ``tilt`` then tips what is left, in decibels per octave, so the one routine
    makes a hiss with the top taken off, the bottom-heavy roar of something
    burning, and the hollow pop of a tube.

    Shaped in the frequency domain rather than by a recursive filter: the
    magnitude wanted is a curve, saying it *is* the curve is the clearest way
    to write it, and one transform costs less than a sample-by-sample loop over
    a clip's worth of frames.
    """
    generator = np.random.default_rng(seed)
    spectrum = np.fft.rfft(generator.uniform(-1.0, 1.0, frames))
    freqs = np.fft.rfftfreq(frames, 1.0 / sample_rate)
    spectrum *= 1.0 / (1.0 + (freqs / max(1e-6, float(cutoff))) ** 2)
    if floor > 0.0:
        spectrum *= (freqs ** 2) / (freqs ** 2 + float(floor) ** 2)
    if tilt:
        octaves = np.log2(np.maximum(freqs, _TILT_FLOOR) / _TILT_FLOOR)
        spectrum *= 10.0 ** (float(tilt) * octaves / 20.0)
        # And away again below the floor, so the level a tilt buys is spent on
        # what a listener can hear rather than on a slow inaudible heave.
        spectrum *= (freqs ** 2) / (freqs ** 2 + _TILT_FLOOR ** 2)
    samples = np.fft.irfft(spectrum, n=frames)
    peak = float(np.abs(samples).max())
    return samples / peak if peak > 0.0 else samples


def _falling(times: np.ndarray, start: float, end: float) -> np.ndarray:
    """A sine sweeping from ``start`` to ``end`` hertz over ``times``.

    The phase is the integral of the frequency, which is the same rule
    :func:`chirp` follows and the same one that is wrong if it is skipped.
    """
    span = max(float(times[-1]), 1e-9)
    rate = (end - start) / span
    return cast(
        np.ndarray, np.sin(2.0 * np.pi * (start * times + 0.5 * rate * times * times))
    )


def _envelope(times: np.ndarray, decay: float, attack: float) -> np.ndarray:
    """An exponential decay, optionally reached over ``attack`` seconds."""
    shape = np.exp(-max(0.0, float(decay)) * times)
    if attack > 0.0:
        shape = shape * (1.0 - np.exp(-times / float(attack)))
    return shape
