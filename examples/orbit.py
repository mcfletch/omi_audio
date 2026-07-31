#!/usr/bin/env python3
"""A sound orbiting the listener: the smallest complete `omi_audio` application.

It needs **no assets and no sound card**.  The sound is synthesised, and if no
device opens the whole thing runs silently and still prints the levels, which is
what makes it something you can run on a build machine as well as at a desk::

    python examples/orbit.py                 # ten seconds, out loud if it can
    python examples/orbit.py --silent        # never opens a device
    python examples/orbit.py --seconds 3

What it demonstrates, in the order the code does it:

1. Building an engine, which opens a device or falls back to silence.
2. Registering a synthesised clip under a name of the application's own.
3. Describing an emitter with `KHR_audio_emitter` fields.
4. The frame loop that matters: move the listener, then **re-aim** the sound.
   The voice is started once and never restarted -- ``aim()`` writes two floats
   and the mixer ramps to them, which is why a scene can follow every sound it
   has, every frame, without the audio thread noticing.

The printed bar is the per-ear level, so a terminal with no speakers still shows
the sound crossing from one side to the other.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from omi_audio import AudioEngine, NullDevice, describe, model, synth

#: How far the sound orbits from the listener, in metres.
RADIUS = 4.0
#: Seconds for one full circle.
PERIOD = 4.0
#: How often the loop runs.  Sixty a second is a frame rate; nothing here needs
#: to keep up with the audio thread, which is running on its own.
FRAME = 1.0 / 60.0


def emitter() -> model.AudioEmitter:
    """A positional emitter with a short reach, so the orbit is audible.

    ``refDistance`` is where the curve reads 1.0 and inside which nothing is
    attenuated; at four metres with a two-metre reference the sound is at half
    amplitude, which leaves the panning plenty of room to be heard.
    """
    return model.AudioEmitter(
        gain=1.0,
        positional=model.PositionalProperties(distanceModel='inverse',
                                              refDistance=2.0, rolloffFactor=1.0),
    )


def bar(left: float, right: float, width: int = 24) -> str:
    """The two ear levels, drawn, so a machine with no speakers still shows it."""
    def marks(gain: float) -> str:
        return '#' * int(round(min(1.0, gain) * width))

    return '%*s|%-*s' % (width, marks(left), width, marks(right))


def orbit(engine: AudioEngine, seconds: float) -> None:
    """Run the frame loop for ``seconds``, following one sound round a circle."""
    sound = emitter()
    handle = engine.play('ping', emitter=sound, position=(0.0, 0.0, -RADIUS),
                         loop=True, priority=0.5)
    if handle is None:
        print('the voice pool refused the sound, which should not happen here')
        return

    started = time.monotonic()
    while (elapsed := time.monotonic() - started) < seconds:
        angle = 2.0 * math.pi * elapsed / PERIOD
        position = (RADIUS * math.sin(angle), 0.0, -RADIUS * math.cos(angle))

        # The whole of the per-frame work: aim the sound at where it now is.
        engine.aim(handle, sound, position=position)

        left, right = engine.gains_for(sound, position=position)
        where = 'ahead ' if position[2] < -0.1 else (
            'behind' if position[2] > 0.1 else 'beside')
        sys.stdout.write('\r  %s  %s  %4.1f s ' % (bar(left, right), where,
                                                   seconds - elapsed))
        sys.stdout.flush()
        time.sleep(FRAME)
    print()
    levels(engine)                      # while it is still sounding
    handle.stop()


def levels(engine: AudioEngine) -> None:
    """Assert-free proof that the chain works even with no device at all.

    Mixes a block directly and reports its peak, which is what the test suite
    does and what an application can do to check its own integration.
    """
    peak = float(np.abs(engine.mixer.mix(256)).max())
    print('  a mixed block peaks at %.3f (%.1f dBFS)'
          % (peak, 20.0 * math.log10(max(peak, 1e-9))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seconds', type=float, default=10.0,
                        help='how long to orbit for (default: 10)')
    parser.add_argument('--silent', action='store_true',
                        help='never open a device, even if one is available')
    parser.add_argument('--rate', type=int, default=44100,
                        help='sample rate to run the whole chain at')
    options = parser.parse_args(argv)

    device = NullDevice(sample_rate=options.rate) if options.silent else None
    engine = AudioEngine(device=device, voices=8)
    try:
        print(describe(engine.device))
        # A sound with no file behind it, so this runs anywhere and ships no
        # licences.  The name is the application's own -- a document's `uri`
        # never becomes one of these; see docs/DATA-MODEL.md.
        engine.clips.put('ping', synth.impact(0.35, sample_rate=engine.sample_rate,
                                              seed=1))
        engine.master_gain = 0.9
        orbit(engine, options.seconds)
    finally:
        engine.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
