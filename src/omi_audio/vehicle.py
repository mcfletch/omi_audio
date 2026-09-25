"""What a road vehicle sounds like, from what its physics already knows.

An electric vehicle's sound is mostly three continuous things: a motor whine
whose pitch follows the road speed, tyres whose hiss is how fast they roll and
whose roar is how much they scrub, and wind that goes as the square of speed.
Each is a :mod:`omi_audio.synth` clip played as a loop, with its gain and
playback rate set every frame; a collision is a one-shot on top.

:class:`VehicleSound` is the behaviour and touches no device: speed, scrub,
throttle and whether any wheel is on the ground in, gains and rates out.
:class:`VehicleSoundTuning` holds every number it uses, so a game gives its own
vehicle's figures (where its tyres let go, how loud its motor is) and the rest
is shared. The scene nodes that play it are
``OpenGLContext.audio.vehicle.VehicleSoundtrack``.

    >>> sound = VehicleSound()
    >>> for _ in range(60):
    ...     sound.update(1.0 / 60.0, speed=30.0, throttle=1.0)
    >>> round(sound.motor.rate, 2), round(sound.wind.gain, 3)
    (1.4, 0.087)
    >>> sound.hit(12.5)
    0.5

Scrub is the sideways speed at a tyre's contact patch, in metres per second;
``omi_physics``' ``Wheel.slip`` is that number.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from omi_audio import synth
from omi_audio.clip import DEFAULT_SAMPLE_RATE, Clip

__all__ = ['Voice', 'VehicleSound', 'VehicleSoundTuning', 'impact_clip',
           'motor_clip', 'tyre_clip', 'wind_clip']


@dataclass(frozen=True)
class VehicleSoundTuning:
    """Every figure a vehicle's sound is made from.

    Speeds are metres per second, gains 0 to 1, times seconds and pitches
    hertz. Frozen, so one tuning can key the clips made from it.
    """

    #: How long each loop is. ``motor_hz * loop`` should be a whole number,
    #: so the whine's clip holds whole cycles and its seam cannot be heard.
    loop: float = 2.0
    #: The motor's written pitch, and the road speed it plays at that pitch.
    motor_hz: float = 400.0
    motor_at: float = 25.0
    #: How many partials the whine has: the 1/n series of a sawtooth.
    motor_partials: int = 7
    #: The slowest the whine plays, as a fraction of its written pitch, so a
    #: vehicle standing still hums rather than stopping its clip.
    motor_idle: float = 0.2
    #: How loud the motor is with full throttle, and from speed alone at
    #: ``motor_at``.
    motor_load: float = 0.16
    motor_roll: float = 0.05
    #: Tyres rolling: how loud, reached at what road speed.
    tyre_roll: float = 0.12
    tyre_roll_at: float = 30.0
    #: Tyres scrubbing: how loud, over which band of scrub. Under
    #: ``tyre_scrub_from`` there is none; at ``tyre_scrub_at`` it is full.
    tyre_scrub: float = 0.5
    tyre_scrub_from: float = 0.25
    tyre_scrub_at: float = 3.5
    #: Wind: how loud at ``wind_at``, going as the square of speed below it.
    wind: float = 0.35
    wind_at: float = 60.0
    #: How long a gain takes to travel full scale; a gain that jumps clicks.
    settle: float = 0.15
    #: The closing speed of a full-scale impact, and the one under which a
    #: touch makes no sound.
    hit_at: float = 25.0
    hit_floor: float = 1.0


DEFAULT_TUNING = VehicleSoundTuning()


def _clamped(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else (high if value > high else value)


# Cached on the tuning and the rate: every vehicle with one tuning shares its
# clips, and nothing writes to a clip.
@cache
def motor_clip(tuning: VehicleSoundTuning = DEFAULT_TUNING,
               sample_rate: int = DEFAULT_SAMPLE_RATE) -> Clip:
    """The motor's whine, as a loop of whole cycles and with no fade, since
    a clip that ramps in and out pulses at the loop rate."""
    return synth.tone(tuning.motor_hz, tuning.loop, sample_rate=sample_rate,
                      amplitude=0.5, fade=0.0, harmonics=tuning.motor_partials)


@cache
def tyre_clip(tuning: VehicleSoundTuning = DEFAULT_TUNING,
              sample_rate: int = DEFAULT_SAMPLE_RATE) -> Clip:
    """Tarmac under a tyre: a hiss with body under it, level throughout."""
    return synth.rumble(tuning.loop, sample_rate=sample_rate, amplitude=0.6,
                        decay=0.0, attack=0.0, tone=0.0, cutoff=3000.0,
                        floor=120.0, tilt=-2.0, seed=7)


@cache
def wind_clip(tuning: VehicleSoundTuning = DEFAULT_TUNING,
              sample_rate: int = DEFAULT_SAMPLE_RATE) -> Clip:
    """Air over a body: noise lower than the tyres', level throughout."""
    return synth.rumble(tuning.loop, sample_rate=sample_rate, amplitude=0.6,
                        decay=0.0, attack=0.0, tone=0.0, cutoff=600.0,
                        tilt=-4.0, seed=11)


@cache
def impact_clip(tuning: VehicleSoundTuning = DEFAULT_TUNING,
                sample_rate: int = DEFAULT_SAMPLE_RATE) -> Clip:
    """Hitting something: a decaying bang with body, played once."""
    return synth.rumble(0.9, sample_rate=sample_rate, amplitude=0.9,
                        decay=7.0, tone=0.35, pitch=90.0, pitch_end=45.0,
                        cutoff=2200.0, tilt=-3.0, drive=1.4, seed=3)


@dataclass
class Voice:
    """One continuous sound: its gain, and its playback rate (speed and pitch
    together, as ``AudioSource.playbackRate`` means)."""

    gain: float = 0.0
    rate: float = 1.0

    def towards(self, gain: float, dt: float, settle: float) -> None:
        """Move ``gain`` no faster than full scale in ``settle`` seconds."""
        step = dt / max(settle, 1e-6)
        self.gain += max(-step, min(step, gain - self.gain))


class VehicleSound:
    """A vehicle's three continuous voices, driven once a frame.

    :meth:`update` takes the road speed, the scrub averaged over the wheels,
    the throttle from 0 to 1 and whether any wheel is on anything; the voices
    :attr:`motor`, :attr:`tyres` and :attr:`wind` then hold the gain and rate
    to play each loop at.
    """

    def __init__(self, tuning: VehicleSoundTuning = DEFAULT_TUNING) -> None:
        self.tuning = tuning
        #: The whine: its rate is the road speed, its gain mostly the load.
        self.motor = Voice(rate=tuning.motor_idle)
        #: Rolling and scrubbing together; silent off the ground.
        self.tyres = Voice()
        #: Speed alone, squared.
        self.wind = Voice()

    def update(self, dt: float, speed: float, slip: float = 0.0,
               throttle: float = 0.0, grounded: bool = True) -> None:
        """Drive the voices for ``dt`` seconds of the vehicle doing this."""
        tuning = self.tuning
        speed, slip = abs(float(speed)), abs(float(slip))
        self.motor.rate = tuning.motor_idle + speed / tuning.motor_at
        self.motor.towards(tuning.motor_load * _clamped(throttle)
                           + tuning.motor_roll * _clamped(speed / tuning.motor_at),
                           dt, tuning.settle)
        rolling = tuning.tyre_roll * _clamped(speed / tuning.tyre_roll_at)
        band = max(tuning.tyre_scrub_at - tuning.tyre_scrub_from, 1e-6)
        scrubbing = tuning.tyre_scrub * _clamped(
            (slip - tuning.tyre_scrub_from) / band)
        self.tyres.towards((rolling + scrubbing) if grounded else 0.0, dt,
                           tuning.settle)
        self.wind.towards(tuning.wind * _clamped(speed / tuning.wind_at) ** 2,
                          dt, tuning.settle)

    def hit(self, closing: float) -> float:
        """The gain of an impact at ``closing`` m/s: 0 under ``hit_floor``,
        rising to 1 at ``hit_at``."""
        closing = abs(float(closing))
        if closing < self.tuning.hit_floor:
            return 0.0
        return _clamped(closing / self.tuning.hit_at)
