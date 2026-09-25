"""A vehicle's sound, out of arithmetic on what its physics already knows.

Road speed, how much the tyres are scrubbing, how hard the motor is asked to
work and whether any wheel is on anything: four numbers a frame in, gains and
playback rates out, tested here with no device and no audio thread.
"""
import numpy as np
import pytest

from omi_audio.vehicle import (
    VehicleSound,
    VehicleSoundTuning,
    impact_clip,
    motor_clip,
    tyre_clip,
    wind_clip,
)

FRAME = 1.0 / 60.0
TUNING = VehicleSoundTuning()


def settled(sound, seconds=2.0, **state):
    """Hold one set of conditions until the gains stop moving."""
    for _ in range(int(seconds / FRAME)):
        sound.update(FRAME, **state)
    return sound


def driving(speed=30.0, slip=0.0, throttle=0.0, grounded=True):
    return {'speed': speed, 'slip': slip, 'throttle': throttle,
            'grounded': grounded}


def rms(samples):
    return float(np.sqrt((np.asarray(samples) ** 2).mean()))


class TestTheClipsLoop:
    """A continuous sound is a short clip played round and round, so the join
    has to be inaudible."""

    def test_the_motor_is_a_whole_number_of_cycles(self) -> None:
        clip = motor_clip(TUNING, sample_rate=8000)
        cycles = TUNING.motor_hz * len(clip.samples) / clip.sample_rate
        assert cycles == pytest.approx(round(cycles))

    def test_and_it_does_not_fade_at_the_ends(self) -> None:
        samples = np.abs(np.asarray(motor_clip(TUNING, 8000).samples))
        edge = max(float(samples[:50].max()), float(samples[-50:].max()))
        assert edge > float(samples.max()) * 0.5

    @pytest.mark.parametrize('made', [tyre_clip, wind_clip])
    def test_the_noises_hold_their_level_throughout(self, made) -> None:
        samples = np.asarray(made(TUNING, 8000).samples)
        third = len(samples) // 3
        assert rms(samples[-third:]) == pytest.approx(rms(samples[:third]),
                                                      rel=0.25)

    def test_an_impact_decays(self) -> None:
        samples = np.abs(np.asarray(impact_clip(TUNING, 8000).samples))
        assert float(samples[-200:].max()) < float(samples[:200].max()) * 0.2

    def test_the_wind_is_lower_than_the_tyres(self) -> None:
        def centroid(clip):
            samples = np.asarray(clip.samples)
            power = np.abs(np.fft.rfft(samples)) ** 2
            freqs = np.fft.rfftfreq(len(samples), 1.0 / clip.sample_rate)
            return float((power * freqs).sum() / power.sum())
        assert centroid(wind_clip(TUNING, 8000)) < centroid(
            tyre_clip(TUNING, 8000))

    def test_a_clip_is_made_once_for_a_tuning(self) -> None:
        assert motor_clip(TUNING, 8000) is motor_clip(TUNING, 8000)

    def test_and_again_for_another(self) -> None:
        other = VehicleSoundTuning(motor_hz=300.0)
        assert motor_clip(other, 8000) is not motor_clip(TUNING, 8000)


class TestAParkedVehicleIsQuiet:
    def test_nothing_plays_when_nothing_is_happening(self) -> None:
        sound = settled(VehicleSound(), **driving(speed=0.0))
        for voice in (sound.motor, sound.tyres, sound.wind):
            assert voice.gain == pytest.approx(0.0, abs=1e-3)

    def test_but_the_motor_answers_the_throttle_before_it_moves(self) -> None:
        sound = settled(VehicleSound(), **driving(speed=0.0, throttle=1.0))
        assert sound.motor.gain > 0.05


class TestSpeedIsHeard:
    def test_the_motor_rises_in_pitch_with_speed(self) -> None:
        slow = settled(VehicleSound(), **driving(speed=10.0)).motor.rate
        fast = settled(VehicleSound(), **driving(speed=50.0)).motor.rate
        assert fast > slow * 1.5

    def test_and_never_falls_to_nothing_at_a_standstill(self) -> None:
        """A rate of zero is a clip that does not advance: a click, then
        silence."""
        rate = settled(VehicleSound(), **driving(speed=0.0)).motor.rate
        assert rate == pytest.approx(TUNING.motor_idle)

    def test_the_wind_goes_as_the_square_of_speed(self) -> None:
        at30 = settled(VehicleSound(), **driving(speed=30.0)).wind.gain
        at60 = settled(VehicleSound(), **driving(speed=60.0)).wind.gain
        assert at60 == pytest.approx(at30 * 4.0, rel=0.01)

    def test_the_tyres_are_heard_rolling(self) -> None:
        assert settled(VehicleSound(), **driving(speed=40.0)).tyres.gain > 0.02


class TestScrubbing:
    """The band ``tyre_scrub_from`` to ``tyre_scrub_at`` is where a tyre is
    heard letting go; a vehicle's own tuning says where that is."""

    def test_under_the_band_it_is_rolling_alone(self) -> None:
        under = settled(VehicleSound(), **driving(
            slip=TUNING.tyre_scrub_from * 0.9))
        rolling = settled(VehicleSound(), **driving(slip=0.0))
        assert under.tyres.gain == pytest.approx(rolling.tyres.gain, abs=1e-3)

    def test_at_the_top_of_it_the_scrub_is_full(self) -> None:
        top = settled(VehicleSound(), **driving(slip=TUNING.tyre_scrub_at))
        rolling = settled(VehicleSound(), **driving(slip=0.0))
        assert top.tyres.gain - rolling.tyres.gain == pytest.approx(
            TUNING.tyre_scrub, abs=1e-3)

    def test_the_band_is_the_tuning_s(self) -> None:
        early = VehicleSoundTuning(tyre_scrub_from=0.0, tyre_scrub_at=1.0)
        assert settled(VehicleSound(early), **driving(slip=1.0)).tyres.gain \
            > settled(VehicleSound(), **driving(slip=1.0)).tyres.gain

    def test_a_vehicle_in_the_air_makes_no_tyre_noise(self) -> None:
        flying = settled(VehicleSound(),
                         **driving(speed=40.0, slip=9.0, grounded=False))
        assert flying.tyres.gain == pytest.approx(0.0, abs=1e-3)
        assert flying.wind.gain > 0.05


class TestNothingSteps:
    """A gain that jumps is a click."""

    def test_no_single_frame_moves_a_gain_far(self) -> None:
        sound = settled(VehicleSound(), **driving(speed=60.0, slip=9.0))
        worst = 0.0
        for state in (driving(speed=0.0), driving(speed=60.0, slip=9.0),
                      driving(speed=0.0, grounded=False)):
            for _ in range(30):
                was = sound.tyres.gain
                sound.update(FRAME, **state)
                worst = max(worst, abs(sound.tyres.gain - was))
        assert worst <= FRAME / TUNING.settle + 1e-9

    def test_a_long_frame_does_not_overshoot(self) -> None:
        sound = VehicleSound()
        sound.update(2.0, **driving(speed=0.0, throttle=1.0))
        steady = settled(VehicleSound(), **driving(speed=0.0, throttle=1.0))
        assert sound.motor.gain == pytest.approx(steady.motor.gain, abs=1e-6)


class TestHittingSomething:
    def test_a_harder_one_is_louder(self) -> None:
        assert VehicleSound().hit(20.0) > VehicleSound().hit(5.0) > 0.0

    def test_a_nudge_is_silent(self) -> None:
        assert VehicleSound().hit(TUNING.hit_floor * 0.5) == 0.0

    def test_and_it_is_never_louder_than_full_scale(self) -> None:
        assert VehicleSound().hit(500.0) == 1.0
