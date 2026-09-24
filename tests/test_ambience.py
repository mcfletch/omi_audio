"""Looping ambience: surf and birdsong, made without a recording."""

import numpy as np
import pytest

from omi_audio import synth

RATE = 22050


def seam(clip):
    """How big the step from the last sample back to the first is, against a typical step."""
    samples = clip.samples.astype('d')
    steps = np.abs(np.diff(samples))
    return abs(samples[0] - samples[-1]) / max(float(np.percentile(steps, 99)), 1e-9)


def share_below(clip, hertz):
    """What share of the clip's energy is below ``hertz``."""
    spectrum = np.abs(np.fft.rfft(clip.samples.astype('d'))) ** 2
    freqs = np.fft.rfftfreq(clip.frames, 1.0 / clip.sample_rate)
    return float(spectrum[freqs < hertz].sum() / spectrum.sum())


@pytest.mark.parametrize('make', [synth.surf, synth.birdsong])
class TestLoops:
    def test_it_is_as_long_as_asked(self, make):
        clip = make(4.0, sample_rate=RATE, seed=3)
        assert clip.frames == 4 * RATE

    def test_it_peaks_at_the_amplitude_asked_for(self, make):
        clip = make(4.0, sample_rate=RATE, seed=3, amplitude=0.3)
        assert float(np.abs(clip.samples).max()) == pytest.approx(0.3, rel=1e-5)

    def test_it_loops_without_a_click(self, make):
        assert seam(make(4.0, sample_rate=RATE, seed=3)) < 1.0

    def test_a_seed_makes_the_same_clip(self, make):
        one = make(2.0, sample_rate=RATE, seed=11)
        two = make(2.0, sample_rate=RATE, seed=11)
        assert np.array_equal(one.samples, two.samples)

    def test_another_seed_makes_another_clip(self, make):
        one = make(2.0, sample_rate=RATE, seed=11)
        two = make(2.0, sample_rate=RATE, seed=12)
        assert not np.array_equal(one.samples, two.samples)

    def test_nothing_long_is_an_empty_clip(self, make):
        assert make(0.0, sample_rate=RATE).frames == 0


class TestSurf:
    def test_surf_is_mostly_below_a_kilohertz(self):
        assert share_below(synth.surf(6.0, sample_rate=RATE, seed=2), 1000.0) > 0.8

    def test_the_waves_rise_and_fall(self):
        clip = synth.surf(12.0, sample_rate=RATE, seed=2, waves=3)
        loudness = np.sqrt(np.convolve(clip.samples.astype('d') ** 2,
                                       np.ones(RATE // 4) / (RATE // 4), 'valid'))
        assert loudness.max() > 2.0 * loudness.min()


class TestBirdsong:
    def test_birdsong_is_mostly_above_two_kilohertz(self):
        clip = synth.birdsong(6.0, sample_rate=RATE, seed=2)
        assert share_below(clip, 2000.0) < 0.1

    def test_there_are_gaps_between_the_phrases(self):
        clip = synth.birdsong(10.0, sample_rate=RATE, seed=2, phrases=0.3)
        quiet = np.abs(clip.samples) < 0.001
        assert quiet.mean() > 0.3
