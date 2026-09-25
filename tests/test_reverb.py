"""The reverb over the whole mix: what a tunnel or a cellar gives back."""

from math import gcd

import numpy as np
import pytest

from omi_audio import synth
from omi_audio.clip import Clip
from omi_audio.device import NullDevice
from omi_audio.engine import AudioEngine
from omi_audio.mixer import Mixer
from omi_audio.reverb import COMB_DELAYS, Reverb, feedback_for

from support import RATE, constant, needs_tracemalloc, tracemalloc


def click(frames=4000):
    """One full-scale sample followed by silence: an impulse to hear a room with."""
    samples = np.zeros(frames, dtype='f')
    samples[0] = 1.0
    return Clip(samples, RATE)


def played(mixer, blocks, frames=256):
    """``blocks`` blocks of the mix, joined into one ``(n, 2)`` array."""
    return np.concatenate([mixer.mix(frames).copy() for _ in range(blocks)])


class TestFeedback:
    def test_a_comb_falls_sixty_decibels_over_the_decay(self):
        delay, decay = 0.03, 1.5
        passes = decay / delay
        assert feedback_for(delay, decay) ** passes == pytest.approx(1e-3)

    def test_a_longer_decay_feeds_back_more(self):
        assert feedback_for(0.03, 3.0) > feedback_for(0.03, 1.0)

    def test_no_decay_is_no_feedback(self):
        assert feedback_for(0.03, 0.0) == 0.0


class TestLevel:
    def test_a_mixer_starts_with_no_reverb(self):
        mixer = Mixer(sample_rate=RATE)
        assert mixer.reverb.level == 0.0
        assert not mixer.reverb.active

    def test_with_no_level_the_mix_is_untouched(self):
        dry = Mixer(sample_rate=RATE)
        dry.play(click())
        wet = Mixer(sample_rate=RATE)
        wet.reverb.decay = 2.0
        wet.play(click())
        assert np.array_equal(played(dry, 8), played(wet, 8))

    def test_a_click_comes_back_after_the_shortest_comb(self):
        mixer = Mixer(sample_rate=RATE)
        mixer.reverb.level = 1.0
        mixer.play(click())
        heard = played(mixer, 8)
        first = int(round(min(COMB_DELAYS) * RATE))
        assert not heard[2:first - 1].any()
        assert np.abs(heard[first:first + 4]).max() > 0.01

    def test_the_tail_dies_away(self):
        mixer = Mixer(sample_rate=RATE)
        mixer.reverb.level = 1.0
        mixer.reverb.decay = 0.5
        mixer.play(click(frames=RATE * 2))
        heard = np.abs(played(mixer, 64))
        early = heard[RATE // 20:RATE // 5].max()
        late = heard[RATE:RATE + RATE // 5].max()
        assert late < early * 0.01

    def test_a_longer_decay_lasts_longer(self):
        def energy_after(decay):
            mixer = Mixer(sample_rate=RATE)
            mixer.reverb.level = 1.0
            mixer.reverb.decay = decay
            mixer.play(click(frames=RATE * 2))
            heard = played(mixer, 64)
            return float((heard[RATE // 2:] ** 2).sum())
        assert energy_after(2.0) > energy_after(0.5) * 10

    def test_the_ears_hear_different_tails(self):
        mixer = Mixer(sample_rate=RATE)
        mixer.reverb.level = 1.0
        mixer.play(click())
        heard = played(mixer, 8)
        assert not np.allclose(heard[400:, 0], heard[400:, 1])

    def test_the_level_arrives_over_a_block_rather_than_at_once(self):
        mixer = Mixer(sample_rate=RATE)
        mixer.play(constant(0.2, frames=RATE * 4), loop=True)
        mixer.mix(256)
        mixer.reverb.level = 1.0
        block = mixer.mix(256).copy()
        steps = np.abs(np.diff(block[:, 0]))
        assert steps.max() < 0.05

    def test_turning_it_off_clears_what_it_held(self):
        mixer = Mixer(sample_rate=RATE)
        mixer.reverb.level = 1.0
        mixer.reverb.decay = 3.0
        mixer.play(click())
        played(mixer, 4)
        mixer.reverb.level = 0.0
        played(mixer, 2)
        assert not mixer.reverb.active
        mixer.reverb.level = 1.0
        assert not played(mixer, 4).any()

    def test_damping_takes_the_top_off_the_tail(self):
        def brightness(damping):
            mixer = Mixer(sample_rate=RATE)
            mixer.reverb.level = 1.0
            mixer.reverb.decay = 2.0
            mixer.reverb.damping = damping
            mixer.play(click())
            tail = played(mixer, 32)[RATE // 4:, 0]
            return float((np.diff(tail) ** 2).sum() / max((tail ** 2).sum(), 1e-12))
        assert brightness(1.0) < brightness(0.0)

    def test_a_non_finite_setting_is_silence_rather_than_noise(self):
        mixer = Mixer(sample_rate=RATE)
        mixer.reverb.level = float('nan')
        mixer.reverb.decay = float('inf')
        mixer.play(click())
        assert np.isfinite(played(mixer, 4)).all()

    def test_blocks_of_any_size_give_the_same_tail(self):
        def heard(frames):
            mixer = Mixer(sample_rate=RATE, max_block=2048)
            mixer.reverb.level = 1.0
            mixer.reverb.decay = 1.0
            mixer.mix(frames)                   # the level has arrived
            mixer.play(click())
            return played(mixer, 4096 // frames, frames)
        assert np.allclose(heard(64), heard(2048), atol=1e-6)


class TestEngine:
    def test_the_engine_hands_out_its_mixers_reverb(self):
        engine = AudioEngine(device=NullDevice(sample_rate=RATE))
        try:
            assert engine.reverb is engine.mixer.reverb
            assert isinstance(engine.reverb, Reverb)
        finally:
            engine.close()


@needs_tracemalloc
def test_reverberating_a_block_allocates_nothing_measurable():
    mixer = Mixer(sample_rate=RATE, max_block=512)
    mixer.reverb.level = 1.0
    mixer.play(synth.noise(4.0, sample_rate=RATE, seed=1), loop=True)
    mixer.mix(256)
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(20):
        mixer.mix(256)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    grew = sum(entry.size_diff for entry in after.compare_to(before, 'filename'))
    assert grew < 4096, 'reverberating allocated %d bytes' % (grew,)


class TestTheCombs:
    @pytest.mark.parametrize('rate', [22050, 44100, 48000, 96000])
    def test_no_two_delays_share_a_factor_in_samples(self, rate):
        delays = Reverb(rate, 256)._delays.ravel().tolist()  # noqa: SLF001 - the tuning under test
        assert len(set(delays)) == len(delays)
        for i, first in enumerate(delays):
            for second in delays[i + 1:]:
                assert gcd(first, second) == 1, (first, second)

    def test_the_delays_stay_near_the_ones_asked_for(self):
        reverb = Reverb(48000, 256)
        asked = np.array(COMB_DELAYS) * 48000
        assert np.abs(reverb._delays[0] - asked).max() < 20  # noqa: SLF001 - the tuning under test


class TestTheTail:
    def test_it_is_dense_rather_than_a_train_of_echoes(self):
        mixer = Mixer(sample_rate=RATE)
        mixer.reverb.level = 1.0
        mixer.reverb.decay = 2.0
        mixer.play(click())
        heard = played(mixer, 32)[:, 0]
        first = int(round(min(COMB_DELAYS) * RATE))
        early = heard[first:first + RATE // 10]
        assert np.mean(np.abs(early) > 1e-5) > 0.5

    @pytest.mark.parametrize('decay', [0.5, 1.2, 4.0])
    def test_at_full_level_it_is_about_as_loud_as_the_sound(self, decay):
        def mix(level):
            mixer = Mixer(sample_rate=RATE)
            mixer.reverb.level = level
            mixer.reverb.decay = decay
            mixer.play(synth.noise(4.0, sample_rate=RATE, seed=3, amplitude=0.2), loop=True)
            return played(mixer, 256)[RATE:]
        dry = mix(0.0)
        wet = mix(1.0) - dry
        ratio = float(np.sqrt((wet ** 2).mean() / (dry ** 2).mean()))
        assert 0.5 < ratio < 1.5

    @pytest.mark.parametrize('decay, headroom', [(1.2, 3.5), (4.0, 4.5)])
    def test_a_tone_on_a_combs_pitch_stays_within_the_headroom(self, decay, headroom):
        delay = int(Reverb(RATE, 256)._delays[0, 0])  # noqa: SLF001 - the tuning under test
        pitch = RATE / delay * round(440.0 * delay / RATE)
        t = np.arange(RATE * 2) / RATE
        tone = Clip((0.2 * np.sin(2 * np.pi * pitch * t)).astype('f'), RATE)

        def mix(level):
            mixer = Mixer(sample_rate=RATE)
            mixer.reverb.level = level
            mixer.reverb.decay = decay
            mixer.reverb.damping = 0.0
            mixer.play(tone, loop=True)
            return played(mixer, 256)[RATE:]
        wet = mix(1.0) - mix(0.0)
        assert np.abs(wet).max() < 0.2 * headroom

    def test_a_silent_tail_is_flushed_to_zero(self):
        mixer = Mixer(sample_rate=RATE)
        mixer.reverb.level = 1.0
        mixer.reverb.decay = 0.3
        mixer.play(click())
        played(mixer, 400)
        lines = mixer.reverb._lines  # noqa: SLF001 - the comb state under test
        assert not lines.any()
