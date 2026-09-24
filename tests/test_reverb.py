"""The reverb over the whole mix: what a tunnel or a cellar gives back."""

import numpy as np
import pytest

from omi_audio import synth
from omi_audio.clip import Clip
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
        from omi_audio.device import NullDevice
        from omi_audio.engine import AudioEngine
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
