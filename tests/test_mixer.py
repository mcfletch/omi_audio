"""The mixer: a fixed pool of voices summed into stereo blocks.

Every assertion here is about an array of numbers, so the whole of the audio
path below the device is tested with no device, no thread and no hardware.
"""

import math
import threading

import numpy as np
import pytest

from omi_audio import mixer as mixermodule
from omi_audio import synth
from omi_audio.clip import Clip
from omi_audio.mixer import Mixer

from support import STEP, constant, needs_tracemalloc, ramp, tracemalloc


class TestEmptyMixer:
    def test_a_mixer_with_no_voices_produces_silence(self):
        mixer = Mixer(sample_rate=8000)
        assert not mixer.mix(64).any()

    def test_the_output_block_is_stereo(self):
        assert Mixer(sample_rate=8000).mix(64).shape == (64, 2)

    def test_the_output_is_float32(self):
        assert Mixer(sample_rate=8000).mix(64).dtype == np.float32

    def test_no_voices_are_active(self):
        assert Mixer(sample_rate=8000).active_voices == 0


class TestOneVoice:
    def test_a_centred_voice_appears_equally_in_both_channels(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(constant(0.5), gain=1.0)
        block = mixer.mix(16)
        assert np.allclose(block[:, 0], block[:, 1])
        assert block[0, 0] > 0.0

    def test_a_centred_voice_is_attenuated_by_equal_power_panning(self):
        """Centre is -3 dB in each ear so the total power is the source's."""
        mixer = Mixer(sample_rate=8000)
        mixer.play(constant(1.0), gain=1.0)
        block = mixer.mix(16)
        assert block[0, 0] == pytest.approx(np.sqrt(0.5), rel=1e-3)

    def test_gain_scales_the_output(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(constant(1.0), gain=0.25)
        assert mixer.mix(16)[0, 0] == pytest.approx(0.25 * np.sqrt(0.5), rel=1e-3)

    def test_panning_hard_right_silences_the_left_channel(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(constant(1.0), pan=1.0)
        block = mixer.mix(16)
        assert block[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert block[0, 1] == pytest.approx(1.0, rel=1e-3)

    def test_the_samples_come_out_in_order(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(ramp(8), gain=1.0, pan=1.0)
        assert mixer.mix(8)[:, 1] == pytest.approx(np.arange(8) * STEP, rel=1e-4)

    def test_playing_continues_across_blocks(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(ramp(8), gain=1.0, pan=1.0)
        mixer.mix(4)
        assert mixer.mix(4)[:, 1] == pytest.approx(np.arange(4, 8) * STEP, rel=1e-4)


class TestVoiceLifetime:
    def test_a_voice_stops_at_the_end_of_a_clip_that_does_not_loop(self):
        mixer = Mixer(sample_rate=8000)
        voice = mixer.play(constant(1.0, frames=4))
        mixer.mix(4)
        assert not voice.playing
        assert mixer.active_voices == 0

    def test_the_tail_of_the_block_past_the_end_is_silent(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(constant(1.0, frames=4), pan=1.0)
        block = mixer.mix(8)
        assert block[:4, 1].min() > 0.0
        assert not block[4:, 1].any()

    def test_a_looping_voice_wraps_and_keeps_playing(self):
        mixer = Mixer(sample_rate=8000)
        voice = mixer.play(ramp(4), gain=1.0, pan=1.0, loop=True)
        assert mixer.mix(8)[:, 1] == pytest.approx(
            np.array([0, 1, 2, 3, 0, 1, 2, 3]) * STEP, rel=1e-4)
        assert voice.playing

    def test_stopping_a_voice_frees_its_slot(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        voice = mixer.play(constant(1.0, frames=10_000, sample_rate=8000), loop=True)
        voice.stop()
        assert mixer.active_voices == 0
        assert mixer.mix(8).max() == pytest.approx(0.0)

    def test_stop_all_silences_everything(self):
        mixer = Mixer(sample_rate=8000, voices=4)
        for _ in range(4):
            mixer.play(constant(1.0, frames=10_000), loop=True)
        mixer.stop_all()
        assert mixer.active_voices == 0

    def test_an_empty_clip_is_never_started(self):
        mixer = Mixer(sample_rate=8000)
        assert mixer.play(Clip([], 8000)) is None
        assert mixer.active_voices == 0


class TestPlaybackRate:
    def test_double_rate_consumes_the_clip_twice_as_fast(self):
        mixer = Mixer(sample_rate=8000)
        voice = mixer.play(constant(1.0, frames=8), rate=2.0)
        mixer.mix(4)
        assert not voice.playing

    def test_half_rate_stretches_the_clip(self):
        mixer = Mixer(sample_rate=8000)
        voice = mixer.play(constant(1.0, frames=8), rate=0.5)
        mixer.mix(8)
        assert voice.playing

    def test_a_clip_at_another_rate_is_resampled_to_the_mixer_rate(self):
        """The mixer runs at one rate; a clip's own rate is a playback ratio."""
        mixer = Mixer(sample_rate=16000)
        voice = mixer.play(constant(1.0, frames=8, sample_rate=8000))
        mixer.mix(15)
        assert voice.playing                    # 8 frames at half speed = 16
        mixer.mix(2)
        assert not voice.playing

    def test_interpolation_between_samples_is_linear(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(ramp(8), gain=1.0, pan=1.0, rate=0.5)
        assert mixer.mix(4)[:, 1] == pytest.approx(
            np.array([0.0, 0.5, 1.0, 1.5]) * STEP, rel=1e-4)

    def test_a_non_positive_rate_is_refused_rather_than_looping_forever(self):
        mixer = Mixer(sample_rate=8000)
        assert mixer.play(constant(1.0), rate=0.0) is None


class TestGainRamping:
    """A gain that jumps between blocks is a click; the mixer ramps instead."""

    def test_a_changed_gain_is_reached_by_the_end_of_the_block(self):
        mixer = Mixer(sample_rate=8000)
        voice = mixer.play(constant(1.0, frames=10_000), gain=1.0, pan=1.0)
        mixer.mix(16)
        voice.set_gain(0.0, 0.0)
        block = mixer.mix(16)
        assert block[0, 1] > 0.0                        # not an instant jump
        assert block[-1, 1] == pytest.approx(0.0, abs=1e-6)

    def test_the_ramp_is_monotonic_rather_than_a_step(self):
        mixer = Mixer(sample_rate=8000)
        voice = mixer.play(constant(1.0, frames=10_000), gain=0.0, pan=1.0)
        mixer.mix(8)
        voice.set_gain(0.0, 1.0)
        channel = mixer.mix(8)[:, 1]
        assert np.all(np.diff(channel) > 0)

    def test_a_new_voice_starts_at_its_gain_without_a_ramp(self):
        """Ramping in would soften the transient that makes an impact read."""
        mixer = Mixer(sample_rate=8000)
        mixer.play(constant(1.0), gain=1.0, pan=1.0)
        assert mixer.mix(16)[0, 1] == pytest.approx(1.0, rel=1e-3)


class TestSumming:
    def test_two_voices_add(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(constant(0.25), pan=1.0)
        mixer.play(constant(0.25), pan=1.0)
        assert mixer.mix(8)[0, 1] == pytest.approx(0.5, rel=1e-3)

    def test_the_master_gain_scales_the_whole_mix(self):
        mixer = Mixer(sample_rate=8000, master_gain=0.5)
        mixer.play(constant(1.0), pan=1.0)
        assert mixer.mix(8)[0, 1] == pytest.approx(0.5, rel=1e-3)

    def test_the_output_is_clipped_rather_than_allowed_to_wrap(self):
        mixer = Mixer(sample_rate=8000, voices=8)
        for _ in range(8):
            mixer.play(constant(1.0), pan=1.0)
        assert mixer.mix(8).max() == pytest.approx(1.0)


class TestVoicePool:
    """A fixed pool, so a scene that fires a thousand sounds costs a fixed amount."""

    def test_the_pool_size_is_the_limit_on_simultaneous_sounds(self):
        mixer = Mixer(sample_rate=8000, voices=3)
        for _ in range(3):
            assert mixer.play(constant(1.0, frames=10_000), priority=0.5) is not None
        assert mixer.active_voices == 3

    def test_a_higher_priority_sound_steals_the_lowest_priority_voice(self):
        mixer = Mixer(sample_rate=8000, voices=2)
        quiet = mixer.play(constant(1.0, frames=10_000), priority=0.1)
        mixer.play(constant(1.0, frames=10_000), priority=0.9)
        assert mixer.play(constant(1.0, frames=10_000), priority=0.5) is not None
        assert not quiet.playing

    def test_a_lower_priority_sound_is_refused_when_the_pool_is_full(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        mixer.play(constant(1.0, frames=10_000), priority=0.8)
        assert mixer.play(constant(1.0, frames=10_000), priority=0.2) is None

    def test_among_equal_priorities_the_quietest_voice_is_stolen(self):
        """Stealing the least audible sound is the least audible theft."""
        mixer = Mixer(sample_rate=8000, voices=2)
        faint = mixer.play(constant(1.0, frames=10_000), priority=0.5, gain=0.01)
        mixer.play(constant(1.0, frames=10_000), priority=0.5, gain=1.0)
        assert mixer.play(constant(1.0, frames=10_000), priority=0.5, gain=0.9) is not None
        assert not faint.playing

    def test_an_equal_priority_quieter_sound_is_refused(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        mixer.play(constant(1.0, frames=10_000), priority=0.5, gain=1.0)
        assert mixer.play(constant(1.0, frames=10_000), priority=0.5, gain=0.001) is None

    def test_a_finished_voice_is_reused_rather_than_stolen_from(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        mixer.play(constant(1.0, frames=4), priority=0.9)
        mixer.mix(8)
        assert mixer.play(constant(1.0, frames=4), priority=0.0) is not None

    def test_the_pool_reports_how_much_of_it_is_in_use(self):
        mixer = Mixer(sample_rate=8000, voices=4)
        mixer.play(constant(1.0, frames=10_000))
        assert mixer.active_voices == 1
        assert len(mixer.voices) == 4


class TestReportingThemselves:
    """``__repr__`` is what a debug overlay and a failed assertion both print."""

    def test_an_idle_voice_says_so(self):
        assert repr(Mixer(sample_rate=8000, voices=1).voices[0]) == '<Voice idle>'

    def test_a_playing_voice_names_its_clip_and_its_gains(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        mixer.play(Clip(np.ones(100, dtype='f'), 8000, name='thunder'),
                   gain=1.0, pan=1.0)
        mixer.mix(50)
        shown = repr(mixer.voices[0])
        assert 'thunder' in shown
        assert '50%' in shown

    def test_a_released_voice_goes_back_to_idle(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        mixer.play(constant(1.0, frames=4))
        mixer.mix(8)
        assert repr(mixer.voices[0]) == '<Voice idle>'

    def test_a_handle_says_whether_its_sound_is_still_going(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        handle = mixer.play(constant(1.0, frames=10_000), loop=True)
        assert repr(handle) == '<VoiceHandle playing>'
        handle.stop()
        assert repr(handle) == '<VoiceHandle finished>'

    def test_a_clip_says_what_it_holds(self):
        assert repr(Clip(np.zeros(4000, dtype='f'), 8000, name='rain')) == (
            "<Clip 'rain' 0.500s @8000Hz>")


class TestNonFiniteGains:
    """A NaN gain would survive the clipping and take every voice with it.

    Positions come out of a scenegraph, and scenegraphs produce NaN for ordinary
    reasons.  The check is at the boundary, where it costs two comparisons per
    aimed sound and nothing at all per frame.
    """

    def nan_and_a_neighbour(self):
        """One voice about to be aimed at nowhere, and one innocent bystander."""
        mixer = Mixer(sample_rate=8000, voices=2)
        broken = mixer.play(constant(1.0, frames=10_000), gain=0.5, loop=True)
        mixer.play(constant(1.0, frames=10_000), gain=0.5, pan=1.0, loop=True)
        mixer.mix(16)
        return mixer, broken

    def test_aiming_a_sound_at_a_nan_gain_silences_only_that_sound(self):
        mixer, broken = self.nan_and_a_neighbour()
        broken.set_gain(float('nan'), float('nan'))
        block = mixer.mix(64)
        assert np.isfinite(block).all(), 'one bad emitter poisoned the whole mix'
        assert np.abs(block[-1, 1]) > 0.1, 'the innocent voice was silenced too'

    def test_an_infinite_gain_is_silence_rather_than_full_scale(self):
        mixer, broken = self.nan_and_a_neighbour()
        broken.set_gain(float('inf'), float('-inf'))
        assert np.isfinite(mixer.mix(64)).all()

    def test_a_sound_started_at_a_nan_gain_never_poisons_anything(self):
        """``play_gains`` is the other door onto the audio thread."""
        mixer = Mixer(sample_rate=8000, voices=2)
        mixer.play_gains(constant(1.0, frames=10_000), float('nan'), 0.5, loop=True)
        assert np.isfinite(mixer.mix(64)).all()

    def test_a_nan_pan_is_treated_as_centre_rather_than_as_silence(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        mixer.play(constant(1.0, frames=10_000), gain=1.0, pan=float('nan'))
        block = mixer.mix(16)
        assert block[0, 0] == pytest.approx(math.sqrt(0.5), rel=1e-3)
        assert block[0, 1] == pytest.approx(math.sqrt(0.5), rel=1e-3)

    def test_a_nan_rate_is_refused_rather_than_run(self):
        """A NaN rate would put NaN in the cursor and index nowhere."""
        mixer = Mixer(sample_rate=8000)
        assert mixer.play(constant(1.0), rate=float('nan')) is None
        assert mixer.play(constant(1.0), rate=float('inf')) is None

    def test_a_nan_master_gain_is_silence_not_a_dead_mix(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(constant(1.0, frames=10_000), loop=True)
        mixer.master_gain = float('nan')
        assert np.isfinite(mixer.mix(16)).all()

    def test_the_mix_recovers_as_soon_as_the_position_is_good_again(self):
        """A transient bad frame must not leave the scene permanently silent."""
        mixer, broken = self.nan_and_a_neighbour()
        broken.set_gain(float('nan'), float('nan'))
        mixer.mix(64)
        broken.set_gain(0.5, 0.5)
        assert np.abs(mixer.mix(64)[-1, 0]) > 0.1


class TestAllocationDiscipline:
    """The audio thread must not allocate; the buffers are made once."""

    def test_every_block_is_a_view_of_the_same_buffer(self):
        mixer = Mixer(sample_rate=8000)
        assert mixer.mix(64).base is mixer.mix(32).base

    def test_a_block_larger_than_the_maximum_is_refused(self):
        mixer = Mixer(sample_rate=8000, max_block=128)
        with pytest.raises(ValueError):
            mixer.mix(129)

    @needs_tracemalloc
    def test_mixing_a_full_pool_allocates_nothing_measurable(self):
        mixer = Mixer(sample_rate=8000, voices=16, max_block=512)
        for _ in range(16):
            mixer.play(constant(0.1, frames=100_000), loop=True)
        mixer.mix(256)                                  # warm any lazy state
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for _ in range(20):
            mixer.mix(256)
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        grew = sum(entry.size_diff for entry in after.compare_to(before, 'filename'))
        assert grew < 4096, 'mixing allocated %d bytes' % (grew,)


class TestDeviceGenerator:
    """The pull generator a playback device drives."""

    def test_the_generator_yields_a_block_of_the_requested_size(self):
        mixer = Mixer(sample_rate=8000, max_block=512)
        blocks = mixer.blocks()
        next(blocks)
        data = blocks.send(64)
        assert len(memoryview(data).cast('B')) == 64 * 2 * 4

    def test_the_generator_keeps_producing(self):
        mixer = Mixer(sample_rate=8000)
        blocks = mixer.blocks()
        next(blocks)
        for _ in range(4):
            assert blocks.send(32) is not None

    def test_a_request_larger_than_the_maximum_is_served_as_silence(self):
        """A device asking for more than the mixer prepared must not raise
        inside the audio callback; it gets silence and a logged warning."""
        mixer = Mixer(sample_rate=8000, max_block=64)
        blocks = mixer.blocks()
        next(blocks)
        data = np.asarray(blocks.send(128))
        assert not data.any()

    def test_the_oversized_block_is_the_size_that_was_asked_for(self):
        """Short is not silent.

        A backend copies the bytes it is handed and leaves the rest of its
        output buffer alone -- holding the previous callback's audio, or
        nothing at all.  Half a block of zeroes followed by half a block of
        last time is a buzz, not a gap, so the block has to be full length.
        """
        mixer = Mixer(sample_rate=8000, max_block=64)
        blocks = mixer.blocks()
        next(blocks)
        block = blocks.send(1024)
        assert len(memoryview(block).cast('B')) == 1024 * 2 * 4
        assert not np.asarray(block).any()

    @needs_tracemalloc
    def test_an_oversized_block_still_costs_nothing_the_second_time(self):
        """The one sanctioned allocation happens once per size, not per block."""
        mixer = Mixer(sample_rate=8000, max_block=64)
        blocks = mixer.blocks()
        next(blocks)
        blocks.send(512)                                # grows the buffer
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for _ in range(20):
            blocks.send(512)
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        grew = sum(entry.size_diff for entry in after.compare_to(before, 'filename'))
        assert grew < 4096, 'the silence path allocated %d bytes' % (grew,)

    def test_the_mismatch_is_logged_once_and_only_once(self, caplog):
        mixer = Mixer(sample_rate=8000, max_block=64)
        blocks = mixer.blocks()
        next(blocks)
        with caplog.at_level('WARNING'):
            for _ in range(5):
                blocks.send(128)
        assert len(caplog.records) == 1


class TestMaxBlockIsFixed:
    """Every buffer was sized from it, so it cannot move afterwards."""

    def test_it_reports_what_the_mixer_was_built_for(self):
        assert Mixer(sample_rate=8000, max_block=128).max_block == 128

    def test_it_cannot_be_raised_after_construction(self):
        """Raising it would pass the guard and then return a short block --
        exactly the silent truncation ``mix`` exists to refuse."""
        mixer = Mixer(sample_rate=8000, max_block=64)
        with pytest.raises(AttributeError):
            mixer.max_block = 4096

    def test_the_guard_still_holds_after_the_attempt(self):
        mixer = Mixer(sample_rate=8000, max_block=64)
        try:
            mixer.max_block = 4096
        except AttributeError:
            pass
        with pytest.raises(ValueError):
            mixer.mix(4096)


class TestThreadSafety:
    """Starting sounds from several threads must not hand out one slot twice."""

    def test_concurrent_plays_never_share_a_voice(self):
        """Sixty-four sounds into a sixty-four voice pool must all survive.

        Two threads handed the same slot would leave fewer playing, since the
        second would silently overwrite the first.
        """
        mixer = Mixer(sample_rate=8000, voices=64)
        barrier = threading.Barrier(8)

        def start():
            barrier.wait()
            for _ in range(8):
                mixer.play(constant(1.0, frames=100_000), loop=True)

        threads = [threading.Thread(target=start) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert mixer.active_voices == 64


class TestStolenVoices:
    """A handle whose slot was recycled must steer nothing, not somebody else."""

    def test_a_stolen_handle_reports_that_it_has_stopped(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        quiet = mixer.play(constant(1.0, frames=10_000), priority=0.1, loop=True)
        mixer.play(constant(1.0, frames=10_000), priority=0.9, loop=True)
        assert not quiet.playing

    def test_a_stolen_handle_cannot_change_the_new_sounds_gain(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        quiet = mixer.play(constant(1.0, frames=10_000), priority=0.1, loop=True)
        loud = mixer.play(constant(1.0, frames=10_000), priority=0.9, loop=True,
                          gain=1.0, pan=1.0)
        quiet.set_gain(0.0, 0.0)
        assert mixer.mix(8)[0, 1] == pytest.approx(1.0, rel=1e-3)
        assert loud.playing

    def test_a_stolen_handle_cannot_stop_the_new_sound(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        quiet = mixer.play(constant(1.0, frames=10_000), priority=0.1, loop=True)
        mixer.play(constant(1.0, frames=10_000), priority=0.9, loop=True)
        quiet.stop()
        assert mixer.active_voices == 1

    def test_a_finished_handle_is_inert_rather_than_an_error(self):
        mixer = Mixer(sample_rate=8000)
        handle = mixer.play(constant(1.0, frames=4))
        mixer.mix(8)
        handle.set_gain(1.0, 1.0)
        handle.set_gain_pan(1.0, 0.5)
        handle.stop()
        assert not handle.playing
        assert handle.elapsed == 0.0

    def test_a_playing_handle_reports_how_far_it_has_got(self):
        mixer = Mixer(sample_rate=8000)
        handle = mixer.play(constant(1.0, frames=8000, sample_rate=8000))
        mixer.mix(800)
        assert handle.elapsed == pytest.approx(0.1, rel=1e-3)


class TestAHalfReleasedVoice:
    """The state the audio thread must never trip over.

    ``Voice.release`` clears ``active`` before ``clip`` so the mixing thread
    cannot see a slot that is flagged as playing and has nothing to play.  The
    mixer carries a guard for it anyway, because "cannot" here rests on the
    order of two stores in another thread.
    """

    def test_an_active_voice_with_no_clip_is_freed_rather_than_mixed(self):
        mixer = Mixer(sample_rate=8000, voices=1)
        mixer.play(constant(1.0, frames=10_000), loop=True)
        voice = mixer.voices[0]
        voice.clip = None                       # what a torn release looks like
        assert not mixer.mix(16).any()
        assert voice.active is False


class TestSteeringWithAPan:
    """``set_gain_pan`` on a sound that is actually playing."""

    def playing(self, mixer=None):
        mixer = mixer if mixer is not None else Mixer(sample_rate=8000)
        return mixer, mixer.play(constant(1.0, frames=10_000), gain=1.0, loop=True)

    def test_panning_a_live_sound_moves_it_to_that_ear(self):
        mixer, handle = self.playing()
        mixer.mix(16)
        handle.set_gain_pan(1.0, 1.0)
        block = mixer.mix(64)
        assert block[-1, 1] == pytest.approx(1.0, rel=1e-3)
        assert block[-1, 0] == pytest.approx(0.0, abs=1e-6)

    def test_the_gain_multiplies_the_pan(self):
        mixer, handle = self.playing()
        mixer.mix(16)
        handle.set_gain_pan(0.25, 0.0)
        assert mixer.mix(64)[-1, 0] == pytest.approx(0.25 * math.sqrt(0.5), rel=1e-3)

    def test_a_pan_past_hard_right_is_clamped_rather_than_wrapping_round(self):
        """Without the clamp the equal-power arc folds and the sound comes
        back out of the other ear, which is worse than doing nothing."""
        mixer, handle = self.playing()
        mixer.mix(16)
        handle.set_gain_pan(1.0, 7.0)
        block = mixer.mix(64)
        assert block[-1, 1] == pytest.approx(1.0, rel=1e-3)
        assert block[-1, 0] == pytest.approx(0.0, abs=1e-6)

    def test_a_pan_past_hard_left_is_clamped_too(self):
        mixer, handle = self.playing()
        mixer.mix(16)
        handle.set_gain_pan(1.0, -7.0)
        block = mixer.mix(64)
        assert block[-1, 0] == pytest.approx(1.0, rel=1e-3)
        assert block[-1, 1] == pytest.approx(0.0, abs=1e-6)

    def test_an_out_of_range_pan_at_the_start_is_clamped_as_well(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(constant(1.0, frames=10_000), gain=1.0, pan=99.0)
        assert mixer.mix(8)[0, 1] == pytest.approx(1.0, rel=1e-3)


def test_a_synthesised_clip_plays_through_the_same_path_as_a_decoded_one():
    mixer = Mixer(sample_rate=8000)
    mixer.play(synth.tone(440.0, 0.1, sample_rate=8000), gain=1.0, pan=1.0)
    assert np.abs(mixer.mix(256)[:, 1]).max() > 0.1


class TestMuffle:
    """A low-pass on the master bus: what being underwater sounds like."""

    def test_a_dry_mixer_leaves_the_signal_alone(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(ramp(8), gain=1.0, pan=1.0)
        assert mixer.mix(8)[:, 1] == pytest.approx(np.arange(8) * STEP, rel=1e-4)

    def test_muffling_removes_the_high_frequencies(self):
        """A 3 kHz tone at 8 kHz is near Nyquist, so a low-pass eats it."""
        loud = synth.tone(3000.0, 0.5, sample_rate=8000, amplitude=0.9, fade=0.0)
        dry, wet = [], []
        for muffle, out in ((0.0, dry), (1.0, wet)):
            mixer = Mixer(sample_rate=8000, muffle=muffle)
            mixer.play(loud, gain=1.0, pan=1.0)
            mixer.mix(512)                              # let the filter settle
            out.append(float(np.abs(mixer.mix(512)[:, 1]).max()))
        assert wet[0] < dry[0] * 0.25

    def test_muffling_leaves_the_low_frequencies_alone(self):
        quiet = synth.tone(100.0, 0.5, sample_rate=8000, amplitude=0.9, fade=0.0)
        levels = []
        for muffle in (0.0, 1.0):
            mixer = Mixer(sample_rate=8000, muffle=muffle)
            mixer.play(quiet, gain=1.0, pan=1.0)
            mixer.mix(512)
            levels.append(float(np.abs(mixer.mix(512)[:, 1]).max()))
        assert levels[1] == pytest.approx(levels[0], rel=0.15)

    def test_the_muffle_can_be_changed_while_playing(self):
        mixer = Mixer(sample_rate=8000)
        mixer.play(synth.tone(3000.0, 1.0, sample_rate=8000, fade=0.0),
                   gain=1.0, pan=1.0, loop=True)
        mixer.mix(256)
        mixer.muffle = 1.0
        mixer.mix(256)
        assert float(np.abs(mixer.mix(256)[:, 1]).max()) < 0.2

    def test_a_partial_muffle_lands_between_dry_and_wet(self):
        """It is a float rather than a flag so an application can fade it in.

        Every other test here uses 0 or 1, which cannot tell a blend from a
        threshold: "anything above zero is fully wet" passes both.
        """
        loud = synth.tone(3000.0, 0.5, sample_rate=8000, amplitude=0.9, fade=0.0)
        levels = {}
        for amount in (0.0, 0.5, 1.0):
            mixer = Mixer(sample_rate=8000, muffle=amount)
            mixer.play(loud, gain=1.0, pan=1.0, loop=True)
            mixer.mix(512)                              # let the filter settle
            levels[amount] = float(np.abs(mixer.mix(512)[:, 1]).max())
        assert levels[1.0] < levels[0.5] < levels[0.0]
        # Halfway in the blend is halfway between the two levels, since the
        # blend is linear in amplitude and the tone's level is steady.
        assert levels[0.5] == pytest.approx(
            (levels[0.0] + levels[1.0]) / 2.0, rel=0.1)

    def test_the_cutoff_is_a_frequency_not_a_fraction_of_the_sample_rate(self):
        """What makes the muffle audible at the rate anything actually runs at.

        A filter specified as a fraction of Nyquist measures the same at every
        sample rate and sounds like nothing at a real one: 3 kHz is most of the
        way to Nyquist at 8 kHz and a fifteenth of the way there at 44.1 kHz.
        The corner is in hertz, so it lands in the same musical place either way.
        """
        levels = {}
        for rate in (8000, 44100):
            tone = synth.tone(mixermodule.MUFFLE_CUTOFF_HZ, 0.5, sample_rate=rate,
                              amplitude=0.9, fade=0.0)
            got = []
            for muffle in (0.0, 1.0):
                mixer = Mixer(sample_rate=rate, muffle=muffle)
                mixer.play(tone, gain=1.0, pan=1.0, loop=True)
                mixer.mix(2048)                         # let the filter settle
                got.append(float(np.abs(mixer.mix(2048)[:, 1]).max()))
            levels[rate] = 20.0 * np.log10(got[1] / got[0])
        for rate, drop in levels.items():
            assert -6.0 < drop < -1.0, (
                'at %d Hz the corner frequency should sit near -3 dB, got %.2f dB'
                % (rate, drop))
        assert abs(levels[8000] - levels[44100]) < 2.0, (
            'the corner should be the same frequency at every rate: %r' % (levels,))

    @pytest.mark.parametrize('frequency,least', [
        (1000.0, 8.0), (2000.0, 12.0), (4000.0, 20.0),
    ])
    def test_muffling_is_audible_at_the_rate_content_is_authored_at(
            self, frequency, least):
        """The muffle has to be heard, and "heard" is decibels at 44.1 kHz.

        A change under about 1 dB is inaudible, so a filter that only bites near
        Nyquist is a filter nobody can hear on a real sound: middle-register
        content is where the energy of most sounds is.
        """
        tone = synth.tone(frequency, 0.5, sample_rate=44100, amplitude=0.9, fade=0.0)
        got = []
        for muffle in (0.0, 1.0):
            mixer = Mixer(sample_rate=44100, muffle=muffle)
            mixer.play(tone, gain=1.0, pan=1.0, loop=True)
            mixer.mix(2048)
            got.append(float(np.abs(mixer.mix(2048)[:, 1]).max()))
        drop = -20.0 * np.log10(got[1] / got[0])
        assert drop > least, (
            '%g Hz should lose at least %g dB when muffled, lost %.2f dB'
            % (frequency, least, drop))

    def test_the_bass_still_comes_through(self):
        """Underwater is the top gone, not the whole mix turned down.  A muffle
        that attenuated everything equally would be a volume knob."""
        tone = synth.tone(100.0, 0.5, sample_rate=44100, amplitude=0.9, fade=0.0)
        got = []
        for muffle in (0.0, 1.0):
            mixer = Mixer(sample_rate=44100, muffle=muffle)
            mixer.play(tone, gain=1.0, pan=1.0, loop=True)
            mixer.mix(2048)
            got.append(float(np.abs(mixer.mix(2048)[:, 1]).max()))
        assert got[1] == pytest.approx(got[0], rel=0.1)

    def test_the_muffle_is_clamped_to_a_sensible_range(self):
        mixer = Mixer(sample_rate=8000)
        mixer.muffle = 5.0
        assert mixer.muffle == 1.0
        mixer.muffle = -3.0
        assert mixer.muffle == 0.0

    @needs_tracemalloc
    def test_muffling_does_not_break_the_allocation_discipline(self):
        mixer = Mixer(sample_rate=8000, voices=4, max_block=512, muffle=0.6)
        for _ in range(4):
            mixer.play(constant(0.1, frames=100_000), loop=True)
        mixer.mix(256)
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for _ in range(20):
            mixer.mix(256)
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        grew = sum(entry.size_diff for entry in after.compare_to(before, 'filename'))
        assert grew < 4096, 'muffled mixing allocated %d bytes' % (grew,)

    def test_the_filter_carries_across_block_boundaries(self):
        """A filter reset each block would tick at the seam."""
        mixer = Mixer(sample_rate=8000, muffle=1.0)
        mixer.play(constant(1.0, frames=10_000), gain=1.0, pan=1.0)
        first = mixer.mix(32)[:, 1].copy()
        second = mixer.mix(32)[:, 1]
        assert second[0] == pytest.approx(first[-1], abs=0.05)
