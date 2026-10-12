"""The processing graph's wiring: order, channel mixing, gain, time and outputs."""

import math

import numpy as np
import pytest

from omi_audio import graph
from omi_audio.mixer import Mixer

from support import RATE, needs_tracemalloc, tracemalloc


class Constant(graph.Node):
    """A source of fixed values, one per channel."""

    def __init__(self, *values, **named):
        super().__init__(**named)
        self.values = values

    def output_channels(self, _computed):
        return len(self.values)

    def process(self, _block, out, frames):
        for channel, value in enumerate(self.values):
            out[channel, :frames] = value


def rendered(node, frames=32, **named):
    """``node``'s output for one block, rendered as the only output of a graph."""
    made = graph.Graph(RATE, max_block=256, **named)
    made.add_output(node)
    made.render(frames)
    return node.output(frames).copy()


class TestMixMatrix:
    def test_the_same_count_is_the_identity(self):
        assert np.array_equal(graph.mix_matrix(2, 2), np.eye(2))

    def test_mono_up_to_stereo_copies(self):
        assert np.array_equal(graph.mix_matrix(1, 2), [[1.0], [1.0]])

    def test_stereo_down_to_mono_averages(self):
        assert np.allclose(graph.mix_matrix(2, 1), [[0.5, 0.5]])

    def test_quad_down_to_stereo(self):
        assert np.allclose(graph.mix_matrix(4, 2), [[0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5]])

    def test_surround_down_to_stereo_folds_the_centre_and_surrounds(self):
        half = math.sqrt(0.5)
        assert np.allclose(graph.mix_matrix(6, 2),
                           [[1, 0, half, 0, half, 0], [0, 1, half, 0, 0, half]])

    def test_surround_down_to_mono(self):
        half = math.sqrt(0.5)
        assert np.allclose(graph.mix_matrix(6, 1), [[half, half, 1, 0, 0.5, 0.5]])

    def test_mono_up_to_surround_is_the_centre(self):
        assert np.allclose(graph.mix_matrix(1, 6)[:, 0], [0, 0, 1, 0, 0, 0])

    def test_discrete_fills_and_drops(self):
        assert np.array_equal(graph.mix_matrix(1, 2, 'discrete'), [[1.0], [0.0]])
        assert np.array_equal(graph.mix_matrix(3, 2, 'discrete'), [[1, 0, 0], [0, 1, 0]])

    def test_an_unlisted_layout_mixes_discretely(self):
        assert np.array_equal(graph.mix_matrix(3, 5), graph.mix_matrix(3, 5, 'discrete'))

    def test_an_unknown_interpretation_is_refused(self):
        with pytest.raises(ValueError):
            graph.mix_matrix(1, 2, 'sideways')


class TestChannelCount:
    def test_max_takes_the_widest_input(self):
        gain = graph.Gain()
        gain.connect(Constant(1.0))
        gain.connect(Constant(0.5, 0.25))
        assert rendered(gain).shape[0] == 2

    def test_inputs_are_summed_after_mixing_up(self):
        gain = graph.Gain()
        gain.connect(Constant(1.0))
        gain.connect(Constant(0.5, 0.25))
        block = rendered(gain)
        assert np.allclose(block[0], 1.5) and np.allclose(block[1], 1.25)

    def test_clamped_max_limits_the_count(self):
        gain = graph.Gain(channel_count=1, channel_count_mode='clamped-max')
        gain.connect(Constant(0.5, 0.25))
        block = rendered(gain)
        assert block.shape[0] == 1 and np.allclose(block, 0.375)

    def test_explicit_uses_the_count_given(self):
        gain = graph.Gain(channel_count=4, channel_count_mode='explicit',
                          channel_interpretation='discrete')
        gain.connect(Constant(1.0))
        block = rendered(gain)
        assert block.shape[0] == 4 and np.allclose(block[1:], 0.0)

    def test_a_node_with_no_inputs_is_one_silent_channel(self):
        block = rendered(graph.Gain())
        assert block.shape[0] == 1 and not block.any()

    def test_the_count_is_capped_at_the_graph_maximum(self):
        gain = graph.Gain(channel_count=12, channel_count_mode='explicit')
        gain.connect(Constant(1.0))
        assert rendered(gain, max_channels=8).shape[0] == 8

    def test_one_channel_of_a_source_is_an_input_of_its_own(self):
        gain = graph.Gain()
        gain.connect(Constant(0.5, 0.25), channel=1)
        block = rendered(gain)
        assert block.shape[0] == 1 and np.allclose(block, 0.25)

    def test_a_channel_the_source_does_not_have_is_silence(self):
        gain = graph.Gain()
        gain.connect(Constant(0.5), channel=3)
        assert not rendered(gain).any()

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError):
            graph.Gain(channel_count_mode='most')


class TestGain:
    def test_gain_scales_the_output(self):
        gain = graph.Gain(gain=0.5)
        gain.connect(Constant(1.0))
        assert np.allclose(rendered(gain), 0.5)

    def test_a_changed_gain_is_ramped_across_the_next_block(self):
        gain = graph.Gain(gain=1.0)
        gain.connect(Constant(1.0))
        made = graph.Graph(RATE, max_block=64)
        made.add_output(gain)
        made.render(16)
        gain.gain = 0.0
        made.render(16)
        block = gain.output(16)[0]
        assert block[0] > 0.9 and block[-1] == pytest.approx(0.0)
        assert np.all(np.diff(block) < 0)

    def test_a_settled_gain_is_not_ramped(self):
        gain = graph.Gain(gain=1.0)
        gain.connect(Constant(1.0))
        gain.gain = 0.25
        gain.settle()
        assert np.allclose(rendered(gain), 0.25)

    def test_a_gain_that_is_not_a_number_is_silence(self):
        gain = graph.Gain(gain=float('nan'))
        gain.connect(Constant(1.0))
        assert not rendered(gain).any()

    def test_a_disabled_node_passes_its_input_unchanged(self):
        gain = graph.Gain(gain=0.25)
        gain.enabled = False
        gain.connect(Constant(1.0))
        assert np.allclose(rendered(gain), 1.0)

    def test_a_disabled_source_is_silent(self):
        source = Constant(1.0)
        source.enabled = False
        assert not rendered(source).any()


class TestTime:
    def test_a_node_that_is_not_playing_is_silent(self):
        gain = graph.Gain()
        gain.connect(Constant(1.0))
        gain.playing = False
        assert not rendered(gain).any()

    def test_a_paused_node_is_silent(self):
        gain = graph.Gain()
        gain.connect(Constant(1.0))
        gain.paused = True
        assert not rendered(gain).any()

    def test_a_paused_node_keeps_its_channel_count(self):
        gain = graph.Gain()
        gain.connect(Constant(1.0, 1.0))
        made = graph.Graph(RATE, max_block=256)
        made.add_output(gain)
        made.render(32)
        gain.paused = True
        made.render(32)
        assert gain.output(32).shape[0] == 2 and not gain.output(32).any()

    def test_a_node_paused_before_it_is_heard_has_one_channel(self):
        gain = graph.Gain()
        gain.connect(Constant(1.0, 1.0))
        gain.paused = True
        assert rendered(gain).shape[0] == 1


class TestGraph:
    def test_an_empty_graph_is_not_active(self):
        assert not graph.Graph(RATE).active

    def test_a_graph_with_an_output_is_active(self):
        made = graph.Graph(RATE)
        made.add_output(Constant(1.0))
        assert made.active

    def test_outputs_are_summed_into_stereo(self):
        made = graph.Graph(RATE, max_block=64)
        made.add_output(Constant(1.0))
        made.add_output(Constant(0.25, 0.5))
        out = np.zeros((16, 2), dtype=np.float32)
        made.render(16, out)
        assert np.allclose(out[:, 0], 1.25) and np.allclose(out[:, 1], 1.5)

    def test_a_shared_input_is_rendered_once(self):
        calls = []

        class Counting(Constant):
            def process(self, _block, out, frames):
                calls.append(frames)
                super().process(_block, out, frames)

        shared = Counting(1.0)
        left, right = graph.Gain(), graph.Gain()
        left.connect(shared)
        right.connect(shared)
        made = graph.Graph(RATE)
        made.add_output(left)
        made.add_output(right)
        made.render(8)
        assert calls == [8]

    def test_inputs_render_before_what_reads_them(self):
        source = Constant(1.0)
        middle = graph.Gain(gain=0.5).connect(source)
        last = graph.Gain(gain=0.5).connect(middle)
        assert np.allclose(rendered(last), 0.25)

    def test_a_cycle_is_refused(self):
        first, second = graph.Gain(), graph.Gain()
        first.connect(second)
        second.connect(first)
        made = graph.Graph(RATE)
        with pytest.raises(ValueError):
            made.add_output(first)

    def test_a_rewired_node_is_heard_after_compiling(self):
        made = graph.Graph(RATE)
        gain = graph.Gain()
        made.add_output(gain)
        gain.connect(Constant(1.0))
        made.render(8)
        assert not gain.output(8).any()
        made.compile()
        made.render(8)
        assert np.allclose(gain.output(8), 1.0)

    def test_a_disconnected_input_is_gone(self):
        source = Constant(1.0)
        gain = graph.Gain().connect(source)
        gain.disconnect(source)
        assert not rendered(gain).any()

    def test_disconnecting_everything(self):
        gain = graph.Gain().connect(Constant(1.0)).connect(Constant(2.0))
        gain.disconnect()
        assert gain.inputs == []

    def test_a_removed_output_is_no_longer_heard(self):
        made = graph.Graph(RATE)
        source = Constant(1.0)
        made.add_output(source)
        made.remove_output(source)
        out = np.zeros((8, 2), dtype=np.float32)
        made.render(8, out)
        assert not out.any() and not made.active

    def test_a_node_belongs_to_one_graph(self):
        source = Constant(1.0)
        graph.Graph(RATE).add_output(source)
        with pytest.raises(ValueError):
            graph.Graph(RATE).add_output(source)

    def test_a_node_released_by_its_graph_may_join_another(self):
        source = Constant(1.0)
        first = graph.Graph(RATE)
        first.add_output(source)
        first.remove_output(source)
        graph.Graph(RATE).add_output(source)

    def test_a_block_larger_than_the_maximum_is_refused(self):
        made = graph.Graph(RATE, max_block=16)
        made.add_output(Constant(1.0))
        with pytest.raises(ValueError):
            made.render(32)

    def test_set_outputs_replaces_them_all(self):
        made = graph.Graph(RATE)
        made.add_output(Constant(1.0))
        replacement = Constant(0.5)
        made.set_outputs([replacement])
        assert made.outputs == (replacement,)


class TestMixer:
    def test_the_mixer_mixes_its_graph(self):
        mixer = Mixer(sample_rate=RATE, max_block=64)
        mixer.graph.add_output(Constant(0.5))
        block = mixer.mix(16)
        assert np.allclose(block, 0.5)

    def test_the_graph_is_at_the_mixer_rate(self):
        assert Mixer(sample_rate=RATE).graph.sample_rate == RATE

    def test_the_engine_offers_the_graph(self, engine):
        assert engine.graph is engine.mixer.graph

    @needs_tracemalloc
    def test_rendering_allocates_nothing_measurable(self):
        made = graph.Graph(RATE, max_block=512)
        source = Constant(0.5, 0.25)
        chain = graph.Gain(gain=0.5).connect(source)
        made.add_output(graph.Gain(channel_count=1, channel_count_mode='explicit').connect(chain))
        out = np.zeros((256, 2), dtype=np.float32)
        made.render(256, out)
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for value in range(20):
            chain.gain = 0.5 + value * 0.01
            made.render(256, out)
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        grew = sum(entry.size_diff for entry in after.compare_to(before, 'filename'))
        assert grew < 4096, 'rendering allocated %d bytes' % (grew,)
