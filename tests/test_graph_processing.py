"""Filters, delay, shaping, compression, convolution, analysis, channels and panning."""

import cmath
import math

import numpy as np
import pytest

from omi_audio import graph, spatial
from omi_audio.graph import convolver as convolution
from omi_audio.graph import dynamics
from omi_audio.graph.panner import HeadShadow
from omi_audio.spatial import HEAD_RADIUS, SPEED_OF_SOUND

from support import RATE, needs_tracemalloc, tracemalloc


class Signal(graph.Node):
    """A source playing a fixed array, one block after another."""

    def __init__(self, samples, **named):
        super().__init__(**named)
        data = np.asarray(samples, dtype=np.float32)
        self.samples = data.reshape(1, -1) if data.ndim == 1 else data
        self.at = 0

    def output_channels(self, _computed):
        return self.samples.shape[0]

    def process(self, _block, out, frames):
        part = self.samples[:, self.at:self.at + frames]
        out[:self.samples.shape[0], :frames] = 0.0
        out[:self.samples.shape[0], :part.shape[1]] = part
        self.at += frames


def run(node, samples, block=256, max_block=1024):
    """``samples`` through ``node`` in blocks of ``block``; the output joined."""
    data = np.asarray(samples, dtype=np.float32)
    source = Signal(data)
    node.connect(source)
    made = graph.Graph(RATE, max_block=max_block)
    made.add_output(node)
    total = data.shape[-1]
    pieces = []
    for at in range(0, total, block):
        frames = min(block, total - at)
        made.render(frames)
        pieces.append(node.output(frames).copy())
    return np.concatenate(pieces, axis=1)


def direct_form(coefficients, samples):
    """The biquad sample by sample, as the difference equation reads."""
    b0, b1, b2, a1, a2 = coefficients
    x1 = x2 = y1 = y2 = 0.0
    out = []
    for x in samples:
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, x, y1, y
        out.append(y)
    return np.array(out)


def noise(frames, seed=1):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, frames).astype(np.float32)


class TestBiquadCoefficients:
    @pytest.mark.parametrize('kind', graph.FILTER_TYPES)
    def test_every_kind_matches_the_difference_equation(self, kind):
        samples = noise(3000)
        node = graph.BiquadFilter(kind, frequency=700.0, q=2.0, filter_gain=6.0)
        result = run(node, samples, block=173)[0]
        expected = direct_form(node.coefficients(), samples.astype('d'))
        assert np.allclose(result, expected, atol=1e-5)

    def test_a_low_pass_passes_the_low_and_stops_the_high(self):
        node = graph.BiquadFilter('lowpass', frequency=500.0, q=0.0)
        graph.Graph(RATE).add_output(node)
        assert abs(node.response(50.0)) == pytest.approx(1.0, abs=0.01)
        assert abs(node.response(3000.0)) < 0.05

    def test_a_high_pass_is_the_reverse(self):
        node = graph.BiquadFilter('highpass', frequency=500.0, q=0.0)
        graph.Graph(RATE).add_output(node)
        assert abs(node.response(3500.0)) == pytest.approx(1.0, abs=0.02)
        assert abs(node.response(50.0)) < 0.05

    def test_low_and_high_pass_q_is_in_decibels(self):
        node = graph.BiquadFilter('lowpass', frequency=1000.0, q=6.0)
        graph.Graph(RATE).add_output(node)
        assert 20 * math.log10(abs(node.response(1000.0))) == pytest.approx(6.0, abs=0.1)

    def test_a_band_pass_peaks_at_unity_at_its_centre(self):
        node = graph.BiquadFilter('bandpass', frequency=1000.0, q=5.0)
        graph.Graph(RATE).add_output(node)
        assert abs(node.response(1000.0)) == pytest.approx(1.0, abs=1e-6)

    def test_a_notch_stops_its_centre(self):
        node = graph.BiquadFilter('notch', frequency=1000.0, q=5.0)
        graph.Graph(RATE).add_output(node)
        assert abs(node.response(1000.0)) < 1e-6

    def test_an_all_pass_passes_everything(self):
        node = graph.BiquadFilter('allpass', frequency=1000.0, q=3.0)
        graph.Graph(RATE).add_output(node)
        for frequency in (100.0, 1000.0, 3000.0):
            assert abs(node.response(frequency)) == pytest.approx(1.0, abs=1e-6)

    def test_peaking_boosts_its_centre_by_its_gain(self):
        node = graph.BiquadFilter('peaking', frequency=1000.0, q=1.0, filter_gain=12.0)
        graph.Graph(RATE).add_output(node)
        assert 20 * math.log10(abs(node.response(1000.0))) == pytest.approx(12.0, abs=0.01)

    def test_shelves_boost_their_side(self):
        low = graph.BiquadFilter('lowshelf', frequency=500.0, filter_gain=6.0)
        high = graph.BiquadFilter('highshelf', frequency=500.0, filter_gain=6.0)
        made = graph.Graph(RATE)
        made.add_output(low)
        made.add_output(high)
        assert 20 * math.log10(abs(low.response(10.0))) == pytest.approx(6.0, abs=0.05)
        assert 20 * math.log10(abs(high.response(3900.0))) == pytest.approx(6.0, abs=0.1)

    def test_detune_moves_the_frequency(self):
        moved = graph.biquad_coefficients('lowpass', 500.0, RATE, detune=1200.0)
        assert moved == graph.biquad_coefficients('lowpass', 1000.0, RATE)

    @pytest.mark.parametrize('kind, edge, expected', [
        ('lowpass', 0.0, 0.0), ('lowpass', RATE, 1.0),
        ('highpass', 0.0, 1.0), ('highpass', RATE, 0.0),
        ('bandpass', 0.0, 0.0), ('notch', RATE, 1.0),
        ('allpass', 0.0, 1.0), ('peaking', RATE, 1.0),
        ('lowshelf', 0.0, 1.0), ('lowshelf', RATE, 4.0),
        ('highshelf', 0.0, 4.0), ('highshelf', RATE, 1.0),
    ])
    def test_frequencies_at_the_ends_reach_the_limits(self, kind, edge, expected):
        b0, b1, b2, a1, a2 = graph.biquad_coefficients(kind, edge, RATE, gain=12.04119983)
        assert (b1, b2, a1, a2) == (0.0, 0.0, 0.0, 0.0)
        assert b0 == pytest.approx(expected)

    @pytest.mark.parametrize('kind, expected', [
        ('bandpass', 1.0), ('notch', 0.0), ('allpass', -1.0), ('peaking', 4.0)])
    def test_no_q_reaches_the_limit(self, kind, expected):
        b0, *_ = graph.biquad_coefficients(kind, 1000.0, RATE, q=0.0, gain=12.04119983)
        assert b0 == pytest.approx(expected)

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError):
            graph.biquad_coefficients('bandstop', 100.0, RATE)
        with pytest.raises(ValueError):
            graph.BiquadFilter('bandstop')
        node = graph.BiquadFilter()
        with pytest.raises(ValueError):
            node.kind = 'bandstop'


class TestBiquadFilter:
    def test_a_change_of_settings_carries_on_from_the_same_signal(self):
        samples = noise(2000)
        node = graph.BiquadFilter('lowpass', frequency=400.0)
        source = Signal(samples)
        node.connect(source)
        made = graph.Graph(RATE, max_block=1000)
        made.add_output(node)
        made.render(1000)
        first = node.output(1000)[0].copy()
        node.frequency = 2000.0
        node.q = 3.0
        node.detune = 0.0
        node.filter_gain = 0.0
        node.kind = 'lowpass'
        made.render(1000)
        second = node.output(1000)[0].copy()
        b0, b1, b2, a1, a2 = node.coefficients()
        # The second block's first sample, from the first block's history.
        expected = (b0 * samples[1000] + b1 * samples[999] + b2 * samples[998]
                    - a1 * first[-1] - a2 * first[-2])
        assert second[0] == pytest.approx(expected, abs=1e-5)

    def test_one_frame_blocks_keep_their_history(self):
        samples = noise(40)
        node = graph.BiquadFilter('bandpass', frequency=900.0, q=4.0)
        result = run(node, samples, block=1)[0]
        assert np.allclose(result, direct_form(node.coefficients(), samples.astype('d')),
                           atol=1e-5)

    def test_every_channel_is_filtered_alone(self):
        samples = np.stack([noise(500, 1), noise(500, 2)])
        node = graph.BiquadFilter('highpass', frequency=300.0)
        result = run(node, samples)
        for channel in range(2):
            expected = direct_form(node.coefficients(), samples[channel].astype('d'))
            assert np.allclose(result[channel], expected, atol=1e-5)

    def test_a_low_corner_stays_accurate_over_a_long_block(self):
        samples = noise(4000)
        node = graph.BiquadFilter('lowpass', frequency=10.0, q=-3.0)
        result = run(node, samples, block=4000, max_block=4096)[0]
        assert np.allclose(result, direct_form(node.coefficients(), samples.astype('d')),
                           atol=1e-5)

    def test_settings_read_back(self):
        node = graph.BiquadFilter('peaking', frequency=200.0, q=2.0, detune=5.0,
                                  filter_gain=3.0)
        assert (node.kind, node.frequency, node.q, node.detune, node.filter_gain) == (
            'peaking', 200.0, 2.0, 5.0, 3.0)

    def test_a_response_at_the_rate_of_its_graph(self):
        node = graph.BiquadFilter('lowpass', frequency=100.0)
        assert node.response(0.0) == pytest.approx(complex(1.0))
        assert isinstance(node.response(10.0), complex)
        assert cmath.isfinite(node.response(10.0))


class TestDelay:
    def test_the_input_arrives_later(self):
        node = graph.Delay(delay_time=3.0 / RATE)
        result = run(node, [1.0, 2.0, 3.0, 4.0, 0, 0, 0, 0], block=8)[0]
        assert np.allclose(result, [0, 0, 0, 1, 2, 3, 4, 0])

    def test_a_settled_delay_does_not_glide_there(self):
        node = graph.Delay()
        node.delay_time = 3.0 / RATE
        node.settle()
        result = run(node, [1.0, 2.0, 3.0, 4.0, 0, 0, 0, 0], block=8)[0]
        assert np.allclose(result, [0, 0, 0, 1, 2, 3, 4, 0])

    def test_no_delay_is_the_input(self):
        result = run(graph.Delay(), [1.0, 2.0, 3.0])[0]
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_the_delay_carries_across_blocks(self):
        samples = np.arange(1, 41, dtype='f')
        result = run(graph.Delay(delay_time=7.0 / RATE), samples, block=5)[0]
        assert np.allclose(result[7:], samples[:-7]) and not result[:7].any()

    def test_a_fractional_delay_interpolates(self):
        result = run(graph.Delay(delay_time=0.5 / RATE), [0.0, 2.0, 2.0, 2.0])[0]
        assert np.allclose(result, [0.0, 1.0, 2.0, 2.0])

    def test_the_delay_is_limited_to_the_line(self):
        node = graph.Delay(delay_time=5.0, max_delay_time=0.5)
        assert node.delay_time == 0.5
        node.delay_time = -1.0
        assert node.delay_time == 0.0

    def test_a_changed_delay_glides(self):
        node = graph.Delay(delay_time=0.0, max_delay_time=1.0)
        source = Signal(np.arange(1000, dtype='f'))
        node.connect(source)
        made = graph.Graph(RATE, max_block=100)
        made.add_output(node)
        made.render(100)
        node.delay_time = 50.0 / RATE
        made.render(100)
        steps = np.diff(node.output(100)[0])
        assert np.allclose(steps, 0.5, atol=1e-3)

    def test_a_new_maximum_empties_the_line(self):
        node = graph.Delay(delay_time=2.0 / RATE)
        run(node, [1.0, 1.0, 1.0])
        node.max_delay_time = 2.0
        assert node.max_delay_time == 2.0
        node.graph.render(2)
        assert not node.output(2).any()


class TestWaveShaper:
    def test_the_curve_maps_minus_one_to_one(self):
        node = graph.WaveShaper(curve=[-0.5, 0.0, 0.5])
        result = run(node, [-1.0, -0.5, 0.0, 0.5, 1.0])[0]
        assert np.allclose(result, [-0.5, -0.25, 0.0, 0.25, 0.5])

    def test_beyond_the_range_is_the_end_of_the_curve(self):
        result = run(graph.WaveShaper(curve=[0.1, 0.9]), [-3.0, 3.0])[0]
        assert np.allclose(result, [0.1, 0.9])

    def test_no_curve_passes_the_input(self):
        assert np.allclose(run(graph.WaveShaper(), [0.3, -0.7])[0], [0.3, -0.7])

    def test_a_new_curve(self):
        node = graph.WaveShaper()
        node.curve = [1.0, 1.0]
        assert np.allclose(node.curve, 1.0)
        assert np.allclose(run(node, [0.0, 0.5])[0], 1.0)

    @pytest.mark.parametrize('oversample', ['2x', '4x'])
    def test_oversampling_a_straight_curve_is_a_delay(self, oversample):
        samples = np.sin(2 * np.pi * 200.0 * np.arange(800) / RATE).astype('f') * 0.5
        result = run(graph.WaveShaper(curve=[-1.0, 1.0], oversample=oversample),
                     samples, block=100)[0]
        delay = 16
        assert np.allclose(result[delay + 32:], samples[32:-delay], atol=0.01)

    def test_oversampling_keeps_folded_harmonics_out(self):
        frequency = 1700.0
        samples = (0.9 * np.sin(2 * np.pi * frequency * np.arange(4096) / RATE)).astype('f')
        curve = np.tanh(4.0 * np.linspace(-1, 1, 1025))

        def folded(oversample):
            result = run(graph.WaveShaper(curve=curve, oversample=oversample), samples,
                         block=512, max_block=4096)[0][512:]
            spectrum = np.abs(np.fft.rfft(result * np.hanning(len(result))))
            bins = np.fft.rfftfreq(len(result), 1.0 / RATE)
            # The third harmonic, 5.1 kHz, folds to 2.9 kHz; measure it
            # there, against the fundamental.
            alias = spectrum[np.abs(bins - 2900.0) < 30].max()
            return alias / spectrum[np.abs(bins - frequency) < 30].max()

        assert folded('4x') < folded('none') / 10

    def test_an_unknown_oversampling_is_refused(self):
        with pytest.raises(ValueError):
            graph.WaveShaper(oversample='8x')
        node = graph.WaveShaper()
        with pytest.raises(ValueError):
            node.oversample = '3x'
        node.oversample = '2x'
        assert node.oversample == '2x'


class TestDynamicsCompressor:
    def test_the_static_curve(self):
        assert dynamics.static_curve(-50.0, -24.0, 0.0, 4.0) == -50.0
        assert dynamics.static_curve(-12.0, -24.0, 0.0, 4.0) == pytest.approx(-21.0)
        # At the knee's edges the curve meets both lines.
        assert dynamics.static_curve(-29.0, -24.0, 10.0, 4.0) == pytest.approx(-29.0)
        assert dynamics.static_curve(-19.0, -24.0, 10.0, 4.0) == pytest.approx(-24 + 5 / 4)

    def test_a_loud_signal_is_reduced_and_reported(self):
        node = graph.DynamicsCompressor(threshold=-24.0, knee=0.0, ratio=4.0,
                                        attack=0.0, release=0.25)
        result = run(node, np.full(800, 0.5), block=200)[0]
        level = 20 * math.log10(0.5)
        expected = dynamics.static_curve(level, -24.0, 0.0, 4.0) - level
        assert node.reduction == pytest.approx(expected, abs=1e-6)
        makeup = -0.6 * dynamics.static_curve(0.0, -24.0, 0.0, 4.0)
        assert result[-1] == pytest.approx(0.5 * 10 ** ((expected + makeup) / 20), rel=1e-4)

    def test_a_quiet_signal_has_only_makeup(self):
        node = graph.DynamicsCompressor(threshold=-24.0, knee=0.0, ratio=4.0)
        result = run(node, np.full(400, 0.01))[0]
        assert node.reduction == pytest.approx(0.0, abs=1e-6)
        makeup = 10 ** (-0.6 * dynamics.static_curve(0.0, -24.0, 0.0, 4.0) / 20)
        assert result[-1] == pytest.approx(0.01 * makeup, rel=1e-4)

    def test_attack_takes_time(self):
        quick = graph.DynamicsCompressor(attack=0.0)
        slow = graph.DynamicsCompressor(attack=0.5)
        run(quick, np.full(64, 0.9), block=64)
        run(slow, np.full(64, 0.9), block=64)
        assert slow.reduction > quick.reduction

    def test_release_recovers(self):
        node = graph.DynamicsCompressor(attack=0.0, release=0.01)
        samples = np.concatenate([np.full(400, 0.9), np.full(4000, 1e-4)])
        run(node, samples, block=400)
        assert node.reduction == pytest.approx(0.0, abs=0.01)

    def test_the_gain_moves_smoothly(self):
        node = graph.DynamicsCompressor(attack=0.01, release=0.1)
        samples = np.concatenate([np.full(512, 0.05), np.full(512, 0.9)])
        result = run(node, samples, block=512)[0]
        gains = result[512:] / 0.9
        assert np.abs(np.diff(gains)).max() < 0.05 * gains.max()

    def test_out_of_range_settings_are_limited(self):
        node = graph.DynamicsCompressor(threshold=10.0, knee=-5.0, ratio=0.0,
                                        attack=-1.0, release=float('nan'))
        assert node.settings() == (0.0, 0.0, 1.0, 0.0, 0.0)

    def test_channels_share_one_gain(self):
        node = graph.DynamicsCompressor(attack=0.0)
        samples = np.stack([np.full(256, 0.9), np.full(256, 0.1)])
        result = run(node, samples)
        assert result[1, -1] / 0.1 == pytest.approx(result[0, -1] / 0.9, rel=1e-4)


class TestConvolver:
    def test_an_impulse_gives_the_response_one_partition_later(self):
        response = [1.0, 0.5, 0.25]
        samples = np.zeros(1024)
        samples[0] = 1.0
        result = run(graph.Convolver(response), samples, block=100)[0]
        delay = convolution.PARTITION
        assert np.allclose(result[delay:delay + 3], response, atol=1e-6)
        assert np.abs(result[:delay]).max() < 1e-6

    def test_a_long_response_matches_direct_convolution(self):
        rng = np.random.default_rng(3)
        response = rng.uniform(-1, 1, 1500) * np.exp(-np.arange(1500) / 300)
        samples = noise(4000)
        result = run(graph.Convolver(response), samples, block=333, max_block=1024)[0]
        expected = np.convolve(samples.astype('d'), response)[:4000]
        delay = convolution.PARTITION
        assert np.allclose(result[delay:], expected[:-delay], atol=1e-4)

    def test_a_stereo_response_makes_stereo_from_mono(self):
        samples = np.zeros(600)
        samples[0] = 1.0
        result = run(graph.Convolver([[1.0], [0.5]]), samples)
        delay = convolution.PARTITION
        assert result.shape[0] == 2
        assert result[0, delay] == pytest.approx(1.0) and result[1, delay] == pytest.approx(0.5)

    def test_a_mono_response_on_stereo_input(self):
        samples = np.zeros((2, 600))
        samples[0, 0], samples[1, 0] = 1.0, 0.25
        result = run(graph.Convolver([2.0]), samples)
        delay = convolution.PARTITION
        assert result[0, delay] == pytest.approx(2.0) and result[1, delay] == pytest.approx(0.5)

    def test_true_stereo_crosses_the_channels(self):
        samples = np.zeros((2, 600))
        samples[1, 0] = 1.0
        response = [[0.0], [0.0], [0.3], [0.7]]
        result = run(graph.Convolver(response), samples)
        delay = convolution.PARTITION
        assert result[0, delay] == pytest.approx(0.3) and result[1, delay] == pytest.approx(0.7)

    def test_no_response_is_silence(self):
        node = graph.Convolver()
        result = run(node, np.ones(300))
        assert result.shape[0] == 1 and not result.any()

    def test_normalisation_follows_the_specification(self):
        response = np.full(100, 0.5)
        expected = 1.0 / 0.5 * 0.00125 * 44100.0 / RATE
        assert convolution.normalization_scale(response, RATE) == pytest.approx(expected)
        quad = np.full((4, 10), 0.5)
        assert convolution.normalization_scale(quad, 0) == pytest.approx(1.0 / 0.5 * 0.00125 / 2)
        silent = convolution.normalization_scale(np.zeros(10), 44100)
        assert silent == pytest.approx(0.00125 / 0.000125)

    def test_a_normalised_response_is_scaled(self):
        samples = np.zeros(600)
        samples[0] = 1.0
        node = graph.Convolver([0.5, 0.5])
        node.normalize = True
        assert node.normalize
        result = run(node, samples)
        scale = convolution.normalization_scale([0.5, 0.5], RATE)
        assert result[0, convolution.PARTITION] == pytest.approx(0.5 * scale, rel=1e-5)

    def test_three_channels_keep_two(self):
        node = graph.Convolver(np.ones((3, 4)))
        assert node.response.shape == (2, 4)


class TestAnalyser:
    def test_the_signal_passes_unchanged_whatever_the_gain(self):
        node = graph.Analyser(gain=0.1)
        assert np.allclose(run(node, [0.5, -0.5])[0], [0.5, -0.5])

    def test_time_domain_data_is_the_latest_samples(self):
        node = graph.Analyser(fft_size=32)
        samples = np.arange(100, dtype='f') / 100
        run(node, samples, block=30)
        assert np.allclose(node.time_domain_data(), samples[-32:])

    def test_the_gain_scales_the_analysis(self):
        node = graph.Analyser(fft_size=32, gain=2.0)
        run(node, np.full(40, 0.25))
        assert np.allclose(node.time_domain_data(), 0.5)

    def test_channels_are_mixed_to_mono(self):
        node = graph.Analyser(fft_size=32)
        run(node, np.stack([np.full(40, 1.0), np.zeros(40)]))
        assert np.allclose(node.time_domain_data(), 0.5)

    def test_fft_size_is_a_power_of_two_in_range(self):
        node = graph.Analyser(fft_size=1000)
        assert node.fft_size == 1024 and node.frequency_bin_count == 512
        node.fft_size = 1
        assert node.fft_size == 32
        node.fft_size = 10 ** 6
        assert node.fft_size == 32768

    def test_a_tone_peaks_in_its_bin(self):
        node = graph.Analyser(fft_size=1024, smoothing=0.0)
        frequency = 1000.0
        run(node, np.sin(2 * np.pi * frequency * np.arange(2048) / RATE))
        data = node.frequency_data()
        assert len(data) == 512
        assert np.argmax(data) == round(frequency * 1024 / RATE)

    def test_a_full_scale_sine_reads_the_window_gain(self):
        node = graph.Analyser(fft_size=1024, smoothing=0.0)
        run(node, np.sin(2 * np.pi * 1000.0 * np.arange(2048) / RATE))
        # |X| / N for a unit sine is half the window's mean, 0.42 / 2.
        assert node.frequency_data().max() == pytest.approx(20 * math.log10(0.21), abs=0.1)

    def test_smoothing_blends_with_the_last_call(self):
        node = graph.Analyser(fft_size=64, smoothing=0.5)
        run(node, np.sin(2 * np.pi * 1000.0 * np.arange(128) / RATE))
        first = 10 ** (node.frequency_data() / 20)
        second = 10 ** (node.frequency_data() / 20)
        assert np.allclose(second, first * 1.5, rtol=1e-6)

    def test_silence_reads_the_floor(self):
        node = graph.Analyser(fft_size=32)
        run(node, np.zeros(40))
        assert np.all(node.frequency_data() == pytest.approx(graph.analyser.SILENT_DB))

    def test_byte_data(self):
        node = graph.Analyser(fft_size=32, min_decibels=-100.0, max_decibels=-30.0)
        run(node, np.zeros(40))
        assert not node.byte_frequency_data().any()
        assert np.all(node.byte_time_domain_data() == 128)

    def test_a_new_size_starts_its_smoothing_afresh(self):
        node = graph.Analyser(fft_size=64)
        run(node, np.ones(100))
        node.frequency_data()
        node.fft_size = 32
        assert len(node.frequency_data()) == 16


class TestChannels:
    def test_a_splitter_keeps_its_channels_apart(self):
        splitter = graph.ChannelSplitter()
        source = Signal(np.stack([np.full(8, 1.0), np.full(8, 2.0)]))
        splitter.connect(source)
        right = graph.Gain().connect(splitter, channel=1)
        made = graph.Graph(RATE)
        made.add_output(right)
        made.render(8)
        assert np.allclose(right.output(8), 2.0)
        assert splitter.channel_interpretation == 'discrete'

    def test_a_selector_picks_one(self):
        selector = graph.ChannelSelector(selection=1)
        result = run(selector, np.stack([np.full(8, 1.0), np.full(8, 2.0)]))
        assert result.shape[0] == 1 and np.allclose(result, 2.0)

    def test_a_selector_past_the_channels_is_silent(self):
        assert not run(graph.ChannelSelector(selection=5), np.ones(8)).any()

    def test_a_merger_makes_a_channel_of_each_input(self):
        merger = graph.ChannelMerger()
        merger.connect(Signal(np.full(8, 1.0)))
        merger.connect(Signal(np.stack([np.full(8, 0.5), np.full(8, 1.5)])))
        merger.connect(Signal(np.full(8, 3.0)), channel=4)
        made = graph.Graph(RATE)
        made.add_output(merger)
        made.render(8)
        block = merger.output(8)
        assert block.shape[0] == 3
        assert np.allclose(block[0], 1.0) and np.allclose(block[1], 1.0)
        assert not block[2].any()

    def test_split_and_merged_swapped(self):
        source = Signal(np.stack([np.full(8, 1.0), np.full(8, 2.0)]))
        splitter = graph.ChannelSplitter().connect(source)
        merger = graph.ChannelMerger()
        merger.connect(graph.ChannelSelector(1).connect(splitter))
        merger.connect(graph.ChannelSelector(0).connect(splitter))
        made = graph.Graph(RATE)
        made.add_output(merger)
        made.render(8)
        assert np.allclose(merger.output(8)[0], 2.0) and np.allclose(merger.output(8)[1], 1.0)


class TestPanner:
    def test_the_gains_place_the_sound(self):
        panner = graph.Panner(left=0.25, right=0.75)
        result = run(panner, np.ones(16))
        assert result.shape[0] == 2
        assert np.allclose(result[0], 0.25) and np.allclose(result[1], 0.75)

    def test_stereo_input_is_mixed_to_mono(self):
        panner = graph.Panner(left=1.0, right=1.0)
        result = run(panner, np.stack([np.ones(8), np.zeros(8)]))
        assert np.allclose(result, 0.5)

    def test_new_gains_are_ramped(self):
        panner = graph.Panner(left=1.0, right=1.0)
        panner.connect(Signal(np.ones(64)))
        made = graph.Graph(RATE)
        made.add_output(panner)
        made.render(16)
        panner.set_gains(0.0, float('nan'))
        assert panner.gains == (0.0, 0.0)
        made.render(16)
        left = panner.output(16)[0]
        assert left[0] > 0.9 and left[-1] == pytest.approx(0.0)

    def test_settled_gains_are_not_ramped(self):
        panner = graph.Panner()
        panner.set_gains(0.5, 0.25)
        panner.propagation_delay = 4.0 / RATE
        panner.settle()
        result = run(panner, np.arange(1, 17, dtype='f'), block=16)
        assert np.allclose(result[0, 4:], 0.5 * np.arange(1, 13))
        assert np.allclose(result[1, 4:], 0.25 * np.arange(1, 13))

    def test_a_propagation_delay_delays(self):
        panner = graph.Panner(left=1.0, right=1.0)
        panner.propagation_delay = 4.0 / RATE
        assert panner.propagation_delay == pytest.approx(4.0 / RATE)
        result = run(panner, np.arange(1, 17, dtype='f'), block=16)[0]
        # The delay glides in from nothing over the first block.
        assert result[-1] == pytest.approx(12.0, abs=0.01)

    def test_a_shrinking_delay_raises_the_pitch(self):
        panner = graph.Panner(left=1.0, right=1.0, max_propagation_delay=2.0)
        assert panner.max_propagation_delay == 2.0
        panner.max_propagation_delay = 1.0
        panner.propagation_delay = 0.5
        tone = np.sin(2 * np.pi * 400.0 * np.arange(RATE * 2) / RATE).astype('f')
        panner.connect(Signal(tone))
        made = graph.Graph(RATE, max_block=1024)
        made.add_output(panner)
        for _ in range(8):
            made.render(1000)
        # Closing at a quarter of a second each second plays 1.25 times as fast.
        pieces = []
        for _ in range(4):
            panner.propagation_delay -= 0.25 * 100 / RATE
            made.render(100)
            pieces.append(panner.output(100)[0].copy())
        samples = np.concatenate(pieces)
        crossings = np.sum(np.diff(np.sign(samples)) != 0)
        assert crossings / 2 / (400 / RATE) == pytest.approx(400 * 1.25, rel=0.1)


@needs_tracemalloc
def test_processing_allocates_nothing_measurable():
    made = graph.Graph(RATE, max_block=512)
    tone = graph.Oscillator(frequency=300.0, wave=graph.PeriodicWave.of_type('sawtooth'))
    chain = graph.BiquadFilter('lowpass', frequency=800.0).connect(tone)
    chain = graph.Delay(delay_time=0.01).connect(chain)
    chain = graph.DynamicsCompressor().connect(chain)
    chain = graph.WaveShaper(curve=[-1, 0, 1], oversample='2x').connect(chain)
    chain = graph.Convolver(np.ones(700) / 700).connect(chain)
    chain = graph.Analyser().connect(chain)
    buffer = graph.BufferSource(np.ones((2, 2000)), buffer_rate=RATE, loop=True)
    merger = graph.ChannelMerger().connect(chain).connect(buffer)
    panner = graph.Panner(left=0.5, right=0.5).connect(merger)
    made.add_output(panner)
    out = np.zeros((256, 2), dtype=np.float32)
    for _ in range(3):
        made.render(256, out)
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(20):
        made.render(256, out)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    grew = sum(entry.size_diff for entry in after.compare_to(before, 'filename'))
    assert grew < 4096, 'rendering allocated %d bytes' % (grew,)


class TestHeadPanning:
    """A Panner given a head response: the far ear later and its highs shadowed"""

    def test_the_far_ear_hears_later(self):
        panner = graph.Panner(left=1.0, right=1.0)
        panner.set_head(spatial.head_response(math.pi / 2))
        panner.settle()
        click = np.zeros(512)
        click[100] = 1.0
        result = run(panner, click, block=128)
        late = spatial.head_response(math.pi / 2).left_delay * RATE
        assert int(np.argmax(np.abs(result[1]))) == 100
        assert int(np.argmax(np.abs(result[0]))) == pytest.approx(100 + late, abs=1.5)

    def test_the_far_ears_highs_are_shadowed(self):
        panner = graph.Panner(left=1.0, right=1.0)
        panner.set_head(spatial.head_response(math.pi / 2))
        panner.settle()
        high = np.sin(2 * np.pi * (RATE * 0.45) * np.arange(4096) / RATE)
        result = run(panner, high)[:, 1024:]
        assert np.abs(result[1]).max() > 1.5
        assert np.abs(result[0]).max() < 0.6

    def test_low_frequencies_pass_both_ears(self):
        panner = graph.Panner(left=1.0, right=1.0)
        panner.set_head(spatial.head_response(math.pi / 2))
        panner.settle()
        result = run(panner, np.ones(2048))[:, 1024:]
        assert np.allclose(result, 1.0, atol=0.02)

    def test_no_head_is_the_gains_alone(self):
        panner = graph.Panner(left=0.5, right=0.25)
        panner.set_head(None)
        result = run(panner, np.ones(16))
        assert np.allclose(result[0], 0.5) and np.allclose(result[1], 0.25)

    @pytest.mark.parametrize('rate', [8000, 48000])
    def test_the_shadow_is_its_difference_equation(self, rate):
        signal = np.random.default_rng(3).standard_normal(700).astype('f')
        shelf, found = HeadShadow(), np.zeros(700, 'f')
        for at in range(0, 700, 300):
            shelf.run(signal[at:at + 300], 0.3, HEAD_RADIUS, rate, found[at:at + 300])
        twice, turn = 2.0 * rate, 2.0 * SPEED_OF_SOUND / HEAD_RADIUS
        b0 = (0.3 * twice + turn) / (twice + turn)
        b1 = (turn - 0.3 * twice) / (twice + turn)
        a1 = (turn - twice) / (twice + turn)
        assert np.allclose(found, direct_form((b0, b1, 0.0, a1, 0.0), signal), atol=1e-5)
