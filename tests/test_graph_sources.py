"""Oscillators, periodic waves, buffers and captured input, sample by sample."""

import math

import numpy as np
import pytest

from omi_audio import graph
from omi_audio.graph import sources

from support import RATE


def play(node, frames=256, blocks=1, max_block=1024):
    """``node``'s output over ``blocks`` blocks of ``frames``, joined."""
    made = graph.Graph(RATE, max_block=max_block)
    made.add_output(node)
    rendered = []
    for _ in range(blocks):
        made.render(frames)
        rendered.append(node.output(frames).copy())
    return np.concatenate(rendered, axis=1)


def strongest_frequency(samples, rate=RATE):
    """The frequency of the largest bin of ``samples``' spectrum."""
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return np.argmax(spectrum) * rate / len(samples)


class TestPeriodicWave:
    def test_a_sine_table_is_a_sine(self):
        table = graph.PeriodicWave.of_type('sine').table(100)
        expected = np.sin(2 * np.pi * np.arange(sources.TABLE_SIZE) / sources.TABLE_SIZE)
        assert np.allclose(table[:-1], expected, atol=1e-5)

    def test_the_table_wraps_one_sample(self):
        table = graph.PeriodicWave.of_type('sawtooth').table(50)
        assert table[-1] == table[0]

    def test_a_normalised_wave_peaks_at_one(self):
        wave = graph.PeriodicWave(imag=[0, 3.0, 1.0])
        assert np.abs(wave.table(10)).max() == pytest.approx(1.0, abs=1e-4)

    def test_an_unnormalised_wave_keeps_its_level(self):
        wave = graph.PeriodicWave(imag=[0, 3.0], normalize=False)
        assert np.abs(wave.table(10)).max() == pytest.approx(3.0, abs=1e-3)

    def test_real_terms_are_cosines(self):
        table = graph.PeriodicWave(real=[0, 1.0]).table(10)
        assert table[0] == pytest.approx(1.0, abs=1e-5)

    def test_the_first_terms_are_ignored(self):
        assert np.allclose(graph.PeriodicWave(real=[5.0, 0.0], imag=[5.0, 0.0]).table(4), 0.0)

    def test_a_square_has_only_odd_harmonics(self):
        table = graph.PeriodicWave.of_type('square').table(sources.TABLE_SIZE)
        spectrum = np.abs(np.fft.rfft(table[:-1]))
        assert spectrum[3] > 0.1 * spectrum[1] and spectrum[2] < 1e-6 * spectrum[1]

    def test_a_triangle_falls_as_the_square_of_the_harmonic(self):
        table = graph.PeriodicWave.of_type('triangle').table(sources.TABLE_SIZE)
        spectrum = np.abs(np.fft.rfft(table[:-1]))
        assert spectrum[3] / spectrum[1] == pytest.approx(1.0 / 9.0, rel=1e-3)

    def test_a_sawtooth_falls_as_the_harmonic(self):
        table = graph.PeriodicWave.of_type('sawtooth').table(sources.TABLE_SIZE)
        spectrum = np.abs(np.fft.rfft(table[:-1]))
        assert spectrum[2] / spectrum[1] == pytest.approx(0.5, rel=1e-3)

    def test_a_table_holds_no_more_harmonics_than_asked(self):
        table = graph.PeriodicWave.of_type('sawtooth').table(8)
        spectrum = np.abs(np.fft.rfft(table[:-1]))
        assert spectrum[9:].max() < 1e-6 * spectrum[1]

    def test_tables_are_kept(self):
        wave = graph.PeriodicWave.of_type('square')
        assert wave.table(100) is wave.table(100)

    def test_custom_is_made_from_coefficients(self):
        with pytest.raises(ValueError):
            graph.PeriodicWave.of_type('custom')

    def test_an_unknown_type_is_refused(self):
        with pytest.raises(ValueError):
            graph.PeriodicWave.of_type('wobble')


class TestOscillator:
    def test_a_sine_is_at_its_frequency(self):
        samples = play(graph.Oscillator(frequency=500.0), frames=1024)[0]
        assert strongest_frequency(samples) == pytest.approx(500.0, abs=RATE / 1024)

    def test_a_sine_starts_at_zero_and_matches_the_formula(self):
        samples = play(graph.Oscillator(frequency=250.0), frames=64)[0]
        expected = np.sin(2 * np.pi * 250.0 * np.arange(64) / RATE)
        assert np.allclose(samples, expected, atol=1e-3)

    def test_phase_carries_across_blocks(self):
        samples = play(graph.Oscillator(frequency=250.0), frames=50, blocks=4)[0]
        expected = np.sin(2 * np.pi * 250.0 * np.arange(200) / RATE)
        assert np.allclose(samples, expected, atol=1e-3)

    def test_detune_moves_the_pitch_by_cents(self):
        oscillator = graph.Oscillator(frequency=400.0, detune=1200.0)
        samples = play(oscillator, frames=1024)[0]
        assert strongest_frequency(samples) == pytest.approx(800.0, abs=RATE / 1024)
        assert oscillator.pitch == pytest.approx(800.0)

    def test_the_pitch_is_limited_to_nyquist(self):
        oscillator = graph.Oscillator(frequency=RATE)
        graph.Graph(RATE).add_output(oscillator)
        assert oscillator.pitch == RATE / 2

    def test_a_change_of_frequency_glides(self):
        oscillator = graph.Oscillator(frequency=100.0)
        made = graph.Graph(RATE, max_block=512)
        made.add_output(oscillator)
        made.render(256)
        oscillator.frequency = 1000.0
        made.render(256)
        made.render(512)
        assert strongest_frequency(oscillator.output(512)[0]) == pytest.approx(
            1000.0, abs=2 * RATE / 512)

    def test_a_square_is_band_limited_at_its_pitch(self):
        oscillator = graph.Oscillator(frequency=1000.0,
                                      wave=graph.PeriodicWave.of_type('square'))
        samples = play(oscillator, frames=1024)[0]
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(1024)))
        bins = np.fft.rfftfreq(1024, 1.0 / RATE)
        # 1 and 3 kHz only: 5 kHz is past Nyquist, and nothing folds back.
        stray = spectrum[(np.abs(bins - 1000) > 60) & (np.abs(bins - 3000) > 60)]
        assert stray.max() < 0.01 * spectrum.max()

    def test_a_new_wave_is_played(self):
        oscillator = graph.Oscillator(frequency=250.0)
        graph.Graph(RATE).add_output(oscillator)
        oscillator.wave = graph.PeriodicWave(real=[0, 1.0])
        assert oscillator.wave.real[1] == 1.0
        assert play_more(oscillator)[0] == pytest.approx(1.0, abs=1e-3)

    def test_no_wave_is_a_sine(self):
        oscillator = graph.Oscillator(wave=graph.PeriodicWave(real=[0, 1.0]))
        oscillator.wave = None
        assert oscillator.wave.imag[1] == 1.0

    def test_start_returns_to_the_start_of_a_cycle(self):
        oscillator = graph.Oscillator(frequency=250.0)
        made = graph.Graph(RATE)
        made.add_output(oscillator)
        made.render(13)
        oscillator.start()
        made.render(8)
        assert oscillator.output(8)[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_detune_set_later(self):
        oscillator = graph.Oscillator(frequency=400.0)
        oscillator.detune = -1200.0
        assert oscillator.detune == -1200.0
        graph.Graph(RATE).add_output(oscillator)
        assert oscillator.pitch == pytest.approx(200.0)

    def test_a_silent_oscillator_has_every_harmonic_table(self):
        oscillator = graph.Oscillator(frequency=0.0)
        assert not play(oscillator, frames=16).any()


def play_more(node, frames=16):
    """One more block of a node already in a graph."""
    node.graph.render(frames)
    return node.output(frames)[0].copy()


class TestBufferSource:
    def test_it_plays_its_samples(self):
        source = graph.BufferSource(np.arange(8, dtype='f') / 10, buffer_rate=RATE)
        assert np.allclose(play(source, frames=8)[0], np.arange(8) / 10)

    def test_it_stops_at_the_end_and_says_so(self):
        source = graph.BufferSource(np.ones(5), buffer_rate=RATE)
        samples = play(source, frames=8)[0]
        assert np.allclose(samples[:5], 1.0) and not samples[5:].any()
        assert source.finished

    def test_channels_are_kept(self):
        source = graph.BufferSource([[1.0] * 4, [0.5] * 4], buffer_rate=RATE)
        block = play(source, frames=4)
        assert block.shape[0] == 2 and np.allclose(block[1], 0.5)
        assert source.length == 4

    def test_a_loop_repeats(self):
        source = graph.BufferSource([0.0, 1.0, 2.0, 3.0], buffer_rate=RATE, loop=True)
        assert np.allclose(play(source, frames=10)[0], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        assert not source.finished

    def test_a_loop_region_repeats_after_the_lead_in(self):
        data = np.arange(8, dtype='f')
        source = graph.BufferSource(data, buffer_rate=RATE, loop=True,
                                    loop_start=2.0 / RATE, loop_end=5.0 / RATE)
        assert np.allclose(play(source, frames=10)[0], [0, 1, 2, 3, 4, 2, 3, 4, 2, 3])

    def test_a_loop_start_past_the_end_loops_everything(self):
        source = graph.BufferSource([0.0, 1.0, 2.0], buffer_rate=RATE, loop=True,
                                    loop_start=2.0 / RATE, loop_end=1.0 / RATE)
        assert np.allclose(play(source, frames=6)[0], [0, 1, 2, 0, 1, 2])

    def test_double_rate_skips_every_other_sample(self):
        source = graph.BufferSource(np.arange(16, dtype='f'), buffer_rate=RATE,
                                    playback_rate=2.0)
        assert np.allclose(play(source, frames=4)[0], [0, 2, 4, 6])

    def test_half_rate_interpolates(self):
        source = graph.BufferSource(np.arange(16, dtype='f'), buffer_rate=RATE,
                                    playback_rate=0.5)
        assert np.allclose(play(source, frames=4)[0], [0, 0.5, 1, 1.5])

    def test_detune_is_a_rate(self):
        source = graph.BufferSource(np.arange(16, dtype='f'), buffer_rate=RATE,
                                    detune=1200.0)
        assert np.allclose(play(source, frames=4)[0], [0, 2, 4, 6])

    def test_a_buffer_at_another_rate_plays_at_its_own_speed(self):
        source = graph.BufferSource(np.arange(16, dtype='f'), buffer_rate=RATE * 2)
        assert np.allclose(play(source, frames=4)[0], [0, 2, 4, 6])

    def test_a_negative_rate_plays_backwards_from_the_end(self):
        source = graph.BufferSource([0.0, 1.0, 2.0, 3.0], buffer_rate=RATE,
                                    playback_rate=-1.0)
        samples = play(source, frames=6)[0]
        assert np.allclose(samples, [3, 2, 1, 0, 0, 0])
        assert source.finished

    def test_a_negative_rate_loops_backwards(self):
        source = graph.BufferSource([0.0, 1.0, 2.0, 3.0], buffer_rate=RATE,
                                    playback_rate=-1.0, loop=True)
        assert np.allclose(play(source, frames=6)[0], [3, 2, 1, 0, 3, 2])

    def test_duration_plays_only_that_much(self):
        source = graph.BufferSource(np.ones(10), buffer_rate=RATE, duration=3.0 / RATE)
        samples = play(source, frames=6)[0]
        assert np.allclose(samples[:3], 1.0) and not samples[3:].any()

    def test_an_empty_buffer_is_silent_and_finished(self):
        source = graph.BufferSource()
        assert not play(source, frames=4).any()
        assert source.finished

    def test_start_plays_again(self):
        source = graph.BufferSource([1.0, 2.0], buffer_rate=RATE)
        play(source, frames=4)
        source.start()
        assert np.allclose(play_more(source, frames=2), [1.0, 2.0])

    def test_start_at_an_offset_plays_from_there(self):
        source = graph.BufferSource(np.arange(8, dtype='f'), buffer_rate=RATE)
        source.start(offset=3.0 / RATE)
        assert np.allclose(play(source, frames=4)[0], [3, 4, 5, 6])

    def test_an_offset_counts_from_the_start_when_playing_backwards(self):
        source = graph.BufferSource(np.arange(8, dtype='f'), buffer_rate=RATE,
                                    playback_rate=-1.0)
        source.start(offset=3.0 / RATE)
        assert np.allclose(play(source, frames=5)[0], [3, 2, 1, 0, 0])

    def test_an_offset_past_the_end_of_a_loop_folds_into_it(self):
        source = graph.BufferSource([0.0, 1.0, 2.0, 3.0], buffer_rate=RATE, loop=True)
        source.start(offset=6.0 / RATE)
        assert np.allclose(play(source, frames=3)[0], [2, 3, 0])

    def test_an_offset_past_the_end_of_a_one_shot_is_silent(self):
        source = graph.BufferSource([1.0, 1.0], buffer_rate=RATE)
        source.start(offset=5.0 / RATE)
        assert not play(source, frames=3).any()
        assert source.finished

    def test_a_non_finite_offset_starts_at_the_start(self):
        source = graph.BufferSource([1.0, 2.0], buffer_rate=RATE)
        source.start(offset=math.nan)
        assert np.allclose(play(source, frames=2)[0], [1.0, 2.0])

    def test_elapsed_is_the_playhead_in_seconds(self):
        source = graph.BufferSource(np.ones(RATE), buffer_rate=RATE)
        play(source, frames=RATE // 10, max_block=RATE)
        assert source.elapsed == pytest.approx(0.1)

    def test_a_zero_rate_holds(self):
        source = graph.BufferSource([0.5, 1.0], buffer_rate=RATE, playback_rate=0.0)
        assert np.allclose(play(source, frames=3)[0], 0.5)

    def test_a_new_buffer(self):
        source = graph.BufferSource()
        source.set_buffer([[0.25, 0.25]], RATE)
        assert source.buffer.shape == (1, 2) and source.buffer_rate == RATE
        assert np.allclose(play(source, frames=2), 0.25)

    def test_the_last_sample_interpolates_to_itself(self):
        source = graph.BufferSource([0.0, 1.0], buffer_rate=RATE, playback_rate=0.5)
        assert np.allclose(play(source, frames=4)[0], [0, 0.5, 1.0, 1.0])

    def test_a_loop_wraps_its_last_sample_to_the_first(self):
        source = graph.BufferSource([0.0, 1.0], buffer_rate=RATE, playback_rate=0.5,
                                    loop=True)
        assert np.allclose(play(source, frames=4)[0], [0, 0.5, 1.0, 0.5])


class TestCapture:
    def test_what_was_written_is_played(self):
        ring = graph.CaptureRing(64)
        ring.write([0.1, 0.2, 0.3])
        samples = play(graph.CaptureSource(ring), frames=4)[0]
        assert np.allclose(samples, [0.1, 0.2, 0.3, 0.0])

    def test_reading_consumes(self):
        ring = graph.CaptureRing(64)
        ring.write([0.5] * 4)
        target = np.zeros(4, dtype=np.float32)
        assert ring.read_into(target) == 4
        assert ring.read_into(target) == 0 and not target.any()

    def test_a_full_ring_keeps_what_fits(self):
        ring = graph.CaptureRing(4)
        assert ring.write(np.ones(6)) == 4
        assert ring.waiting == 4 and ring.capacity == 4

    def test_writes_wrap_round_the_ring(self):
        ring = graph.CaptureRing(4)
        target = np.zeros(3, dtype=np.float32)
        ring.write([1.0, 2.0, 3.0])
        ring.read_into(target)
        ring.write([4.0, 5.0, 6.0])
        ring.read_into(target)
        assert np.allclose(target, [4.0, 5.0, 6.0])

    def test_a_source_has_a_ring_of_its_own(self):
        assert isinstance(graph.CaptureSource().ring, graph.CaptureRing)


def test_nearby_harmonic_counts_share_a_table_holding_no_more():
    wave = graph.PeriodicWave.of_type('sawtooth')
    assert wave.table(1000) is wave.table(1001)
    for count in (1, 7, 300, 10 ** 6):
        spectrum = np.abs(np.fft.rfft(wave.table(count)[:-1]))
        beyond = spectrum[min(count, sources.TABLE_SIZE // 2 - 1) + 1:]
        assert beyond.max(initial=0.0) < 1e-6 * spectrum[1]


def test_a_sine_needs_no_harmonics_beyond_its_first():
    oscillator = graph.Oscillator(frequency=3000.0)
    samples = play(oscillator, frames=1024)[0]
    assert np.abs(samples).max() == pytest.approx(1.0, abs=0.01)
    assert math.isclose(strongest_frequency(samples), 3000.0, abs_tol=RATE / 1024)
