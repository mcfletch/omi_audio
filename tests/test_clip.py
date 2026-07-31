"""Decoded clips, the decode seam, and the cache in front of it."""

import math

import numpy as np
import pytest

from omi_audio import clip as clipmodule
from omi_audio import synth
from omi_audio.clip import Clip, ClipCache, DecodeError

from support import needs_miniaudio, wav_bytes, write_wav


class TestClip:
    """A clip is mono float32 samples and the rate they were taken at."""

    def test_duration_is_frames_over_rate(self):
        clip = Clip(np.zeros(4410, dtype='f'), 44100)
        assert clip.duration == pytest.approx(0.1)
        assert clip.frames == 4410

    def test_samples_are_coerced_to_mono_float32(self):
        clip = Clip([0, 1, 0, -1], 8000)
        assert clip.samples.dtype == np.float32
        assert clip.samples.ndim == 1

    def test_an_empty_clip_has_zero_duration(self):
        assert Clip([], 8000).duration == 0.0

    def test_a_clip_needs_a_positive_sample_rate(self):
        with pytest.raises(ValueError):
            Clip([0.0], 0)

    def test_stereo_samples_are_mixed_down_to_mono(self):
        """Spatialising a stereo source is meaningless, so clips are mono."""
        stereo = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype='f')
        clip = Clip(stereo, 8000)
        assert clip.samples.tolist() == pytest.approx([0.5, 0.5, 1.0])

    def test_resampling_changes_the_rate_and_the_length(self):
        clip = Clip(np.zeros(100, dtype='f'), 8000).resampled(16000)
        assert clip.sample_rate == 16000
        assert clip.frames == 200

    def test_resampling_to_the_same_rate_returns_the_same_clip(self):
        clip = Clip(np.zeros(100, dtype='f'), 8000)
        assert clip.resampled(8000) is clip

    def test_resampling_preserves_the_waveform(self):
        """A tone resampled to twice the rate is still the same tone."""
        original = synth.tone(440.0, 0.05, sample_rate=8000)
        doubled = original.resampled(16000)
        assert doubled.duration == pytest.approx(original.duration, rel=1e-3)
        # Peak amplitude survives; a broken resample shows up as attenuation.
        assert float(np.abs(doubled.samples).max()) == pytest.approx(
            float(np.abs(original.samples).max()), rel=0.05)


class TestSynth:
    """Procedural clips: demos and tests need sound, not sound *files*."""

    def test_tone_has_the_requested_duration_and_rate(self):
        clip = synth.tone(440.0, 0.25, sample_rate=22050)
        assert clip.sample_rate == 22050
        assert clip.frames == 22050 // 4

    def test_tone_is_at_the_requested_frequency(self):
        """The strongest frequency bin is the one asked for."""
        clip = synth.tone(1000.0, 0.5, sample_rate=16000, fade=0.0)
        spectrum = np.abs(np.fft.rfft(clip.samples))
        peak = np.fft.rfftfreq(clip.frames, 1.0 / clip.sample_rate)[spectrum.argmax()]
        assert peak == pytest.approx(1000.0, abs=5.0)

    def test_tone_fades_in_and_out_so_it_does_not_click(self):
        clip = synth.tone(440.0, 0.2, sample_rate=8000, fade=0.02)
        assert abs(float(clip.samples[0])) < 1e-3
        assert abs(float(clip.samples[-1])) < 1e-3

    def test_noise_is_broadband_and_bounded(self):
        clip = synth.noise(0.2, sample_rate=8000, seed=7)
        assert float(np.abs(clip.samples).max()) <= 1.0
        assert float(clip.samples.std()) > 0.1

    def test_noise_is_reproducible_from_its_seed(self):
        first = synth.noise(0.1, sample_rate=8000, seed=3)
        second = synth.noise(0.1, sample_rate=8000, seed=3)
        assert np.array_equal(first.samples, second.samples)

    def test_impact_decays_to_silence(self):
        clip = synth.impact(0.3, sample_rate=8000, seed=1)
        head = float(np.abs(clip.samples[:100]).mean())
        tail = float(np.abs(clip.samples[-100:]).mean())
        assert tail < head * 0.1

    def test_a_zero_length_request_is_a_clip_with_no_frames(self):
        """Every generator has to survive a duration of nothing."""
        for clip in (synth.chirp(200.0, 2000.0, 0.0, sample_rate=8000),
                     synth.noise(0.0, sample_rate=8000),
                     synth.impact(0.0, sample_rate=8000),
                     synth.silence(0.0, sample_rate=8000)):
            assert clip.frames == 0

    def test_chirp_sweeps_from_one_frequency_to_another(self):
        clip = synth.chirp(200.0, 2000.0, 0.5, sample_rate=16000)
        half = clip.frames // 2
        low = np.abs(np.fft.rfft(clip.samples[:half]))
        high = np.abs(np.fft.rfft(clip.samples[half:]))
        freqs = np.fft.rfftfreq(half, 1.0 / clip.sample_rate)
        assert freqs[low.argmax()] < freqs[high.argmax()]


class TestDecode:
    """The one place an encoded file becomes samples."""

    @needs_miniaudio
    def test_decodes_a_wav_to_the_requested_rate(self, tmp_path):
        original = synth.tone(440.0, 0.25, sample_rate=8000)
        path = write_wav(tmp_path / 'tone.wav', original.samples, sample_rate=8000)
        clip = clipmodule.decode_file(str(path), sample_rate=22050)
        assert clip.sample_rate == 22050
        assert clip.duration == pytest.approx(0.25, rel=0.02)

    @needs_miniaudio
    def test_decodes_stereo_content_down_to_one_channel(self, tmp_path):
        interleaved = np.zeros(800, dtype='f')
        interleaved[::2] = 0.5                          # left only
        path = write_wav(tmp_path / 'stereo.wav', interleaved,
                         sample_rate=8000, channels=2)
        clip = clipmodule.decode_file(str(path), sample_rate=8000)
        assert clip.samples.ndim == 1
        assert clip.frames == pytest.approx(400, abs=2)

    @needs_miniaudio
    def test_names_the_clip_after_the_file(self, tmp_path):
        path = write_wav(tmp_path / 'named.wav', np.zeros(100, dtype='f'))
        assert clipmodule.decode_file(str(path)).name.endswith('named.wav')

    @needs_miniaudio
    def test_a_file_that_is_not_audio_raises_decode_error(self, tmp_path):
        path = tmp_path / 'broken.wav'
        path.write_bytes(b'this is not a wave file')
        with pytest.raises(DecodeError):
            clipmodule.decode_file(str(path))

    def test_a_missing_file_raises_decode_error(self, tmp_path):
        with pytest.raises(DecodeError):
            clipmodule.decode_file(str(tmp_path / 'absent.wav'))

    def test_decoding_without_the_backend_raises_decode_error(self, tmp_path, no_backend):
        """The package-absent path is real code, so it is tested rather than assumed."""
        with pytest.raises(DecodeError):
            clipmodule.decode_file(str(tmp_path / 'anything.wav'))

    def test_the_module_says_decoding_is_impossible_without_the_backend(self, no_backend):
        assert clipmodule.decoder_available() is False


class TestDecodeBytes:
    """Encoded audio with no file behind it: a ``.glb``, a ``data:`` URI, a download."""

    @needs_miniaudio
    def test_it_decodes_the_same_content_a_file_would(self, tmp_path):
        original = synth.tone(440.0, 0.25, sample_rate=8000)
        path = write_wav(tmp_path / 'tone.wav', original.samples, sample_rate=8000)
        from_file = clipmodule.decode_file(str(path), sample_rate=8000)
        from_bytes = clipmodule.decode_bytes(
            wav_bytes(original.samples, sample_rate=8000), sample_rate=8000)
        assert np.allclose(from_bytes.samples, from_file.samples, atol=1e-4)

    @needs_miniaudio
    def test_it_resamples_while_decoding_as_the_file_path_does(self):
        clip = clipmodule.decode_bytes(
            wav_bytes(synth.tone(440.0, 0.25, sample_rate=8000).samples,
                      sample_rate=8000), sample_rate=22050)
        assert clip.sample_rate == 22050
        assert clip.duration == pytest.approx(0.25, rel=0.02)

    @needs_miniaudio
    def test_it_mixes_stereo_content_down_to_one_channel(self):
        interleaved = np.zeros(800, dtype='f')
        interleaved[::2] = 0.5                          # left only
        clip = clipmodule.decode_bytes(
            wav_bytes(interleaved, sample_rate=8000, channels=2), sample_rate=8000)
        assert clip.samples.ndim == 1

    @needs_miniaudio
    def test_bytes_that_are_not_audio_raise_decode_error(self):
        with pytest.raises(DecodeError):
            clipmodule.decode_bytes(b'this is not a wave file')

    @needs_miniaudio
    def test_the_name_given_is_the_name_the_clip_carries(self):
        clip = clipmodule.decode_bytes(wav_bytes(np.zeros(100, dtype='f')),
                                       sample_rate=8000, name='audio 0')
        assert clip.name == 'audio 0'

    def test_decoding_without_the_backend_raises_decode_error(self, no_backend):
        with pytest.raises(DecodeError):
            clipmodule.decode_bytes(b'anything')


class TestClipCache:
    """One decode per file, however many times a sound is fired."""

    def test_a_clip_decodes_once_and_is_then_returned_from_the_cache(self):
        """The decoder is a stub, so no file has to exist for this to be true."""
        path = 'sounds/once.wav'
        calls = []

        def counting_decode(name, sample_rate):
            calls.append(name)
            return synth.tone(440.0, 0.01, sample_rate=sample_rate)

        cache = ClipCache(sample_rate=8000, decode=counting_decode)
        first = cache.get(path)
        second = cache.get(path)
        assert first is second
        assert calls == [path]

    def test_a_failing_decode_yields_silence_and_warns_once(self, tmp_path, caplog):
        def failing_decode(name, sample_rate):
            raise DecodeError(name)

        cache = ClipCache(decode=failing_decode)
        with caplog.at_level('WARNING'):
            assert cache.get('missing.wav') is None
            assert cache.get('missing.wav') is None
        assert sum('missing.wav' in record.getMessage()
                   for record in caplog.records) == 1

    def test_a_clip_may_be_put_in_directly(self):
        """Synthesised and procedural sounds need no file behind them."""
        cache = ClipCache()
        clip = synth.tone(440.0, 0.01)
        cache.put('beep', clip)
        assert cache.get('beep') is clip

    def test_clearing_drops_the_decoded_samples(self):
        cache = ClipCache()
        cache.put('beep', synth.tone(440.0, 0.01))
        cache.clear()
        assert len(cache) == 0

    def test_the_cache_reports_what_it_holds(self):
        cache = ClipCache(sample_rate=8000)
        cache.put('a', synth.tone(440.0, 0.01, sample_rate=8000))
        cache.put('b', synth.tone(880.0, 0.02, sample_rate=8000))
        assert len(cache) == 2
        assert cache.frames_held == 8000 * 0.01 + 8000 * 0.02

    def test_a_clip_put_at_the_wrong_rate_is_resampled_to_the_cache_rate(self):
        """The mixer assumes one rate, so nothing at another rate gets in."""
        cache = ClipCache(sample_rate=16000)
        cache.put('a', synth.tone(440.0, 0.01, sample_rate=8000))
        assert cache.get('a').sample_rate == 16000

    def test_the_cache_decodes_at_its_own_rate(self):
        seen = {}

        def recording_decode(name, sample_rate):
            seen['rate'] = sample_rate
            return synth.tone(440.0, 0.01, sample_rate=sample_rate)

        ClipCache(sample_rate=32000, decode=recording_decode).get('x.wav')
        assert seen['rate'] == 32000


class TestNamesAreNotPaths:
    """A name is a key.  The cache never resolves one against the filesystem.

    It matters because a cache that normalised names would be interpreting
    strings, and for a glTF document those strings come from a third party.
    Whatever a name means is settled before it gets here.
    """

    def names(self):
        seen = []

        def recording_decode(name, sample_rate):
            seen.append(name)
            return synth.tone(440.0, 0.01, sample_rate=sample_rate)

        return seen, ClipCache(sample_rate=8000, decode=recording_decode)

    def test_the_decoder_is_handed_the_name_exactly_as_written(self):
        seen, cache = self.names()
        cache.get('../../../../etc/passwd')
        assert seen == ['../../../../etc/passwd']

    def test_two_spellings_of_one_file_are_two_entries(self):
        """The cost of not interpreting: one extra decode, which is cheap."""
        seen, cache = self.names()
        cache.get('sounds/shot.wav')
        cache.get('./sounds/shot.wav')
        assert len(seen) == 2

    def test_a_name_that_is_not_a_path_is_used_as_written(self):
        cache = ClipCache()
        cache.put('weapon/fire', synth.tone(440.0, 0.01))
        assert cache.get('weapon/fire') is not None

    def test_a_registered_name_is_never_handed_to_the_decoder(self):
        seen, cache = self.names()
        cache.put('beep', synth.tone(440.0, 0.01, sample_rate=8000))
        cache.get('beep')
        assert seen == []

    def test_membership_can_be_asked_without_decoding(self):
        seen, cache = self.names()
        cache.put('beep', synth.tone(440.0, 0.01, sample_rate=8000))
        assert 'beep' in cache
        assert 'shot' not in cache
        assert seen == []


class TestCachingBytes:
    """Encoded audio handed straight to the cache, with no file in the way."""

    @needs_miniaudio
    def test_bytes_can_be_registered_under_a_name(self):
        cache = ClipCache(sample_rate=8000)
        assert cache.put_bytes('audio 0', wav_bytes(
            synth.tone(440.0, 0.05, sample_rate=8000).samples, sample_rate=8000))
        assert cache.get('audio 0') is not None

    @needs_miniaudio
    def test_bytes_that_will_not_decode_are_a_silence_and_a_warning(self, caplog):
        cache = ClipCache(sample_rate=8000)
        with caplog.at_level('WARNING'):
            assert cache.put_bytes('audio 0', b'not audio') is None
        assert len(caplog.records) == 1

    @needs_miniaudio
    def test_a_failed_name_is_not_retried(self):
        cache = ClipCache(sample_rate=8000)
        cache.put_bytes('audio 0', b'not audio')
        assert cache.get('audio 0') is None


def test_the_module_reports_whether_decoding_is_possible():
    """Applications ask this to decide whether to offer sound at all."""
    assert isinstance(clipmodule.decoder_available(), bool)


def test_resampling_an_empty_clip_returns_it_unchanged():
    """There is nothing to interpolate between, and no rate to report but its own."""
    empty = Clip([], 8000)
    assert empty.resampled(16000) is empty


def test_silence_is_a_clip_of_zeroes():
    clip = synth.silence(0.5, sample_rate=8000)
    assert clip.frames == 4000
    assert not clip.samples.any()


def test_a_clip_reports_its_peak_for_normalisation():
    clip = Clip([0.0, 0.25, -0.5], 8000)
    assert clip.peak == pytest.approx(0.5)


def test_normalising_scales_the_peak_to_one():
    clip = Clip([0.0, 0.25, -0.5], 8000).normalised()
    assert clip.peak == pytest.approx(1.0)


def test_normalising_silence_leaves_it_silent():
    """Dividing by a zero peak would turn a silent clip into not-a-number."""
    clip = synth.silence(0.01, sample_rate=8000).normalised()
    assert not np.isnan(clip.samples).any()
    assert clip.peak == 0.0


def test_pi_is_not_needed_for_a_zero_length_tone():
    """A zero-length request is a clip with no frames, not an error."""
    assert synth.tone(math.pi, 0.0, sample_rate=8000).frames == 0
