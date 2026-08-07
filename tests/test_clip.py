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

    def test_a_tone_is_a_pure_sine_by_default(self):
        """One partial and nothing above it, which is what a sine is."""
        clip = synth.tone(500.0, 0.5, sample_rate=16000, fade=0.0)
        spectrum = np.abs(np.fft.rfft(clip.samples))
        freqs = np.fft.rfftfreq(clip.frames, 1.0 / clip.sample_rate)
        above = spectrum[freqs > 750.0].max()
        assert above < spectrum.max() * 0.01

    def test_harmonics_put_energy_above_the_fundamental(self):
        """A low-pass can only change the *level* of a sine, never its timbre --
        there is nothing above the fundamental for it to take away.  Anything
        demonstrating a filter needs partials."""
        clip = synth.tone(500.0, 0.5, sample_rate=16000, fade=0.0, harmonics=4)
        spectrum = np.abs(np.fft.rfft(clip.samples))
        freqs = np.fft.rfftfreq(clip.frames, 1.0 / clip.sample_rate)
        for partial in (2, 3, 4):
            near = np.abs(freqs - 500.0 * partial) < 20.0
            assert spectrum[near].max() > spectrum.max() * 0.1, (
                'partial %d is missing' % partial)

    def test_harmonics_keep_the_fundamental_the_strongest_partial(self):
        clip = synth.tone(500.0, 0.5, sample_rate=16000, fade=0.0, harmonics=6)
        spectrum = np.abs(np.fft.rfft(clip.samples))
        freqs = np.fft.rfftfreq(clip.frames, 1.0 / clip.sample_rate)
        assert freqs[spectrum.argmax()] == pytest.approx(500.0, abs=5.0)

    def test_harmonics_do_not_push_the_clip_past_full_scale(self):
        """Summed partials would clip if the amplitude were not shared out."""
        clip = synth.tone(500.0, 0.5, sample_rate=16000, amplitude=0.9,
                          harmonics=8)
        assert float(np.abs(clip.samples).max()) <= 0.9 + 1e-6

    def test_a_partial_above_nyquist_is_left_out_rather_than_aliased(self):
        """A partial that will not fit folds back as an out-of-tune whistle."""
        clip = synth.tone(3000.0, 0.5, sample_rate=16000, fade=0.0, harmonics=6)
        spectrum = np.abs(np.fft.rfft(clip.samples))
        freqs = np.fft.rfftfreq(clip.frames, 1.0 / clip.sample_rate)
        # 3rd partial (9 kHz) and up exceed the 8 kHz Nyquist; nothing may
        # appear at the frequencies they would alias to (16000 - 9000 = 7 kHz).
        near = np.abs(freqs - 7000.0) < 50.0
        assert spectrum[near].max() < spectrum.max() * 0.01

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
                     synth.rumble(0.0, sample_rate=8000),
                     synth.silence(0.0, sample_rate=8000)):
            assert clip.frames == 0

    def test_chirp_sweeps_from_one_frequency_to_another(self):
        clip = synth.chirp(200.0, 2000.0, 0.5, sample_rate=16000)
        half = clip.frames // 2
        low = np.abs(np.fft.rfft(clip.samples[:half]))
        high = np.abs(np.fft.rfft(clip.samples[half:]))
        freqs = np.fft.rfftfreq(half, 1.0 / clip.sample_rate)
        assert freqs[low.argmax()] < freqs[high.argmax()]


def centroid(clip):
    """The centre of gravity of a clip's spectrum, in hertz.

    One number for "how bright is this", which is the property a rumble is
    defined by and the one a listener names first.
    """
    spectrum = np.abs(np.fft.rfft(clip.samples))
    freqs = np.fft.rfftfreq(clip.frames, 1.0 / clip.sample_rate)
    return float((freqs * spectrum).sum() / max(spectrum.sum(), 1e-12))


class TestRumble:
    """The low end: a motor, and the thump of something going off.

    :func:`~omi_audio.synth.impact` is white noise, so it is a *crack*
    however long it is left to decay -- there is nothing below a few kilohertz
    in it to hear.  A rumble is the other half of the range: noise with the
    top taken off, over a tone that falls as it goes, which is what an
    explosion is made of.
    """

    def test_it_is_far_lower_than_an_impact(self):
        low = synth.rumble(0.5, sample_rate=16000, seed=1)
        bright = synth.impact(0.5, sample_rate=16000, seed=1)
        assert centroid(low) < centroid(bright) * 0.25

    def test_the_cutoff_is_what_decides_how_low(self):
        """Declared in hertz, and it means what it says."""
        deep = synth.rumble(0.5, sample_rate=16000, seed=1, cutoff=120.0)
        wider = synth.rumble(0.5, sample_rate=16000, seed=1, cutoff=900.0)
        assert centroid(deep) < centroid(wider)

    def test_its_body_falls_in_pitch_as_it_goes(self):
        """What makes a bang read as a bang rather than as a hum."""
        clip = synth.rumble(0.8, sample_rate=16000, seed=1, pitch=90.0,
                            pitch_end=30.0, decay=0.5, tone=1.0)
        half = clip.frames // 2
        freqs = np.fft.rfftfreq(half, 1.0 / clip.sample_rate)
        first = np.abs(np.fft.rfft(clip.samples[:half]))
        second = np.abs(np.fft.rfft(clip.samples[half:]))
        assert freqs[second.argmax()] < freqs[first.argmax()]

    def test_it_decays_to_silence(self):
        clip = synth.rumble(0.6, sample_rate=16000, seed=1, decay=8.0)
        head = float(np.abs(clip.samples[:200]).mean())
        tail = float(np.abs(clip.samples[-200:]).mean())
        assert tail < head * 0.1

    def test_an_attack_starts_it_softly(self):
        """A motor spooling up, rather than a hit.  Zero is the hit."""
        soft = synth.rumble(0.5, sample_rate=16000, seed=1, attack=0.15)
        hard = synth.rumble(0.5, sample_rate=16000, seed=1, attack=0.0)
        assert float(np.abs(soft.samples[:100]).mean()) \
            < float(np.abs(hard.samples[:100]).mean()) * 0.5

    def test_drive_puts_harmonics_over_the_body_tone(self):
        """The growl: a saturated tone has partials a clean one has not."""
        clean = synth.rumble(0.5, sample_rate=16000, seed=1, pitch=80.0,
                             pitch_end=80.0, tone=1.0, decay=0.0, drive=1.0)
        driven = synth.rumble(0.5, sample_rate=16000, seed=1, pitch=80.0,
                              pitch_end=80.0, tone=1.0, decay=0.0, drive=12.0)
        freqs = np.fft.rfftfreq(clean.frames, 1.0 / clean.sample_rate)
        above = (freqs > 150.0) & (freqs < 600.0)
        def share( clip ):
            return (np.abs(np.fft.rfft(clip.samples))[above].sum()
                              / max(np.abs(np.fft.rfft(clip.samples)).sum(), 1e-12))
        assert share(driven) > share(clean) * 2.0

    def test_it_never_goes_past_the_amplitude_asked_for(self):
        """Two sources and a saturator, so the sum has to be brought back."""
        clip = synth.rumble(0.5, sample_rate=16000, seed=1, amplitude=0.8,
                            drive=20.0)
        assert float(np.abs(clip.samples).max()) <= 0.8 + 1e-6

    def test_it_is_reproducible_from_its_seed(self):
        first = synth.rumble(0.2, sample_rate=8000, seed=3)
        second = synth.rumble(0.2, sample_rate=8000, seed=3)
        assert np.array_equal(first.samples, second.samples)

    def under(self, clip, hertz):
        spectrum = np.abs(np.fft.rfft(clip.samples)) ** 2
        freqs = np.fft.rfftfreq(clip.frames, 1.0 / clip.sample_rate)
        return float(spectrum[freqs < hertz].sum() / max(spectrum.sum(), 1e-20))

    def test_tilt_moves_the_noise_energy_downward(self):
        """A blast is not white: its noise rises toward the bottom.

        This is the difference between weight and *pitch*.  Weight from a tone
        is a drum however quiet the tone is -- a low sine with a hard attack is
        exactly what a drum is -- and weight from noise is a thump.
        """
        white = synth.rumble(0.2, sample_rate=16000, seed=5, tone=0.0,
                             cutoff=6000.0)
        tilted = synth.rumble(0.2, sample_rate=16000, seed=5, tone=0.0,
                              cutoff=6000.0, tilt=-4.0)
        assert self.under(tilted, 400.0) > self.under(white, 400.0) * 3.0

    def test_more_tilt_is_more_of_it(self):
        shares = [self.under(synth.rumble(0.2, sample_rate=16000, seed=5,
                                          tone=0.0, cutoff=6000.0, tilt=tilt),
                             400.0)
                  for tilt in (0.0, -2.0, -4.0, -6.0)]
        assert all(more > less for less, more in zip(shares, shares[1:], strict=False))

    def test_a_floor_takes_the_bottom_off_as_the_cutoff_takes_the_top(self):
        """The two together are a band, which is what anything hollow is.

        A tube resonates at a pitch and has very little under it; without a
        bottom edge the same sound is a thump with the tube missing.
        """
        wide = synth.rumble(0.2, sample_rate=16000, seed=6, tone=0.0,
                            cutoff=500.0)
        banded = synth.rumble(0.2, sample_rate=16000, seed=6, tone=0.0,
                              cutoff=500.0, floor=250.0)
        assert self.under(banded, 150.0) < self.under(wide, 150.0) * 0.5
        assert self.under(banded, 800.0) > 0.7      # and still down there

    def test_no_floor_is_the_noise_it_always_was(self):
        assert np.array_equal(
            synth.rumble(0.2, sample_rate=8000, seed=5, floor=0.0).samples,
            synth.rumble(0.2, sample_rate=8000, seed=5).samples)

    def test_no_tilt_is_the_noise_it_always_was(self):
        assert np.array_equal(
            synth.rumble(0.2, sample_rate=8000, seed=5, tilt=0.0).samples,
            synth.rumble(0.2, sample_rate=8000, seed=5).samples)


class TestReverberation:
    """A dense tail rather than a handful of returns.

    The difference is not subtle and it is not a matter of degree: discrete
    repeats are heard as *repeats* -- a clap, and then another clap -- and a
    room is heard as one sound going on.  A rifle recorded outdoors keeps
    within a few decibels of its peak for the better part of a second and
    darkens as it goes, and nothing built out of three taps can be that.

    Baked into the clip at the moment it is made, so this costs the audio
    thread nothing and is not the bus effect the mixer still does not have.
    """

    def burst(self, rate=16000):
        return synth.impact(0.02, sample_rate=rate, seed=1, decay=200.0)

    def level(self, clip, when, window=0.03):
        at = int(when * clip.sample_rate)
        part = clip.samples[at:at + int(window * clip.sample_rate)]
        return float(np.sqrt((part.astype('d') ** 2).mean())) if len(part) else 0.0

    def brightness(self, clip, when, window=0.06):
        at = int(when * clip.sample_rate)
        part = clip.samples[at:at + int(window * clip.sample_rate)]
        spectrum = np.abs(np.fft.rfft(part)) ** 2
        freqs = np.fft.rfftfreq(len(part), 1.0 / clip.sample_rate)
        return float((freqs * spectrum).sum() / max(spectrum.sum(), 1e-20))

    def test_the_sound_goes_on_after_it_has_stopped(self):
        wet = synth.reverberated(self.burst(), seconds=0.6, level=0.7, seed=2)
        assert self.level(wet, 0.3) > self.level(wet, 0.02) * 0.02

    def test_it_dies_away_rather_than_holding(self):
        wet = synth.reverberated(self.burst(), seconds=0.4, level=0.7, seed=2)
        assert self.level(wet, 0.35) < self.level(wet, 0.10)

    def test_a_longer_tail_is_still_going_when_a_shorter_one_has_gone(self):
        short = synth.reverberated(self.burst(), seconds=0.25, level=0.7, seed=2)
        long = synth.reverberated(self.burst(), seconds=0.9, level=0.7, seed=2)
        assert self.level(long, 0.4) > self.level(short, 0.4) * 3.0

    def test_the_tail_darkens_as_it_goes(self):
        """Air and soft surfaces take the top first, which is what makes it a room."""
        wet = synth.reverberated(self.burst(), seconds=0.9, level=0.8, seed=2)
        assert self.brightness(wet, 0.5) < self.brightness(wet, 0.05)

    def test_it_never_makes_a_sound_louder(self):
        dry = synth.rumble(0.2, sample_rate=16000, seed=3, amplitude=0.9)
        wet = synth.reverberated(dry, seconds=0.5, level=1.0, seed=2)
        assert float(np.abs(wet.samples).max()) \
            <= float(np.abs(dry.samples).max()) + 1e-6

    def test_no_level_is_the_clip_itself(self):
        dry = self.burst()
        assert synth.reverberated(dry, seconds=0.5, level=0.0) is dry

    def test_a_clip_with_no_frames_survives_it(self):
        assert synth.reverberated(synth.silence(0.0, sample_rate=8000),
                                  seconds=0.4, level=0.5).frames == 0

    def test_it_is_reproducible_from_its_seed(self):
        first = synth.reverberated(self.burst(), seconds=0.3, level=0.6, seed=4)
        second = synth.reverberated(self.burst(), seconds=0.3, level=0.6, seed=4)
        assert np.array_equal(first.samples, second.samples)


class TestEcho:
    """A sound repeating quieter, which is the cheapest thing that says *space*.

    Not reverb -- there is no room here to model -- but a slap-back, which is
    what a hard, sharp sound in a large place actually gives back and what tells
    a listener that the sound was hard in the first place.
    """

    def burst(self, rate=8000):
        """A short hit, over well before any echo of it starts."""
        return synth.impact(0.02, sample_rate=rate, seed=1, decay=200.0)

    def peaks(self, clip, delay, taps):
        """The loudest sample in the window each repeat lands in."""
        step = int(delay * clip.sample_rate)
        return [float(np.abs(clip.samples[at * step:(at + 1) * step]).max())
                for at in range(taps + 1)]

    def test_the_sound_comes_back_quieter_each_time(self):
        echoed = synth.echoed(self.burst(), delay=0.1, level=0.5, taps=3)
        heard = self.peaks(echoed, delay=0.1, taps=3)
        assert all(later < earlier for earlier, later in zip(heard, heard[1:], strict=False))

    def test_each_repeat_is_the_level_asked_for(self):
        echoed = synth.echoed(self.burst(), delay=0.1, level=0.5, taps=2)
        first, second, third = self.peaks(echoed, delay=0.1, taps=2)
        assert second == pytest.approx(first * 0.5, rel=0.02)
        assert third == pytest.approx(first * 0.25, rel=0.02)

    def test_the_repeats_arrive_when_they_are_asked_to(self):
        rate = 8000
        echoed = synth.echoed(self.burst(rate), delay=0.05, level=0.6, taps=1)
        quiet = echoed.samples[int(0.03 * rate):int(0.045 * rate)]
        assert float(np.abs(quiet).max()) < 1e-4      # nothing in between
        assert float(np.abs(echoed.samples[int(0.05 * rate):]).max()) > 0.01

    def test_it_makes_room_for_the_tail(self):
        original = self.burst()
        echoed = synth.echoed(original, delay=0.1, level=0.5, taps=3)
        assert echoed.duration > original.duration + 0.29

    def test_an_echo_never_makes_a_sound_louder(self):
        """Overlapping repeats must not push a clip past where it started."""
        original = synth.rumble(0.4, sample_rate=8000, seed=2, amplitude=0.9)
        echoed = synth.echoed(original, delay=0.02, level=0.9, taps=6)
        assert float(np.abs(echoed.samples).max()) \
            <= float(np.abs(original.samples).max()) + 1e-6

    def test_no_level_is_the_clip_itself(self):
        """So a voice that declares no echo pays nothing and is unchanged."""
        original = self.burst()
        assert synth.echoed(original, delay=0.1, level=0.0) is original

    def test_a_clip_with_no_frames_survives_it(self):
        assert synth.echoed(synth.silence(0.0, sample_rate=8000),
                            delay=0.1, level=0.5).frames == 0

    def test_the_rate_is_kept(self):
        echoed = synth.echoed(self.burst(22050), delay=0.05, level=0.5)
        assert echoed.sample_rate == 22050

    def brightness(self, samples, rate):
        spectrum = np.abs(np.fft.rfft(samples)) ** 2
        freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)
        return float((freqs * spectrum).sum() / max(spectrum.sum(), 1e-20))

    def test_damping_darkens_each_repeat(self):
        """Air and soft surfaces take the top off, and take more of it each time."""
        rate = 16000
        source = synth.noise(0.04, sample_rate=rate, seed=4, amplitude=0.8)
        echoed = synth.echoed(source, delay=0.1, level=0.9, taps=3,
                              damping=900.0)
        step = int(0.1 * rate)
        heard = [self.brightness(echoed.samples[at * step:at * step + int(0.04 * rate)],
                                 rate) for at in range(4)]
        assert all(later < earlier for earlier, later in zip(heard, heard[1:], strict=False))

    def test_thinning_takes_the_bottom_out_of_each_repeat(self):
        """The other end, and the reason a return is not a second gunshot.

        What makes a report *heavy* is a near-field thump that never comes
        back off anything; a repeat carrying it reads as somebody firing again
        rather than as the first shot answering.
        """
        rate = 16000
        source = synth.rumble(0.04, sample_rate=rate, seed=4, cutoff=3000.0,
                              pitch=60.0, pitch_end=60.0, tone=0.5)
        echoed = synth.echoed(source, delay=0.1, level=0.9, taps=2,
                              thinning=600.0)
        step = int(0.1 * rate)
        heard = [self.brightness(echoed.samples[at * step:at * step + int(0.04 * rate)],
                                 rate) for at in range(3)]
        assert all(later > earlier for earlier, later in zip(heard, heard[1:],strict=False))

    def test_undamped_repeats_are_the_same_sound_again(self):
        rate = 16000
        source = synth.noise(0.04, sample_rate=rate, seed=4, amplitude=0.8)
        echoed = synth.echoed(source, delay=0.1, level=0.9, taps=1)
        step = int(0.1 * rate)
        first = self.brightness(echoed.samples[:int(0.04 * rate)], rate)
        second = self.brightness(echoed.samples[step:step + int(0.04 * rate)], rate)
        assert second == pytest.approx(first, rel=0.05)


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
