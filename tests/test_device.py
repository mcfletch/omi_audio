"""The device seam: where mixed blocks go, and what happens when nowhere.

Two ways audio can be unavailable -- the package is not installed, and no device
opens -- and both must end in one warning and a silent run.  Both are exercised
here, because they are the paths a developer with working sound never sees.

The last class in the file is different in kind: it drives the **real**
``miniaudio``, because everything else here stands in for it, and a stand-in
cannot notice that what the mixer yields is a shape the C layer will not take.
"""

import time

import numpy as np
import pytest

from omi_audio import _backend
from omi_audio import device as devicemodule
from omi_audio.device import DeviceError, NullDevice, describe, open_device
from omi_audio.mixer import Mixer

from support import needs_miniaudio


class FakePlaybackDevice:
    """Stands in for ``miniaudio.PlaybackDevice`` with no hardware behind it."""

    instances = []

    def __init__(self, backend='ALSA', fail=False, **named):
        if fail:
            raise RuntimeError('no device')
        self.backend = backend
        self.named = named
        self.generator = None
        self.closed = False
        FakePlaybackDevice.instances.append(self)

    def start(self, generator):
        self.generator = generator

    def stop(self):
        self.generator = None

    def close(self):
        self.closed = True


class FakeBackend:
    """A stand-in ``miniaudio`` module: just what the device seam touches."""

    class SampleFormat:
        FLOAT32 = 'float32'

    def __init__(self, backend='ALSA', fail=False):
        self._backend = backend
        self._fail = fail

    def PlaybackDevice(self, **named):
        return FakePlaybackDevice(backend=self._backend, fail=self._fail, **named)


@pytest.fixture(autouse=True)
def clear_instances():
    FakePlaybackDevice.instances = []
    yield
    FakePlaybackDevice.instances = []


@pytest.fixture
def fake_backend(monkeypatch):
    """Install a stand-in backend: ``fake_backend()``, or ``fake_backend(fail=True)``."""
    def install(**named):
        backend = FakeBackend(**named)
        monkeypatch.setattr(_backend, 'backend', lambda: backend)
        return backend
    return install


class TestNullDevice:
    """Silence is a backend, not an error path bolted on the side."""

    def test_it_reports_itself_as_silent(self):
        assert NullDevice().silent is True

    def test_it_has_the_rate_and_channels_it_was_asked_for(self):
        device = NullDevice(sample_rate=22050, channels=2)
        assert (device.sample_rate, device.channels) == (22050, 2)

    def test_starting_and_stopping_are_safe_and_repeatable(self):
        device = NullDevice()
        mixer = Mixer(sample_rate=device.sample_rate)
        device.start(mixer.blocks())
        device.start(mixer.blocks())
        device.stop()
        device.stop()
        device.close()
        device.close()

    def test_it_never_pulls_from_the_mixer(self):
        """No thread, no callback, no cost: a silent run costs nothing."""
        pulled = []

        def watching():
            frames = yield None
            while True:
                pulled.append(frames)
                frames = yield None

        device = NullDevice()
        device.start(watching())
        assert pulled == []

    def test_it_is_not_running_until_started(self):
        device = NullDevice()
        assert device.running is False
        device.start(Mixer().blocks())
        assert device.running is True
        device.stop()
        assert device.running is False


class TestMiniaudioDevice:
    """The real backend, driven through a stand-in for the C library."""

    def test_it_asks_for_float32_stereo_at_the_mixers_rate(self, fake_backend):
        fake_backend()
        device = devicemodule.MiniaudioDevice(sample_rate=22050, channels=2)
        named = FakePlaybackDevice.instances[0].named
        assert named['sample_rate'] == 22050
        assert named['nchannels'] == 2
        assert named['output_format'] is FakeBackend.SampleFormat.FLOAT32
        assert device.silent is False

    def test_starting_primes_the_generator_before_handing_it_over(self, fake_backend):
        """A device sends into the generator, so it must already be started."""
        fake_backend()
        device = devicemodule.MiniaudioDevice(sample_rate=8000)
        mixer = Mixer(sample_rate=8000)
        device.start(mixer.blocks())
        block = FakePlaybackDevice.instances[0].generator.send(16)
        assert np.asarray(block).shape == (16, 2)

    def test_closing_closes_the_underlying_device(self, fake_backend):
        fake_backend()
        devicemodule.MiniaudioDevice(sample_rate=8000).close()
        assert FakePlaybackDevice.instances[0].closed is True

    def test_stopping_tells_the_underlying_device_to_stop(self, fake_backend):
        fake_backend()
        device = devicemodule.MiniaudioDevice(sample_rate=8000)
        device.start(Mixer(sample_rate=8000).blocks())
        device.stop()
        assert FakePlaybackDevice.instances[0].generator is None
        assert device.running is False

    def test_stopping_a_device_that_never_started_is_harmless(self, fake_backend):
        fake_backend()
        devicemodule.MiniaudioDevice(sample_rate=8000).stop()

    def test_closing_twice_is_harmless(self, fake_backend):
        fake_backend()
        device = devicemodule.MiniaudioDevice(sample_rate=8000)
        device.close()
        device.close()

    def test_a_device_that_will_not_open_raises_device_error(self, fake_backend):
        fake_backend(fail=True)
        with pytest.raises(DeviceError):
            devicemodule.MiniaudioDevice(sample_rate=8000)

    def test_the_backends_own_null_output_counts_as_no_device(self, fake_backend):
        """A machine with no sound card gets miniaudio's null backend.

        That is the no-device case wearing a disguise; treating it as a device
        would spin an audio thread that can never be heard.
        """
        fake_backend(backend='Null')
        with pytest.raises(DeviceError):
            devicemodule.MiniaudioDevice(sample_rate=8000)

    def test_without_the_package_it_raises_device_error(self, no_backend):
        with pytest.raises(DeviceError):
            devicemodule.MiniaudioDevice(sample_rate=8000)


class TestOpenDevice:
    """The one call an application makes, which can never fail."""

    def test_it_returns_the_real_device_when_one_opens(self, fake_backend):
        fake_backend()
        assert open_device(sample_rate=8000).silent is False

    def test_a_missing_package_gives_silence_and_one_warning(self, no_backend, caplog):
        with caplog.at_level('WARNING'):
            device = open_device(sample_rate=8000)
        assert device.silent is True
        assert len(caplog.records) == 1
        assert 'miniaudio' in caplog.records[0].getMessage()

    def test_a_device_that_will_not_open_gives_silence_and_one_warning(
            self, fake_backend, caplog):
        fake_backend(fail=True)
        with caplog.at_level('WARNING'):
            device = open_device(sample_rate=8000)
        assert device.silent is True
        assert len(caplog.records) == 1

    def test_the_silent_device_keeps_the_requested_rate(self, no_backend):
        """The mixer is built around the device's rate, so it must be honest."""
        assert open_device(sample_rate=22050).sample_rate == 22050

    def test_opening_never_raises_when_the_import_itself_explodes(self, monkeypatch):
        """A broken install fails on import, not on ``PlaybackDevice``, and
        lands outside the DeviceError the rest of this module raises."""
        def exploding():
            raise RuntimeError('this miniaudio build is broken')

        monkeypatch.setattr(_backend, 'backend', exploding)
        assert open_device(sample_rate=8000).silent is True

    def test_opening_never_raises_whatever_the_backend_does(self, monkeypatch):
        class Exploding:
            def __getattr__(self, name):
                raise RuntimeError('this backend is broken')

        monkeypatch.setattr(_backend, 'backend', lambda: Exploding())
        assert open_device(sample_rate=8000).silent is True


class TestAvailability:
    def test_availability_is_reported_as_a_boolean(self):
        assert isinstance(devicemodule.miniaudio_available(), bool)

    def test_availability_follows_the_backend(self, no_backend):
        assert devicemodule.miniaudio_available() is False


class TestDescribe:
    """One line for a start-up log or a debug overlay."""

    def test_no_device_at_all(self):
        assert describe(None) == 'audio: none'

    def test_a_silent_device_says_so(self):
        assert describe(NullDevice(sample_rate=8000)) == 'audio: silent'

    def test_a_real_device_names_its_backend_rate_and_channels(self, fake_backend):
        fake_backend()
        device = devicemodule.MiniaudioDevice(sample_rate=22050, channels=2)
        assert describe(device) == 'audio: ALSA 22050 Hz x2'

    def test_a_device_that_never_named_a_backend_still_describes(self):
        """``AudioDevice`` itself has no ``backend_name``; it must not raise."""
        assert describe(devicemodule.AudioDevice(sample_rate=8000)) == (
            'audio: device 8000 Hz x2')


@needs_miniaudio
class TestTheRealBackendAcceptsWhatTheMixerYields:
    """The one integration a stand-in cannot check.

    Everything above replaces ``miniaudio`` with a fake, so nothing above would
    notice if :meth:`~omi_audio.mixer.Mixer.blocks` yielded a shape the C layer
    refuses -- and playback would break with a green suite.  These drive the
    real library.
    """

    def test_a_yielded_block_converts_to_exactly_the_bytes_a_period_needs(self):
        """``miniaudio`` memmoves what this returns straight into the device."""
        import miniaudio

        mixer = Mixer(sample_rate=8000, max_block=512)
        stream = mixer.blocks()
        next(stream)
        for frames in (1, 32, 256, 512):
            raw = miniaudio._bytes_from_generator_samples(stream.send(frames))
            assert len(raw) == frames * 2 * 4, 'a %d-frame block short-changed the device' % (
                frames,)

    def test_an_oversized_block_also_converts_to_a_full_period(self):
        """The path that used to hand back a short buffer of stale audio."""
        import miniaudio

        mixer = Mixer(sample_rate=8000, max_block=64)
        stream = mixer.blocks()
        next(stream)
        raw = miniaudio._bytes_from_generator_samples(stream.send(1024))
        assert len(raw) == 1024 * 2 * 4
        assert not any(raw)

    def test_a_real_device_runs_the_mixer_on_its_own_audio_thread(self):
        """miniaudio's own null backend is a genuine device with a genuine
        thread and no hardware, so the whole hand-off runs in CI."""
        import miniaudio

        from omi_audio import synth

        mixer = Mixer(sample_rate=8000, max_block=4096)
        mixer.play(synth.tone(440.0, 5.0, sample_rate=8000, fade=0.0), loop=True)
        stream = mixer.blocks()
        next(stream)
        device = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.FLOAT32, nchannels=2,
            sample_rate=8000, buffersize_msec=20,
            backends=[miniaudio.Backend.NULL])
        try:
            device.start(stream)
            deadline = 200
            while mixer.voices[0].position == 0.0 and deadline:
                deadline -= 1
                time.sleep(0.005)
            assert mixer.voices[0].position > 0.0, (
                'the device never pulled a block from the mixer')
        finally:
            device.close()
