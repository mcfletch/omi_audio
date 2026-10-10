"""Capture: an input's samples into a ring, or silence where there is no input."""

import numpy as np
import pytest

from omi_audio import _backend, graph
from omi_audio.device import (
    DeviceError, MiniaudioCapture, NullCapture, capture_devices, open_capture,
)

from support import RATE


class FakeCaptureDevice:
    """Stands in for ``miniaudio.CaptureDevice``."""

    def __init__(self, backend='ALSA', fail=False, **named):
        if fail:
            raise RuntimeError('no input')
        self.backend = backend
        self.named = named
        self.sink = None
        self.closed = False

    def start(self, sink):
        self.sink = sink

    def close(self):
        self.closed = True


class FakeDevices:
    def __init__(self, names, fail=False):
        self.names = names
        self.fail = fail

    def get_captures(self):
        if self.fail:
            raise RuntimeError('cannot enumerate')
        return [{'name': name, 'id': 'id-%d' % (index,)}
                for index, name in enumerate(self.names)]


class FakeBackend:
    class SampleFormat:
        FLOAT32 = 'float32'

    def __init__(self, backend='ALSA', fail=False, names=('Built-in Microphone',),
                 enumerate_fails=False):
        self._backend = backend
        self._fail = fail
        self._names = names
        self._enumerate_fails = enumerate_fails
        self.opened = []

    def CaptureDevice(self, **named):  # noqa: N802 - the name miniaudio gives it
        device = FakeCaptureDevice(backend=self._backend, fail=self._fail, **named)
        self.opened.append(device)
        return device

    def Devices(self):  # noqa: N802 - the name miniaudio gives it
        return FakeDevices(self._names, self._enumerate_fails)


@pytest.fixture
def fake_backend(monkeypatch):
    def install(**named):
        backend = FakeBackend(**named)
        monkeypatch.setattr(_backend, 'backend', lambda: backend)
        return backend
    return install


class TestMiniaudioCapture:
    def test_captured_chunks_arrive_in_the_ring(self, fake_backend):
        backend = fake_backend()
        capture = MiniaudioCapture(sample_rate=RATE)
        capture.start()
        sink = backend.opened[0].sink
        sink.send(np.array([0.25, -0.5], dtype=np.float32).tobytes())
        target = np.zeros(3, dtype=np.float32)
        assert capture.ring.read_into(target) == 2
        assert np.allclose(target, [0.25, -0.5, 0.0])

    def test_it_asks_for_one_channel_of_floats(self, fake_backend):
        backend = fake_backend()
        MiniaudioCapture(sample_rate=RATE)
        named = backend.opened[0].named
        assert named['nchannels'] == 1 and named['input_format'] == 'float32'
        assert named['sample_rate'] == RATE and named['device_id'] is None

    def test_a_named_input_is_chosen_by_part_of_its_name(self, fake_backend):
        backend = fake_backend(names=('Line In', 'USB Microphone'))
        MiniaudioCapture(device='usb')
        assert backend.opened[0].named['device_id'] == 'id-1'

    def test_an_input_nobody_has_is_refused(self, fake_backend):
        fake_backend()
        with pytest.raises(DeviceError):
            MiniaudioCapture(device='Theremin')

    def test_an_input_that_will_not_open_is_refused(self, fake_backend):
        fake_backend(fail=True)
        with pytest.raises(DeviceError):
            MiniaudioCapture()

    def test_the_null_backend_is_no_input(self, fake_backend):
        backend = fake_backend(backend='Null')
        with pytest.raises(DeviceError):
            MiniaudioCapture()
        assert backend.opened[0].closed

    @pytest.mark.usefixtures('no_backend')
    def test_no_package_is_no_input(self):
        with pytest.raises(DeviceError):
            MiniaudioCapture()

    def test_closing_twice_is_safe(self, fake_backend):
        backend = fake_backend()
        capture = MiniaudioCapture()
        capture.close()
        capture.close()
        assert backend.opened[0].closed


class TestOpenCapture:
    def test_an_input_is_opened_where_there_is_one(self, fake_backend):
        fake_backend()
        capture = open_capture(RATE)
        assert isinstance(capture, MiniaudioCapture) and not capture.silent

    def test_no_input_is_silence(self, fake_backend, caplog):
        fake_backend(fail=True)
        capture = open_capture(RATE)
        assert isinstance(capture, NullCapture) and capture.silent
        assert 'capturing silence' in caplog.text

    def test_an_unexpected_failure_is_silence(self, monkeypatch):
        def broken():
            raise OSError('driver fell over')
        monkeypatch.setattr(_backend, 'backend', broken)
        assert isinstance(open_capture(RATE), NullCapture)

    @pytest.mark.usefixtures('no_backend')
    def test_the_ring_given_is_used(self):
        ring = graph.CaptureRing(16)
        capture = open_capture(RATE, ring=ring)
        assert capture.ring is ring
        capture.start()
        capture.close()


class TestCaptureDevices:
    def test_the_names(self, fake_backend):
        fake_backend(names=('A', 'B'))
        assert capture_devices() == ['A', 'B']

    @pytest.mark.usefixtures('no_backend')
    def test_none_without_the_package(self):
        assert capture_devices() == []

    def test_a_failed_listing_is_empty(self, fake_backend):
        fake_backend(enumerate_fails=True)
        assert capture_devices() == []


def test_a_capture_source_plays_the_device_ring(fake_backend):
    backend = fake_backend()
    capture = open_capture(RATE)
    capture.start()
    backend.opened[0].sink.send(np.full(4, 0.5, dtype=np.float32).tobytes())
    source = graph.CaptureSource(capture.ring)
    made = graph.Graph(RATE)
    made.add_output(source)
    made.render(6)
    assert np.allclose(source.output(6)[0], [0.5] * 4 + [0.0] * 2)
