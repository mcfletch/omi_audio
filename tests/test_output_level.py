"""What actually reaches the sound card, for an emitter at a given position.

Every other audio test stops somewhere short of the device: it asserts a gain, a
ratio, or the contents of ``Mixer.mix``.  None of them would notice if the last
step -- the generator the device pulls, and the bytes it hands to the C layer --
produced silence.

So these tests drive the chain **exactly as the device driver does**: prime the
generator, ``send()`` it a frame count, and decode the bytes that come back.
Then they assert a *level*, in dBFS, for an emitter at a stated position
relative to the listener.  A number, at a place, at the boundary.
"""

import math

import numpy as np
import pytest

from omi_audio import model, synth
from omi_audio.device import NullDevice
from omi_audio.engine import AudioEngine
from omi_audio.mixer import Mixer
from omi_audio.spatial import Listener

from support import RATE, dbfs, device_bytes

#: How loud a sound at the reference distance must be, at the device.  A source
#: at full scale, centred, is -3 dBFS after equal-power panning; anything within
#: 3 dB of that has survived the whole chain intact.
FULL_SCALE_DBFS = -6.0
#: Where "you can hear it" stops.  Below this a sound is competing with the room.
AUDIBLE_DBFS = -20.0


class Placed:
    """The test tone looping at a world position, driven as an application drives it.

    Started once and thereafter *re-aimed*, which is the whole point of the
    handle: the level at the device has to follow the emitter without the voice
    ever being restarted.
    """

    def __init__(self, engine, position, gain=1.0, **positional):
        self.engine = engine
        self.emitter = model.AudioEmitter(
            gain=gain, positional=model.PositionalProperties(**positional))
        self.handle = engine.play('test', emitter=self.emitter,
                                  position=position, loop=True)

    def move_to(self, position):
        self.engine.aim(self.handle, self.emitter, position)


@pytest.fixture
def engine(sounding_engine):
    """The shared engine, with the full-scale test tone already in it.

    Named ``engine`` locally so every test in this file reads as being about a
    level rather than about which fixture it asked for; ``sounding_engine`` is
    in ``conftest.py``.
    """
    return sounding_engine


class TestTheDeviceReceivesSound:
    """The bytes the driver copies out, not the array the mixer built."""

    def test_a_source_at_the_reference_distance_arrives_at_full_scale(self, engine):
        Placed(engine, (0.0, 0.0, -1.0), refDistance=1.0)
        assert dbfs(device_bytes(engine)) > FULL_SCALE_DBFS

    def test_an_empty_scene_arrives_as_silence(self, engine):
        assert dbfs(device_bytes(engine)) < -80.0

    def test_the_stream_keeps_producing_block_after_block(self, engine):
        """A generator that yields once and then stops is a click and silence."""
        Placed(engine, (0.0, 0.0, -1.0), refDistance=1.0)
        stream = engine.mixer.blocks()
        next(stream)
        levels = [dbfs(np.frombuffer(bytes(memoryview(stream.send(256)).cast('B')),
                                     dtype=np.float32))
                  for _ in range(20)]
        assert min(levels) > AUDIBLE_DBFS

    def test_the_block_handed_over_is_the_size_that_was_asked_for(self, engine):
        Placed(engine, (0.0, 0.0, -1.0), refDistance=1.0)
        stream = engine.mixer.blocks()
        next(stream)
        raw = memoryview(stream.send(333)).cast('B')
        assert len(raw) == 333 * 2 * 4          # frames x channels x float32

    def test_a_block_bigger_than_the_mixer_expects_is_silence_not_a_crash(self):
        """A device period larger than ``max_block`` must not kill the thread.

        The small mixer is built small rather than shrunk afterwards, because
        ``max_block`` is fixed at construction -- every buffer was sized from
        it, and moving it would produce the silent truncation it exists to
        refuse.
        """
        engine = AudioEngine(device=NullDevice(sample_rate=RATE), voices=4)
        try:
            engine.mixer = Mixer(sample_rate=RATE, voices=4, max_block=128)
            engine.clips.put('test', synth.tone(440.0, 1.0, sample_rate=RATE,
                                                amplitude=1.0, fade=0.0))
            Placed(engine, (0.0, 0.0, -1.0), refDistance=1.0)
            stream = engine.mixer.blocks()
            next(stream)
            raw = bytes(memoryview(stream.send(4096)).cast('B'))
            assert len(raw) == 4096 * 2 * 4, 'the device was handed a short block'
            assert dbfs(np.frombuffer(raw, dtype=np.float32)) < -80.0
        finally:
            engine.close()


class TestLevelAtAPosition:
    """A level, in dBFS, for an emitter at a stated position."""

    #: (label, position, minimum dBFS, maximum dBFS) at refDistance 1.
    PLACES = [
        ('one metre ahead', (0.0, 0.0, -1.0), -6.0, -2.0),
        ('one metre right', (1.0, 0.0, 0.0), -1.0, 0.5),
        ('one metre left', (-1.0, 0.0, 0.0), -1.0, 0.5),
        ('one metre behind', (0.0, 0.0, 1.0), -6.0, -2.0),
        ('two metres ahead', (0.0, 0.0, -2.0), -12.0, -7.0),
        ('four metres ahead', (0.0, 0.0, -4.0), -18.0, -13.0),
        ('sixteen metres ahead', (0.0, 0.0, -16.0), -30.0, -25.0),
    ]

    @pytest.mark.parametrize('label,position,low,high',
                             PLACES, ids=[p[0] for p in PLACES])
    def test_level_at(self, engine, label, position, low, high):
        Placed(engine, position, refDistance=1.0)
        level = dbfs(device_bytes(engine))
        assert low <= level <= high, '%s: %.1f dBFS' % (label, level)

    def test_doubling_the_distance_halves_the_amplitude(self, engine):
        """The inverse model, read at the device rather than in the maths."""
        Placed(engine, (0.0, 0.0, -1.0), refDistance=1.0)
        near = dbfs(device_bytes(engine))
        engine.stop_all()
        Placed(engine, (0.0, 0.0, -2.0), refDistance=1.0)
        far = dbfs(device_bytes(engine))
        assert far - near == pytest.approx(-6.0, abs=1.0)

    def test_a_larger_reference_distance_keeps_a_far_sound_loud(self, engine):
        """The knob to reach for when a sound has to carry across a room."""
        Placed(engine, (0.0, 0.0, -20.0), refDistance=20.0)
        assert dbfs(device_bytes(engine)) > -6.0


class TestWhichEarHearsIt:
    """Panning, measured on the interleaved stream the device is handed."""

    def channels(self, engine):
        interleaved = device_bytes(engine)
        return dbfs(interleaved[0::2]), dbfs(interleaved[1::2])

    def test_a_source_on_the_right_is_louder_in_the_right_channel(self, engine):
        Placed(engine, (4.0, 0.0, 0.0), refDistance=8.0)
        left, right = self.channels(engine)
        assert right > left + 6.0

    def test_a_source_on_the_left_is_louder_in_the_left_channel(self, engine):
        Placed(engine, (-4.0, 0.0, 0.0), refDistance=8.0)
        left, right = self.channels(engine)
        assert left > right + 6.0

    def test_a_source_ahead_is_even(self, engine):
        Placed(engine, (0.0, 0.0, -4.0), refDistance=8.0)
        left, right = self.channels(engine)
        assert abs(left - right) < 0.5

    def test_turning_the_listener_moves_the_sound_to_the_other_ear(self, engine):
        placed = Placed(engine, (4.0, 0.0, 0.0), refDistance=8.0)
        before_left, before_right = self.channels(engine)
        engine.listener = Listener(forward=(0.0, 0.0, 1.0))
        placed.move_to((4.0, 0.0, 0.0))
        after_left, after_right = self.channels(engine)
        assert before_right > before_left
        assert after_left > after_right


class TestMovingWithoutRestarting:
    """A sound that moves must stay audible while it moves."""

    def test_a_source_that_moves_closer_gets_louder(self, engine):
        placed = Placed(engine, (0.0, 0.0, -8.0), refDistance=1.0)
        far = dbfs(device_bytes(engine))
        placed.move_to((0.0, 0.0, -1.0))
        assert dbfs(device_bytes(engine)) > far + 12.0

    def test_a_moving_source_never_drops_out(self, engine):
        """The re-aim writes gains; it must never stop and restart the voice."""
        placed = Placed(engine, (0.0, 0.0, -2.0), refDistance=2.0)
        levels = []
        for step in range(12):
            angle = step * math.pi / 6.0
            placed.move_to((2.0 * math.sin(angle), 0.0, -2.0 * math.cos(angle)))
            levels.append(dbfs(device_bytes(engine, frames=256, blocks=2)))
        assert min(levels) > AUDIBLE_DBFS
        assert engine.active_voices == 1


class TestLoopingSurvives:
    """The demo's sounds all loop; a loop that dies is a demo that goes quiet."""

    def test_a_looping_source_is_still_sounding_after_many_passes(self, engine):
        engine.clips.put('test', synth.tone(440.0, 0.05, sample_rate=RATE,
                                            amplitude=1.0, fade=0.0))
        Placed(engine, (0.0, 0.0, -1.0), refDistance=1.0)
        early = dbfs(device_bytes(engine, frames=512, blocks=4))
        late = dbfs(device_bytes(engine, frames=512, blocks=60))
        assert early > FULL_SCALE_DBFS
        assert late > FULL_SCALE_DBFS

    def test_a_looping_source_never_goes_silent_between_passes(self, engine):
        engine.clips.put('test', synth.tone(440.0, 0.02, sample_rate=RATE,
                                            amplitude=1.0, fade=0.0))
        Placed(engine, (0.0, 0.0, -1.0), refDistance=1.0)
        stream = engine.mixer.blocks()
        next(stream)
        for _ in range(40):
            block = np.frombuffer(bytes(memoryview(stream.send(64)).cast('B')),
                                  dtype=np.float32)
            assert dbfs(block) > AUDIBLE_DBFS
