"""The engine: the one object an application holds.

It ties the model, the clips, the spatial maths, the mixer and the device
together, and it is where "a sound at a place in the world" becomes "these two
numbers on this voice".
"""

import math

import numpy as np
import pytest

from omi_audio import model, synth
from omi_audio.device import NullDevice
from omi_audio.engine import AudioEngine
from omi_audio.library import AudioLibrary
from omi_audio.spatial import Listener

from support import RATE, beep


def supplying(clip):
    """A ``fetch`` that answers every ask with ``clip``, and records the asks."""
    asked = []

    def fetch(library, index, audio):
        asked.append((index, audio))
        library.supply(index, clip)

    fetch.asked = asked
    return fetch


class TestConstruction:
    def test_the_mixer_runs_at_the_devices_rate(self, engine):
        assert engine.mixer.sample_rate == engine.device.sample_rate == RATE

    def test_the_engine_reports_the_rate_the_chain_runs_at(self, engine):
        assert engine.sample_rate == RATE

    def test_a_silent_device_makes_a_silent_engine(self, engine):
        assert engine.silent is True

    def test_it_opens_a_device_for_itself_when_given_none(self):
        made = AudioEngine()
        try:
            assert made.device is not None
        finally:
            made.close()

    def test_closing_stops_every_voice(self, engine):
        engine.play(beep(), loop=True)
        engine.close()
        assert engine.mixer.active_voices == 0

    def test_closing_twice_is_harmless(self, engine):
        engine.close()
        engine.close()

    def test_the_listener_starts_at_the_origin_looking_down_minus_z(self, engine):
        assert np.allclose(engine.listener.position, (0, 0, 0))
        assert np.allclose(engine.listener.forward, (0, 0, -1))

    def test_it_reports_itself_for_a_start_up_log(self, engine):
        engine.play(beep(), loop=True)
        assert repr(engine) == '<AudioEngine 1/8 voices %d Hz silent>' % (RATE,)


class TestPlaying:
    def test_playing_a_clip_starts_a_voice(self, engine):
        assert engine.play(beep()) is not None
        assert engine.mixer.active_voices == 1

    def test_playing_a_name_decodes_through_the_cache(self, engine):
        engine.clips.put('beep', beep())
        assert engine.play('beep') is not None

    def test_playing_a_name_that_will_not_decode_is_a_silence_not_an_error(self, engine):
        assert engine.play('nowhere/at/all.wav') is None
        assert engine.mixer.active_voices == 0

    def test_a_sound_with_no_emitter_is_heard_wherever_the_listener_is(self, engine):
        engine.play(beep(), gain=1.0)
        engine.listener = Listener(position=(1000.0, 0.0, 0.0))
        assert np.abs(engine.mixer.mix(64)).max() > 0.1

    def test_the_engine_reports_how_many_voices_are_sounding(self, engine):
        engine.play(beep(), loop=True)
        engine.play(beep(), loop=True)
        assert engine.active_voices == 2


class TestPositionalGain:
    """A positional emitter's gains follow its distance, cone and pan."""

    EMITTER = model.AudioEmitter(
        positional=model.PositionalProperties(distanceModel='inverse',
                                              refDistance=1.0))

    def test_a_sound_at_the_reference_distance_is_at_full_gain(self, engine):
        left, right = engine.gains_for(self.EMITTER, position=(0.0, 0.0, -1.0))
        assert math.hypot(left, right) == pytest.approx(1.0)

    def test_a_sound_further_away_is_quieter(self, engine):
        near = engine.gains_for(self.EMITTER, position=(0.0, 0.0, -1.0))
        far = engine.gains_for(self.EMITTER, position=(0.0, 0.0, -10.0))
        assert math.hypot(*far) < math.hypot(*near)

    def test_a_sound_on_the_right_is_louder_in_the_right_ear(self, engine):
        left, right = engine.gains_for(self.EMITTER, position=(5.0, 0.0, 0.0))
        assert right > left

    def test_a_sound_on_the_left_is_louder_in_the_left_ear(self, engine):
        left, right = engine.gains_for(self.EMITTER, position=(-5.0, 0.0, 0.0))
        assert left > right

    def test_panning_follows_the_listener_turning_rather_than_the_sound_moving(
            self, engine):
        engine.listener = Listener(forward=(1.0, 0.0, 0.0))
        left, right = engine.gains_for(self.EMITTER, position=(5.0, 0.0, 0.0))
        assert left == pytest.approx(right)

    def test_the_emitter_gain_multiplies_the_result(self, engine):
        loud = model.AudioEmitter(gain=1.0)
        quiet = model.AudioEmitter(gain=0.25)
        at = dict(position=(0.0, 0.0, -1.0))
        assert (math.hypot(*engine.gains_for(quiet, **at))
                == pytest.approx(0.25 * math.hypot(*engine.gains_for(loud, **at))))

    def test_a_cone_emitter_is_quiet_behind_itself(self, engine):
        emitter = model.AudioEmitter(positional=model.PositionalProperties(
            shapeType='cone', coneInnerAngle=math.pi / 2,
            coneOuterAngle=math.pi * 0.75, coneOuterGain=0.0))
        # The emitter's -Z axis is its forward, as glTF specifies.
        facing = engine.gains_for(emitter, position=(0.0, 0.0, 1.0),
                                  forward=(0.0, 0.0, -1.0))
        away = engine.gains_for(emitter, position=(0.0, 0.0, 1.0),
                                forward=(0.0, 0.0, 1.0))
        assert math.hypot(*facing) > 0.5
        assert math.hypot(*away) == pytest.approx(0.0)

    def test_a_global_emitter_ignores_distance_entirely(self, engine):
        emitter = model.AudioEmitter(type='global', gain=0.5)
        near = engine.gains_for(emitter, position=(0.0, 0.0, -1.0))
        far = engine.gains_for(emitter, position=(0.0, 0.0, -1000.0))
        assert near == pytest.approx(far)

    def test_a_linear_emitter_outside_the_maximum_distance_is_silent(self, engine):
        emitter = model.AudioEmitter(positional=model.PositionalProperties(
            distanceModel='linear', refDistance=1.0, maxDistance=10.0))
        assert engine.gains_for(emitter, position=(0.0, 0.0, -50.0)) == (0.0, 0.0)

    def test_an_inverse_emitter_past_its_maximum_distance_keeps_falling(self, engine):
        """The formula does not use ``maxDistance``; ``in_range`` is the cull."""
        properties = model.PositionalProperties(distanceModel='inverse',
                                                refDistance=1.0, maxDistance=10.0)
        emitter = model.AudioEmitter(positional=properties)
        far = math.hypot(*engine.gains_for(emitter, position=(0.0, 0.0, -50.0)))
        further = math.hypot(*engine.gains_for(emitter, position=(0.0, 0.0, -500.0)))
        assert 0.0 < further < far
        assert properties.in_range(50.0) is False


class TestConeGeometryDegeneracies:
    """Positions and axes that have no angle between them at all."""

    CONE = model.AudioEmitter(positional=model.PositionalProperties(
        shapeType='cone', coneInnerAngle=0.1, coneOuterAngle=0.2,
        coneOuterGain=0.0))

    def test_an_emitter_with_no_forward_axis_is_not_attenuated_by_direction(self, engine):
        """A zero axis has no direction to be off; that is not a reason to
        silence the emitter, and dividing by its length would be worse."""
        left, right = engine.gains_for(self.CONE, position=(0.0, 0.0, -1.0),
                                       forward=(0.0, 0.0, 0.0))
        assert math.hypot(left, right) == pytest.approx(1.0)

    def test_an_emitter_on_top_of_the_listener_is_not_attenuated_either(self, engine):
        left, right = engine.gains_for(self.CONE, position=(0.0, 0.0, 0.0),
                                       forward=(0.0, 0.0, -1.0))
        assert math.hypot(left, right) == pytest.approx(1.0)


class TestAiming:
    """A moving sound is re-aimed every frame, not restarted."""

    def test_aiming_updates_a_playing_voice(self, engine):
        emitter = model.AudioEmitter()
        handle = engine.play(beep(), emitter=emitter, position=(0.0, 0.0, -1.0))
        engine.mixer.mix(16)
        engine.aim(handle, emitter, position=(1000.0, 0.0, 0.0))
        assert engine.mixer.mix(64)[-1].max() == pytest.approx(0.0, abs=1e-3)

    def test_aiming_a_finished_sound_does_nothing(self, engine):
        handle = engine.play(synth.tone(440.0, 0.001, sample_rate=RATE))
        engine.mixer.mix(64)
        engine.aim(handle, model.AudioEmitter(), position=(0.0, 0.0, -1.0))
        assert not handle.playing

    def test_aiming_tolerates_a_sound_that_was_never_started(self, engine):
        """``play`` returns None when the pool refuses; ``aim`` must accept that."""
        engine.aim(None, model.AudioEmitter(), position=(0.0, 0.0, -1.0))

    def test_aiming_at_a_degenerate_position_does_not_poison_the_mix(self, engine):
        """A scenegraph produces NaN for ordinary reasons; one emitter doing so
        must not take every other sound in the scene with it."""
        emitter = model.AudioEmitter()
        broken = engine.play(beep(5.0), emitter=emitter, position=(0.0, 0.0, -1.0),
                             loop=True)
        engine.play(beep(5.0), gain=1.0, loop=True)
        engine.mixer.mix(16)
        engine.aim(broken, emitter, position=(float('nan'), 0.0, 0.0))
        block = engine.mixer.mix(64)
        assert np.isfinite(block).all()
        assert np.abs(block).max() > 0.1


class TestSourcePlayback:
    """``KHR_audio_emitter`` sources carry their own playback settings."""

    def library(self, engine, clip=None, **source):
        """A one-source, one-audio document, resolved to ``clip``."""
        document = model.AudioDocument(
            audio=[model.Audio(uri='ambience.wav')],
            sources=[model.AudioSource(audio=0, **source)])
        fetch = supplying(clip if clip is not None else beep(5.0))
        return engine.library(document, fetch=fetch)

    def level(self, engine, **source):
        """The peak of the mix for one source played on its own."""
        library = self.library(engine, **source)
        assert engine.play_source(library.document.sources[0], library) is not None
        return float(np.abs(engine.mixer.mix(256)).max())

    def test_the_sources_gain_is_applied(self, engine):
        """The document's gain is the author's decision, and honouring it is
        the whole point of reading the extension rather than inventing a format."""
        full = self.level(engine, gain=1.0)
        engine.stop_all()
        quarter = self.level(engine, gain=0.25)
        assert quarter == pytest.approx(0.25 * full, rel=0.02)

    def test_a_source_gain_of_zero_is_silence(self, engine):
        assert self.level(engine, gain=0.0) == pytest.approx(0.0, abs=1e-6)

    def test_the_sources_loop_flag_is_honoured(self, engine):
        library = self.library(engine, loop=True,
                               clip=synth.tone(440.0, 0.001, sample_rate=RATE))
        handle = engine.play_source(library.document.sources[0], library)
        engine.mixer.mix(256)
        assert handle.playing

    def test_a_source_that_does_not_loop_stops_at_the_end(self, engine):
        library = self.library(engine, clip=synth.tone(440.0, 0.001, sample_rate=RATE))
        handle = engine.play_source(library.document.sources[0], library)
        engine.mixer.mix(256)
        assert not handle.playing

    def test_the_playback_rate_speeds_the_clip_up(self, engine):
        library = self.library(engine, playbackRate=4.0,
                               clip=synth.tone(440.0, 0.02, sample_rate=RATE))
        handle = engine.play_source(library.document.sources[0], library)
        engine.mixer.mix(64)                        # 0.008s at 8 kHz
        assert not handle.playing

    def test_a_source_naming_no_audio_plays_nothing(self, engine):
        document = model.AudioDocument(sources=[model.AudioSource()])
        library = engine.library(document, fetch=supplying(beep()))
        assert engine.play_source(document.sources[0], library) is None

    def test_a_source_whose_audio_has_not_arrived_plays_nothing_yet(self, engine):
        """An asynchronous fetch is an ordinary silence, not an error."""
        document = model.AudioDocument(audio=[model.Audio(uri='late.wav')],
                                       sources=[model.AudioSource(audio=0)])
        library = engine.library(document, fetch=lambda *ignored: None)
        assert engine.play_source(document.sources[0], library) is None
        assert library.pending == (0,)

    def test_a_source_plays_once_its_audio_arrives(self, engine):
        document = model.AudioDocument(audio=[model.Audio(uri='late.wav')],
                                       sources=[model.AudioSource(audio=0)])
        library = engine.library(document, fetch=lambda *ignored: None)
        engine.play_source(document.sources[0], library)
        library.supply(0, beep(5.0))
        assert engine.play_source(document.sources[0], library) is not None

    def test_the_emitter_and_the_source_gains_multiply(self, engine):
        library = self.library(engine, gain=0.5)
        emitter = model.AudioEmitter(type='global', gain=0.5)
        engine.play_source(library.document.sources[0], library, emitter=emitter)
        both = float(np.abs(engine.mixer.mix(256)).max())
        engine.stop_all()
        engine.play_source(library.document.sources[0], library)
        source_only = float(np.abs(engine.mixer.mix(256)).max())
        assert both == pytest.approx(0.5 * source_only, rel=0.02)


class TestAutoplay:
    """What starts when the scene does -- and only when an application says so."""

    def document(self):
        return model.AudioDocument(
            audio=[model.Audio(uri='river.wav')],
            sources=[model.AudioSource(audio=0, autoplay=True, loop=True),
                     model.AudioSource(audio=0)],
            emitters=[model.AudioEmitter(type='global', name='river', sources=[0, 1])])

    def test_nothing_starts_by_itself(self, engine):
        """Loading a document must not make a noise; only the application knows
        when the scene it belongs to has actually begun."""
        engine.library(self.document(), fetch=supplying(beep(5.0)))
        assert engine.active_voices == 0

    def test_starting_autoplay_plays_the_marked_source_and_only_it(self, engine):
        library = engine.library(self.document(), fetch=supplying(beep(5.0)))
        handles = engine.start_autoplay(library)
        assert len(handles) == 1
        assert engine.active_voices == 1

    def test_the_handles_come_back_so_the_caller_can_steer_and_stop_them(self, engine):
        library = engine.library(self.document(), fetch=supplying(beep(5.0)))
        for handle in engine.start_autoplay(library):
            handle.stop()
        assert engine.active_voices == 0

    def test_an_emitter_the_consumer_places_is_heard_from_there(self, engine):
        document = self.document()
        document.emitters[0] = model.AudioEmitter(name='river', sources=[0])
        library = engine.library(document, fetch=supplying(beep(5.0)))
        placed = {}

        def place(emitter):
            placed[emitter.name] = True
            return (100.0, 0.0, 0.0), (0.0, 0.0, -1.0)

        engine.start_autoplay(library, place=place)
        assert placed == {'river': True}
        # A hundred metres away under the default inverse curve is very quiet.
        assert float(np.abs(engine.mixer.mix(64)).max()) < 0.05

    def test_an_emitter_the_consumer_does_not_place_is_heard_as_global(self, engine):
        document = self.document()
        document.emitters[0] = model.AudioEmitter(name='river', sources=[0])
        library = engine.library(document, fetch=supplying(beep(5.0)))
        engine.start_autoplay(library, place=lambda emitter: None)
        assert float(np.abs(engine.mixer.mix(64)).max()) > 0.1

    def test_audio_that_will_not_resolve_is_simply_absent(self, engine):
        library = engine.library(self.document(),
                                 fetch=lambda lib, i, a: lib.fail(i, 'no such file'))
        assert engine.start_autoplay(library) == []

    def test_a_document_asking_for_nothing_starts_nothing(self, engine):
        library = engine.library(model.AudioDocument(), fetch=supplying(beep()))
        assert engine.start_autoplay(library) == []


class TestListenerFromPlatform:
    def test_the_listener_follows_the_view_platform(self, engine, pose):
        engine.listen(pose(position=(1.0, 2.0, 3.0)))
        assert np.allclose(engine.listener.position, (1.0, 2.0, 3.0))

    def test_listening_returns_the_listener_it_installed(self, engine, pose):
        assert engine.listen(pose(position=(0.0, 1.0, 0.0))) is engine.listener


class TestMasterControls:
    """Two levels, and they multiply.

    Conflating them is the one mistake here that produces a volume control
    which prints a new number and changes nothing, so the product is asserted
    rather than either half of it.
    """

    def test_the_master_gain_reaches_the_mixer(self, engine):
        engine.master_gain = 0.25
        assert engine.mixer.master_gain == pytest.approx(0.25)

    def test_the_volume_reaches_the_mixer(self, engine):
        engine.volume = 0.25
        assert engine.mixer.master_gain == pytest.approx(0.25)

    def test_they_multiply_rather_than_overwrite_each_other(self, engine):
        engine.master_gain = 0.5
        engine.volume = 0.5
        assert engine.mixer.master_gain == pytest.approx(0.25)

    def test_the_order_they_are_written_in_does_not_matter(self, engine):
        engine.volume = 0.5
        engine.master_gain = 0.5
        assert engine.mixer.master_gain == pytest.approx(0.25)

    def test_a_gain_passed_at_construction_reaches_the_mixer(self, engine):
        """Two writers for one number would leave whichever lost last."""
        made = AudioEngine(device=NullDevice(sample_rate=RATE), master_gain=0.5)
        try:
            assert made.mixer.master_gain == pytest.approx(0.5)
            made.volume = 0.5
            assert made.mixer.master_gain == pytest.approx(0.25)
        finally:
            made.close()

    def test_the_volume_is_clamped_to_what_a_settings_screen_can_ask_for(self, engine):
        engine.volume = 7.0
        assert engine.volume == 1.0
        engine.volume = -3.0
        assert engine.volume == 0.0

    def test_the_master_gain_may_go_above_one_but_not_below_zero(self, engine):
        engine.master_gain = 2.0
        assert engine.master_gain == 2.0
        engine.master_gain = -1.0
        assert engine.master_gain == 0.0

    def test_the_volume_is_audible_in_the_mix(self, engine):
        engine.play(beep(5.0), gain=1.0, loop=True)
        loud = float(np.abs(engine.mixer.mix(64)).max())
        engine.volume = 0.5
        assert float(np.abs(engine.mixer.mix(64)).max()) == pytest.approx(
            0.5 * loud, rel=0.05)

    def test_the_muffle_reaches_the_mixer(self, engine):
        engine.muffle = 0.75
        assert engine.mixer.muffle == pytest.approx(0.75)
        assert engine.muffle == pytest.approx(0.75)

    def test_stopping_everything_silences_the_engine(self, engine):
        engine.play(beep(), loop=True)
        engine.stop_all()
        assert engine.active_voices == 0


class TestClipAccess:
    def test_a_clip_object_is_passed_straight_through(self, engine):
        clip = beep()
        assert engine.clip(clip) is clip

    def test_a_name_is_resolved_through_the_cache(self, engine):
        clip = beep()
        engine.clips.put('beep', clip)
        assert engine.clip('beep') is clip

    def test_a_library_built_by_the_engine_shares_its_cache(self, engine):
        """So a file two documents both name is decoded once, at the right rate."""
        library = engine.library(model.AudioDocument())
        assert isinstance(library, AudioLibrary)
        assert library.cache is engine.clips
