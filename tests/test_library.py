"""The seam where a document's audio *references* become audio.

The point of this module is a negative: `omi_audio` must never turn a glTF
``uri`` into a filename, a URL or anything else it then opens.  So the tests
below are as much about what does **not** happen as about what does.
"""

import numpy as np
import pytest

from omi_audio import formats, model, synth
from omi_audio.clip import Clip, ClipCache
from omi_audio.library import AudioLibrary

from support import RATE, needs_miniaudio, wav_bytes, write_wav

HOSTILE = [
    '../../../../etc/passwd',
    '/etc/shadow',
    'file:///etc/passwd',
    'http://evil.example/x.mp3',
    'data:audio/mpeg;base64,AAAA',
    '\\\\attacker\\share\\x.wav',
    'sounds%2Fshot.wav',
]


def document(*uris):
    """A document naming ``uris``, one source per entry."""
    return model.AudioDocument(
        audio=[model.Audio(uri=uri) for uri in uris],
        sources=[model.AudioSource(audio=index) for index in range(len(uris))])


def tone():
    return synth.tone(440.0, 0.5, sample_rate=RATE, fade=0.0)


class TestTheUriIsNeverResolved:
    """The security property, stated as tests rather than as a comment."""

    def watching(self, *uris):
        """A library whose fetch records the ask and resolves nothing."""
        seen = []
        library = AudioLibrary(document(*uris), cache=ClipCache(sample_rate=RATE),
                               fetch=lambda lib, index, audio: seen.append(audio.uri))
        return library, seen

    @pytest.mark.parametrize('uri', HOSTILE)
    def test_a_hostile_uri_reaches_the_application_and_nothing_else(self, uri):
        """It is handed back verbatim, for the application to accept or refuse."""
        library, seen = self.watching(uri)
        assert library.clip(0) is None
        assert seen == [uri]

    @pytest.mark.parametrize('uri', HOSTILE)
    def test_no_decoder_is_ever_handed_a_hostile_uri(self, uri):
        """The cache must not see it either: a decode attempt on a real path is
        a file-existence oracle even when the decode fails."""
        decoded = []
        cache = ClipCache(sample_rate=RATE,
                          decode=lambda name, rate: decoded.append(name))
        library = AudioLibrary(document(uri), cache=cache,
                               fetch=lambda lib, index, audio: None)
        library.clip(0)
        assert decoded == []

    def test_a_library_with_no_fetch_resolves_nothing_at_all(self, caplog):
        """The default is refusal, not a guess at what the uri might mean."""
        library = AudioLibrary(document('sounds/shot.wav'))
        with caplog.at_level('WARNING'):
            assert library.clip(0) is None
        assert len(caplog.records) == 1

    def test_the_application_may_supply_a_file_of_a_wholly_different_name(self):
        """Which is what a download cache does: the uri names a resource, the
        path names a blob, and the two have nothing in common."""
        library = AudioLibrary(
            document('http://example.invalid/river.mp3'),
            cache=ClipCache(sample_rate=RATE),
            fetch=lambda lib, i, a: lib.supply(i, tone()))
        assert library.clip(0) is not None


class TestAsking:
    def test_the_first_ask_fetches_and_later_asks_do_not(self):
        asks = []
        library = AudioLibrary(document('a.wav'), cache=ClipCache(sample_rate=RATE),
                               fetch=lambda lib, i, a: (asks.append(i),
                                                        lib.supply(i, tone())))
        library.clip(0)
        library.clip(0)
        assert asks == [0]

    def test_an_index_that_names_nothing_is_never_fetched(self):
        asks = []
        library = AudioLibrary(document('a.wav'), cache=ClipCache(sample_rate=RATE),
                               fetch=lambda lib, i, a: asks.append(i))
        assert library.clip(7) is None
        assert library.clip(-1) is None
        assert library.clip(None) is None
        assert asks == []

    def test_a_source_resolves_through_its_own_index(self):
        library = AudioLibrary(document('a.wav', 'b.wav'),
                               cache=ClipCache(sample_rate=RATE),
                               fetch=lambda lib, i, a: lib.supply(i, tone()))
        assert library.clip_for(library.document.sources[1]) is not None

    def test_a_source_naming_no_audio_resolves_to_nothing(self):
        library = AudioLibrary(model.AudioDocument(sources=[model.AudioSource()]))
        assert library.clip_for(library.document.sources[0]) is None

    def test_a_slow_fetch_leaves_the_index_pending(self):
        library = AudioLibrary(document('a.wav', 'b.wav'),
                               cache=ClipCache(sample_rate=RATE),
                               fetch=lambda lib, i, a: None)
        library.clip(0)
        assert library.pending == (0,)
        assert library.ready(0) is False

    def test_a_download_that_lands_later_is_playable_from_then_on(self):
        library = AudioLibrary(document('a.wav'), cache=ClipCache(sample_rate=RATE),
                               fetch=lambda lib, i, a: None)
        assert library.clip(0) is None
        library.supply(0, tone())
        assert library.pending == ()
        assert library.ready(0) is True
        assert library.clip(0) is not None

    def test_a_failed_index_is_never_asked_for_again(self):
        asks = []
        library = AudioLibrary(document('a.wav'), cache=ClipCache(sample_rate=RATE),
                               fetch=lambda lib, i, a: (asks.append(i),
                                                        lib.fail(i, 'gone')))
        library.clip(0)
        library.clip(0)
        assert asks == [0]

    def test_a_failure_is_reported_once_and_only_once(self, caplog):
        library = AudioLibrary(document('a.wav'), cache=ClipCache(sample_rate=RATE))
        with caplog.at_level('WARNING'):
            library.fail(0, 'gone')
            library.fail(0, 'still gone')
        assert len(caplog.records) == 1

    def test_the_reason_names_something_a_reader_can_find(self):
        library = AudioLibrary(model.AudioDocument(
            audio=[model.Audio(uri='sounds/river.ogg', name='River')]))
        assert 'River' in library._name(0)
        assert 'audio 0' in library._name(0)

    def test_an_unnamed_entry_still_produces_a_usable_label(self):
        assert AudioLibrary(document(''))._name(0) == 'audio 0'
        assert AudioLibrary(model.AudioDocument())._name(3) == 'audio 3'


class TestChoosingACodec:
    """``OMI_audio_ogg_vorbis`` / ``OMI_audio_opus``: a better encoding, offered.

    The library asks for the best encoding it can decode and keeps the source's
    own ``audio`` as the fallback the base extension guarantees.  Nothing here
    needs a real Ogg stream: what is under test is *which* entry is asked for,
    so the entries hold ordinary ``.wav`` bytes and differ in length.
    """

    def coded(self, **encodings):
        """One source offering its ``audio`` plus the alternatives named."""
        return model.AudioDocument(
            audio=[model.Audio(uri='shot.mp3'), model.Audio(uri='shot.ogg'),
                   model.Audio(uri='shot.opus')],
            sources=[model.AudioSource(audio=0, encodings=dict(encodings))])

    def library(self, document, encodings, fetch=None):
        made = AudioLibrary(document, cache=ClipCache(sample_rate=RATE),
                            fetch=fetch or (lambda lib, i, a: lib.supply(i, tone())))
        made.encodings = tuple(encodings)
        return made

    def test_a_build_that_decodes_vorbis_asks_for_the_vorbis_entry(self):
        asks = []
        library = self.library(self.coded(OMI_audio_ogg_vorbis=1), [formats.VORBIS],
                               fetch=lambda lib, i, a: (asks.append(i),
                                                        lib.supply(i, tone())))
        assert library.clip_for(library.document.sources[0]) is not None
        assert asks == [1]

    def test_a_build_that_cannot_decode_it_asks_for_the_fallback(self):
        asks = []
        library = self.library(self.coded(OMI_audio_ogg_vorbis=1), [],
                               fetch=lambda lib, i, a: (asks.append(i),
                                                        lib.supply(i, tone())))
        assert library.clip_for(library.document.sources[0]) is not None
        assert asks == [0]

    def test_an_alternative_that_will_not_resolve_falls_back_to_the_mp3(self):
        """The whole point of the fallback: a bad Ogg costs quality, not sound."""
        def fetch(library, index, audio):
            if index == 1:
                library.fail(index, 'the ogg is not there')
            else:
                library.supply(index, tone())

        library = self.library(self.coded(OMI_audio_ogg_vorbis=1),
                               [formats.VORBIS], fetch=fetch)
        assert library.clip_for(library.document.sources[0]) is not None

    def test_an_alternative_still_downloading_does_not_start_the_fallback(self):
        """Falling through on *pending* would play the worse encoding of every
        sound whose better one simply had not landed yet."""
        asks = []
        library = self.library(self.coded(OMI_audio_ogg_vorbis=1), [formats.VORBIS],
                               fetch=lambda lib, i, a: asks.append(i))
        assert library.clip_for(library.document.sources[0]) is None
        assert asks == [1]
        library.supply(1, tone())
        assert library.clip_for(library.document.sources[0]) is not None
        assert asks == [1]

    def test_an_index_outside_the_audio_array_is_skipped_for_the_fallback(self):
        library = self.library(self.coded(OMI_audio_ogg_vorbis=9), [formats.VORBIS])
        assert library.clip_for(library.document.sources[0]) is not None

    def test_a_library_decodes_what_this_build_decodes_unless_told_otherwise(self):
        library = AudioLibrary(self.coded(OMI_audio_ogg_vorbis=1))
        assert library.encodings == formats.decodable()

    def test_a_source_with_no_alternatives_is_unaffected(self):
        asks = []
        library = self.library(self.coded(), formats.ENCODINGS,
                               fetch=lambda lib, i, a: (asks.append(i),
                                                        lib.supply(i, tone())))
        assert library.clip_for(library.document.sources[0]) is not None
        assert asks == [0]

    @needs_miniaudio
    def test_the_chosen_entry_is_the_one_whose_samples_come_back(self):
        """End to end through the real decoder: two entries, two lengths."""
        short, long = tone(), synth.tone(440.0, 1.5, sample_rate=RATE, fade=0.0)
        encoded = {0: wav_bytes(short.samples, sample_rate=RATE),
                   1: wav_bytes(long.samples, sample_rate=RATE)}
        library = self.library(
            self.coded(OMI_audio_ogg_vorbis=1), [formats.VORBIS],
            fetch=lambda lib, i, a: lib.supply_bytes(i, encoded[i]))
        clip = library.clip_for(library.document.sources[0])
        assert clip.frames == long.frames


class TestSupplying:
    def test_a_clip_at_another_rate_is_resampled_to_the_engines(self):
        """The mixer runs at one rate, so nothing at another rate gets in."""
        library = AudioLibrary(document('a.wav'), cache=ClipCache(sample_rate=16000))
        assert library.supply(0, tone()).sample_rate == 16000

    def test_supplying_clears_an_earlier_failure(self):
        """A retry is the application's decision, and it must be able to make one."""
        library = AudioLibrary(document('a.wav'), cache=ClipCache(sample_rate=RATE))
        library.fail(0, 'the network was down')
        library.supply(0, tone())
        assert library.clip(0) is not None

    def test_a_file_is_decoded_through_the_shared_cache(self, tmp_path):
        """So one file behind two documents costs one decode."""
        decoded = []

        def counting(name, rate):
            decoded.append(name)
            return tone()

        cache = ClipCache(sample_rate=RATE, decode=counting)
        for _ in range(2):
            library = AudioLibrary(document('a.wav'), cache=cache)
            library.supply_file(0, str(tmp_path / 'river.wav'))
        assert len(decoded) == 1

    def test_a_file_that_will_not_decode_fails_the_index(self, tmp_path, caplog):
        library = AudioLibrary(document('a.wav'), cache=ClipCache(sample_rate=RATE))
        with caplog.at_level('WARNING'):
            assert library.supply_file(0, str(tmp_path / 'absent.wav')) is None
        assert library.clip(0) is None

    @needs_miniaudio
    def test_a_real_file_supplied_by_path_plays(self, tmp_path):
        path = write_wav(tmp_path / 'river.wav', tone().samples, sample_rate=RATE)
        library = AudioLibrary(document('river.wav'), cache=ClipCache(sample_rate=RATE))
        clip = library.supply_file(0, str(path))
        assert clip is not None and clip.frames > 0


@needs_miniaudio
class TestBufferViewAudio:
    """The ``.glb`` path: audio inside the file, with nothing to open.

    This is the dominant shipping format for glTF, so "we model the field and
    cannot play it" was never an acceptable place to stop.
    """

    def glb_like(self):
        """A document whose audio lives in a buffer view, as a ``.glb``'s does."""
        return model.AudioDocument(
            audio=[model.Audio(bufferView=0, mimeType='audio/mpeg')],
            sources=[model.AudioSource(audio=0)])

    def loader(self, data):
        """A stand-in for the consumer's glTF loader handing over the bytes."""
        def fetch(library, index, audio):
            assert audio.bufferView is not None
            library.supply_bytes(index, data)
        return fetch

    def test_audio_in_a_buffer_view_plays(self):
        data = wav_bytes(tone().samples, sample_rate=RATE)
        library = AudioLibrary(self.glb_like(), cache=ClipCache(sample_rate=RATE),
                               fetch=self.loader(data))
        clip = library.clip_for(library.document.sources[0])
        assert clip is not None
        assert clip.frames > 0

    def test_the_samples_are_the_ones_that_were_encoded(self):
        original = tone()
        library = AudioLibrary(self.glb_like(), cache=ClipCache(sample_rate=RATE),
                               fetch=self.loader(wav_bytes(original.samples,
                                                           sample_rate=RATE)))
        assert np.allclose(library.clip(0).samples, original.samples, atol=1e-4)

    def test_bytes_that_will_not_decode_fail_the_index_rather_than_raising(self, caplog):
        library = AudioLibrary(self.glb_like(), cache=ClipCache(sample_rate=RATE),
                               fetch=self.loader(b'not audio at all'))
        with caplog.at_level('WARNING'):
            assert library.clip(0) is None
        assert len(caplog.records) == 1

    def test_a_data_uri_is_the_same_path_once_the_application_has_decoded_it(self):
        """glTF permits ``data:audio/...;base64,``; unpacking it is the loader's
        job, and what comes out is bytes, which is a case already covered."""
        import base64

        encoded = base64.b64encode(wav_bytes(tone().samples, sample_rate=RATE))
        uri = 'data:audio/wav;base64,' + encoded.decode('ascii')

        def fetch(library, index, audio):
            head, _, payload = audio.uri.partition(',')
            assert head.endswith('base64')
            library.supply_bytes(index, base64.b64decode(payload))

        library = AudioLibrary(document(uri), cache=ClipCache(sample_rate=RATE),
                               fetch=fetch)
        assert library.clip(0) is not None


class TestReporting:
    def test_it_says_how_much_of_the_document_has_resolved(self):
        library = AudioLibrary(document('a.wav', 'b.wav', 'c.wav'),
                               cache=ClipCache(sample_rate=RATE),
                               fetch=lambda lib, i, a: None)
        library.supply(0, tone())
        library.clip(1)
        library.fail(2, 'gone')
        assert repr(library) == '<AudioLibrary 1/3 ready, 1 pending, 1 failed>'

    def test_its_length_is_how_many_clips_it_holds(self):
        library = AudioLibrary(document('a.wav', 'b.wav'),
                               cache=ClipCache(sample_rate=RATE))
        assert len(library) == 0
        library.supply(0, Clip(np.zeros(10, dtype='f'), RATE))
        assert len(library) == 1
