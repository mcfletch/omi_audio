"""The glTF audio codec extensions: what a document may offer, and what plays.

``KHR_audio_emitter`` guarantees only MP3.  ``OMI_audio_ogg_vorbis`` and
``OMI_audio_opus`` each name a second entry in the same ``audio`` array holding
the same sound in a better encoding, leaving the source's own ``audio`` as the
fallback -- so the interesting behaviour is *choosing*, and choosing correctly
on a build that cannot decode everything a document offers.
"""

import pytest

from omi_audio import formats, model


class _Formats:
    """A stand-in for `miniaudio.FileFormat`, carrying the members named."""

    def __init__(self, *names):
        self.members = [type('Member', (), {'name': name})() for name in names]

    def __iter__(self):
        return iter(self.members)


class _Backend:
    """A stand-in for the `miniaudio` module, advertising given formats."""

    def __init__(self, *names):
        self.FileFormat = _Formats(*names)


class TestTheRegistry:
    """What the two extensions are, as the specifications state them."""

    def test_the_extensions_are_named_as_the_specifications_name_them(self):
        assert formats.VORBIS.extension == 'OMI_audio_ogg_vorbis'
        assert formats.OPUS.extension == 'OMI_audio_opus'

    def test_opus_is_offered_before_vorbis(self):
        assert list(formats.ENCODINGS) == [formats.OPUS, formats.VORBIS]

    def test_vorbis_carries_its_mime_type_and_suffix(self):
        assert formats.VORBIS.mime_types == ('audio/ogg',)
        assert formats.VORBIS.suffixes == ('.ogg',)

    def test_opus_carries_both_of_its_containers(self):
        assert 'audio/opus' in formats.OPUS.mime_types
        assert 'audio/webm' in formats.OPUS.mime_types
        assert formats.OPUS.suffixes == ('.opus', '.webm')

    def test_an_extension_name_finds_its_encoding(self):
        assert formats.by_extension('OMI_audio_ogg_vorbis') is formats.VORBIS
        assert formats.by_extension('KHR_materials_unlit') is None


class TestWhatThisBuildCanDecode:
    """`decodable` asks the backend rather than asserting what it supports."""

    def test_a_backend_that_reads_vorbis_makes_vorbis_playable(self, monkeypatch):
        monkeypatch.setattr(formats._backend, 'backend',
                            lambda: _Backend('UNKNOWN', 'WAV', 'MP3', 'VORBIS'))
        assert formats.decodable() == (formats.VORBIS,)

    def test_a_backend_that_gains_opus_makes_opus_playable(self, monkeypatch):
        monkeypatch.setattr(formats._backend, 'backend',
                            lambda: _Backend('MP3', 'VORBIS', 'OPUS'))
        assert formats.decodable() == (formats.OPUS, formats.VORBIS)

    def test_no_backend_decodes_nothing(self, monkeypatch):
        monkeypatch.setattr(formats._backend, 'backend', lambda: None)
        assert formats.decodable() == ()

    def test_the_installed_backend_reads_ogg_vorbis(self):
        """The claim the documentation makes, checked against what is installed."""
        if formats._backend.backend() is None:
            pytest.skip('miniaudio is not installed; nothing decodes anything')
        assert formats.VORBIS in formats.decodable()


class TestReadingASource:
    """Pulling the codec extensions off one entry of the ``sources`` array."""

    def test_a_source_with_no_extensions_offers_no_alternatives(self):
        assert formats.read({'audio': 0}) == {}

    def test_a_vorbis_alternative_is_read(self):
        block = {'audio': 0, 'extensions':
                 {'OMI_audio_ogg_vorbis': {'audio': 1}}}
        assert formats.read(block) == {'OMI_audio_ogg_vorbis': 1}

    def test_both_alternatives_are_read(self):
        block = {'audio': 0, 'extensions': {
            'OMI_audio_ogg_vorbis': {'audio': 1},
            'OMI_audio_opus': {'audio': 2}}}
        assert formats.read(block) == {'OMI_audio_ogg_vorbis': 1,
                                       'OMI_audio_opus': 2}

    def test_an_unrelated_extension_is_left_alone(self):
        block = {'audio': 0, 'extensions': {'SOME_other_thing': {'audio': 9}}}
        assert formats.read(block) == {}

    @pytest.mark.parametrize('extensions', [
        None, [], 'OMI_audio_ogg_vorbis', 3,
    ])
    def test_an_extensions_object_that_is_not_an_object_is_ignored(self, extensions):
        assert formats.read({'audio': 0, 'extensions': extensions}) == {}

    @pytest.mark.parametrize('entry', [
        {}, {'audio': 'one'}, {'audio': True}, {'audio': None}, [1], 'x', 7,
    ])
    def test_an_alternative_without_a_usable_index_is_ignored(self, entry):
        block = {'audio': 0, 'extensions': {'OMI_audio_ogg_vorbis': entry}}
        assert formats.read(block) == {}

    def test_a_block_that_is_not_an_object_is_ignored(self):
        assert formats.read(['audio']) == {}


class TestWritingASource:
    """The inverse, for an exporter."""

    def test_nothing_offered_writes_nothing(self):
        assert formats.write({}) == {}

    def test_an_alternative_is_written_the_way_it_was_read(self):
        assert formats.write({'OMI_audio_ogg_vorbis': 1}) == {
            'OMI_audio_ogg_vorbis': {'audio': 1}}

    def test_the_written_block_shares_nothing_with_its_input(self):
        given = {'OMI_audio_ogg_vorbis': 1}
        written = formats.write(given)
        written['OMI_audio_ogg_vorbis']['audio'] = 99
        assert given == {'OMI_audio_ogg_vorbis': 1}


class TestChoosingAnEncoding:
    """A source's audio indices, most preferred first."""

    def source(self, **encodings):
        return model.AudioSource(audio=0, encodings=dict(encodings))

    def test_a_source_with_no_alternatives_offers_only_its_own_audio(self):
        assert self.source().audio_indices(formats.ENCODINGS) == [0]

    def test_a_caller_that_decodes_nothing_gets_the_mp3_fallback(self):
        source = self.source(OMI_audio_ogg_vorbis=1)
        assert source.audio_indices() == [0]

    def test_a_decodable_alternative_comes_before_the_fallback(self):
        source = self.source(OMI_audio_ogg_vorbis=1)
        assert source.audio_indices([formats.VORBIS]) == [1, 0]

    def test_an_alternative_the_caller_cannot_decode_is_skipped(self):
        source = self.source(OMI_audio_opus=2, OMI_audio_ogg_vorbis=1)
        assert source.audio_indices([formats.VORBIS]) == [1, 0]

    def test_the_callers_own_order_decides_which_alternative_wins(self):
        source = self.source(OMI_audio_opus=2, OMI_audio_ogg_vorbis=1)
        assert source.audio_indices([formats.OPUS, formats.VORBIS]) == [2, 1, 0]
        assert source.audio_indices([formats.VORBIS, formats.OPUS]) == [1, 2, 0]

    def test_a_source_with_no_audio_at_all_offers_nothing(self):
        source = model.AudioSource(encodings={'OMI_audio_ogg_vorbis': 1})
        assert source.audio_indices() == []
        assert source.audio_indices([formats.VORBIS]) == [1]

    def test_an_alternative_pointing_at_the_fallback_is_named_once(self):
        source = self.source(OMI_audio_ogg_vorbis=0)
        assert source.audio_indices([formats.VORBIS]) == [0]
