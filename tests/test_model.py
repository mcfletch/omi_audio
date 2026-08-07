"""The ``KHR_audio_emitter`` data model, and reading it out of a glTF document."""

import math

import pytest

from omi_audio import formats, model, spatial


TAU = 2.0 * math.pi


class TestDefaults:
    """Every default is the one the extension writes down."""

    def test_audio_source_defaults(self):
        source = model.AudioSource()
        assert source.gain == 1.0
        assert source.playbackRate == 1.0
        assert source.loop is False
        assert source.autoplay is False
        assert source.audio is None

    def test_emitter_defaults(self):
        emitter = model.AudioEmitter()
        assert emitter.type == 'positional'
        assert emitter.gain == 1.0
        assert emitter.sources == []

    def test_positional_defaults(self):
        positional = model.PositionalProperties()
        assert positional.shapeType == 'omnidirectional'
        assert positional.coneInnerAngle == pytest.approx(TAU)
        assert positional.coneOuterAngle == pytest.approx(TAU)
        assert positional.coneOuterGain == 0.0
        assert positional.distanceModel == 'inverse'
        assert positional.maxDistance == 0.0
        assert positional.refDistance == 1.0
        assert positional.rolloffFactor == 1.0

    def test_a_positional_emitter_gets_default_positional_properties(self):
        """Nothing downstream should have to test for a missing sub-object."""
        assert model.AudioEmitter().positional == model.PositionalProperties()


class TestPositionalGain:
    """Distance and cone combined, which is what an emitter is asked for."""

    def test_omnidirectional_emitter_ignores_the_angle(self):
        positional = model.PositionalProperties(shapeType='omnidirectional',
                                                coneInnerAngle=0.1,
                                                coneOuterAngle=0.2,
                                                coneOuterGain=0.0)
        assert positional.gain(1.0, math.pi) == pytest.approx(1.0)

    def test_cone_emitter_applies_the_cone(self):
        positional = model.PositionalProperties(shapeType='cone',
                                                coneInnerAngle=math.pi / 2,
                                                coneOuterAngle=math.pi,
                                                coneOuterGain=0.0)
        assert positional.gain(1.0, math.pi) == pytest.approx(0.0)
        assert positional.gain(1.0, 0.0) == pytest.approx(1.0)

    def test_distance_and_cone_multiply(self):
        positional = model.PositionalProperties(shapeType='cone',
                                                coneInnerAngle=0.0,
                                                coneOuterAngle=math.pi,
                                                coneOuterGain=0.0)
        distance_only = spatial.distance_gain(2.0, 'inverse', ref_distance=1.0)
        cone_only = spatial.cone_gain(math.pi / 4, 0.0, math.pi, 0.0)
        assert positional.gain(2.0, math.pi / 4) == pytest.approx(
            distance_only * cone_only)


class TestGlobalEmitters:
    """A global emitter is heard the same wherever the listener stands."""

    def test_global_emitter_has_no_positional_properties(self):
        assert model.AudioEmitter(type='global').positional is None

    def test_global_emitter_is_not_positional(self):
        assert model.AudioEmitter(type='global').positional_audio is False
        assert model.AudioEmitter(type='positional').positional_audio is True


class TestFromGltf:
    """Reading the extension block a glTF document carries."""

    DOCUMENT = {
        'emitters': [
            {'name': 'Positional Emitter', 'type': 'positional', 'gain': 0.8,
             'sources': [0, 1],
             'positional': {'shapeType': 'cone', 'distanceModel': 'linear',
                            'maxDistance': 10.0, 'refDistance': 1.0,
                            'rolloffFactor': 0.8, 'coneInnerAngle': 1.0,
                            'coneOuterAngle': 2.0, 'coneOuterGain': 0.1}},
            {'name': 'Global Emitter', 'type': 'global', 'gain': 0.5,
             'sources': [1]},
        ],
        'sources': [
            {'name': 'Clip 1', 'gain': 0.6, 'autoplay': True, 'loop': True,
             'audio': 0},
            {'name': 'Clip 2', 'audio': 1, 'playbackRate': 2.0},
        ],
        'audio': [
            {'uri': 'audio1.mp3'},
            {'bufferView': 0, 'mimeType': 'audio/mpeg'},
        ],
    }

    def test_reads_every_array(self):
        document = model.from_gltf(self.DOCUMENT)
        assert len(document.emitters) == 2
        assert len(document.sources) == 2
        assert len(document.audio) == 2

    def test_reads_emitter_fields(self):
        emitter = model.from_gltf(self.DOCUMENT).emitters[0]
        assert emitter.name == 'Positional Emitter'
        assert emitter.gain == pytest.approx(0.8)
        assert emitter.sources == [0, 1]
        assert emitter.positional.distanceModel == 'linear'
        assert emitter.positional.maxDistance == pytest.approx(10.0)
        assert emitter.positional.coneOuterGain == pytest.approx(0.1)

    def test_a_global_emitter_reads_back_without_positional_properties(self):
        emitter = model.from_gltf(self.DOCUMENT).emitters[1]
        assert emitter.type == 'global'
        assert emitter.positional is None

    def test_reads_source_fields_and_supplies_missing_defaults(self):
        first, second = model.from_gltf(self.DOCUMENT).sources
        assert (first.gain, first.autoplay, first.loop, first.audio) == (0.6, True, True, 0)
        assert (second.gain, second.autoplay, second.loop) == (1.0, False, False)
        assert second.playbackRate == pytest.approx(2.0)

    def test_reads_audio_data_by_uri_and_by_buffer_view(self):
        by_uri, by_view = model.from_gltf(self.DOCUMENT).audio
        assert by_uri.uri == 'audio1.mp3'
        assert by_uri.bufferView is None
        assert by_view.bufferView == 0
        assert by_view.mimeType == 'audio/mpeg'

    def test_an_empty_block_reads_as_an_empty_document(self):
        document = model.from_gltf({})
        assert (document.emitters, document.sources, document.audio) == ([], [], [])

    def test_unknown_keys_are_ignored_rather_than_raising(self):
        """A document using an extension we do not implement must still load."""
        document = model.from_gltf({'sources': [{'gain': 0.5, 'somethingNew': 3}]})
        assert document.sources[0].gain == pytest.approx(0.5)


class TestRoundTrip:
    """The model writes back what it read, so a scene can be re-exported."""

    def test_gltf_round_trip_is_the_identity(self):
        document = model.from_gltf(TestFromGltf.DOCUMENT)
        assert model.from_gltf(model.to_gltf(document)) == document

    def test_defaults_are_omitted_from_the_written_block(self):
        """A written document should be no larger than it needs to be."""
        document = model.AudioDocument(sources=[model.AudioSource(audio=0)])
        assert model.to_gltf(document)['sources'] == [{'audio': 0}]

    def test_a_document_with_no_audio_writes_no_audio_array(self):
        """Only non-empty arrays are written, so nothing gains an empty one."""
        block = model.to_gltf(model.AudioDocument(sources=[model.AudioSource()]))
        assert 'audio' not in block and 'emitters' not in block

    def test_an_empty_document_writes_an_empty_block(self):
        assert model.to_gltf(model.AudioDocument()) == {}

    def test_the_written_block_shares_no_state_with_the_model(self):
        """Renumbering indices in an exported block is the ordinary reason to
        touch one; it must not rewrite the document it came from."""
        document = model.AudioDocument(
            sources=[model.AudioSource(), model.AudioSource()],
            emitters=[model.AudioEmitter(sources=[0, 1])])
        block = model.to_gltf(document)
        block['emitters'][0]['sources'].append(99)
        assert document.emitters[0].sources == [0, 1]


class TestMalformedDocuments:
    """Third-party content is not always well formed, and must never raise.

    Two failures matter and they are different.  A crash on load loses the whole
    scene; a value of the wrong type that *loads* detonates later, in the mixer,
    on another thread, far from the document that caused it.  Neither happens.
    """

    NONSENSE = [
        ('the whole block', 'not a block'),
        ('an emitters object', {'emitters': {'a': 1}}),
        ('an emitter that is a string', {'emitters': ['nonsense']}),
        ('an audio entry of None', {'audio': [None]}),
        ('a sources string', {'sources': 'abc'}),
        ('a positional block that is a list', {'emitters': [{'positional': [1, 2]}]}),
    ]

    @pytest.mark.parametrize('label,block', NONSENSE, ids=[n[0] for n in NONSENSE])
    def test_it_loads_rather_than_raising(self, label, block):
        assert isinstance(model.from_gltf(block), model.AudioDocument)

    def test_a_gain_that_is_not_a_number_falls_back_to_the_default(self):
        """Left alone, a string gain reaches ``numpy`` and fails there."""
        document = model.from_gltf({'sources': [{'gain': 'loud', 'loop': True}]})
        assert document.sources[0].gain == 1.0
        assert document.sources[0].loop is True     # the rest of the record survives

    def test_a_source_list_that_is_not_a_list_falls_back_to_the_default(self):
        document = model.from_gltf({'emitters': [{'sources': 'abc'}]})
        assert document.emitters[0].sources == []

    def test_a_source_list_holding_a_non_index_falls_back_to_the_default(self):
        document = model.from_gltf({'emitters': [{'sources': [0, 'two']}]})
        assert document.emitters[0].sources == []

    def test_a_boolean_is_not_accepted_as_a_number(self):
        """``true`` for a gain is a document error, not a request for 1.0."""
        assert model.from_gltf({'sources': [{'gain': True}]}).sources[0].gain == 1.0

    def test_a_number_is_not_accepted_as_a_name(self):
        assert model.from_gltf({'audio': [{'name': 5}]}).audio[0].name == ''

    def test_a_number_is_not_accepted_as_a_flag(self):
        assert model.from_gltf({'sources': [{'loop': 1}]}).sources[0].loop is False

    def test_an_integer_gain_is_accepted_as_a_number(self):
        """JSON has one number type; ``1`` and ``1.0`` are the same value."""
        assert model.from_gltf({'sources': [{'gain': 0}]}).sources[0].gain == 0.0

    def test_what_was_dropped_is_logged(self, caplog):
        with caplog.at_level('WARNING'):
            model.from_gltf({'sources': [{'gain': 'loud'}]})
        assert any('gain' in record.getMessage() for record in caplog.records)


class TestGlobalEmittersDiscardPositionalProperties:
    """Correct per the extension, but the author deserves to hear about it."""

    def test_the_properties_are_dropped(self):
        emitter = model.AudioEmitter(type='global',
                                     positional=model.PositionalProperties())
        assert emitter.positional is None

    def test_dropping_them_is_reported(self, caplog):
        with caplog.at_level('INFO'):
            model.AudioEmitter(type='global', name='music',
                               positional=model.PositionalProperties())
        assert any('music' in record.getMessage() for record in caplog.records)

    def test_a_global_emitter_that_never_had_any_says_nothing(self, caplog):
        with caplog.at_level('INFO'):
            model.AudioEmitter(type='global')
        assert caplog.records == []


class TestMaximumDistanceAsARange:
    """``in_range`` is the extension's prose, kept apart from its formulas."""

    def test_no_maximum_means_everything_is_in_range(self):
        assert model.PositionalProperties().in_range(1e6) is True

    def test_inside_the_maximum_is_in_range(self):
        assert model.PositionalProperties(maxDistance=10.0).in_range(9.9) is True

    def test_at_the_maximum_is_still_in_range(self):
        assert model.PositionalProperties(maxDistance=10.0).in_range(10.0) is True

    def test_beyond_the_maximum_is_not(self):
        assert model.PositionalProperties(maxDistance=10.0).in_range(10.1) is False

    def test_it_is_the_reading_the_gain_curve_does_not_take(self):
        """The two disagree on purpose, and this pins that they do."""
        properties = model.PositionalProperties(distanceModel='inverse',
                                                refDistance=1.0, maxDistance=10.0)
        assert properties.gain(500.0) > 0.0
        assert properties.in_range(500.0) is False


class TestNodeAndSceneEmitters:
    """Where a positional emitter *is* -- the link the three arrays do not hold."""

    DOCUMENT = model.AudioDocument(emitters=[
        model.AudioEmitter(type='positional', name='Torch'),
        model.AudioEmitter(type='global', name='Music'),
    ])

    def node(self, *indices):
        return {'extensions': model.emitter_reference(list(indices))}

    def test_a_node_names_its_emitters(self):
        found = self.DOCUMENT.emitters_for_node(self.node(0, 1))
        assert [emitter.name for emitter in found] == ['Torch', 'Music']

    def test_a_node_with_no_extension_names_none(self):
        assert self.DOCUMENT.emitters_for_node({'mesh': 0}) == []

    def test_a_node_with_other_extensions_names_none(self):
        node = {'extensions': {'KHR_materials_unlit': {}}}
        assert self.DOCUMENT.emitters_for_node(node) == []

    def test_an_out_of_range_reference_is_skipped_rather_than_raising(self):
        assert len(self.DOCUMENT.emitters_for_node(self.node(0, 7))) == 1

    def test_a_scene_may_carry_a_global_emitter(self):
        found = self.DOCUMENT.emitters_for_scene(self.node(1))
        assert [emitter.name for emitter in found] == ['Music']

    def test_a_scene_may_not_carry_a_positional_one(self, caplog):
        """The extension forbids it; a malformed scene loses the sound, not itself."""
        with caplog.at_level('WARNING'):
            found = self.DOCUMENT.emitters_for_scene(self.node(0, 1))
        assert [emitter.name for emitter in found] == ['Music']
        assert any('Torch' in record.getMessage() for record in caplog.records)

    def test_a_reference_that_is_not_a_list_is_ignored(self):
        node = {'extensions': {model.EXTENSION: {'emitters': 'nonsense'}}}
        assert self.DOCUMENT.emitters_for_node(node) == []

    def test_a_malformed_extensions_object_is_ignored(self):
        assert self.DOCUMENT.emitters_for_node({'extensions': 'nonsense'}) == []
        assert model.emitter_indices('not a node') == []

    def test_a_malformed_extension_entry_is_ignored(self):
        node = {'extensions': {model.EXTENSION: 'nonsense'}}
        assert self.DOCUMENT.emitters_for_node(node) == []

    def test_the_reference_written_is_the_reference_read(self):
        assert model.emitter_indices(
            {'extensions': model.emitter_reference([2, 5])}) == [2, 5]

    def test_writing_a_reference_copies_the_indices(self):
        indices = [1, 2]
        written = model.emitter_reference(indices)
        indices.append(3)
        assert written[model.EXTENSION]['emitters'] == [1, 2]


class TestAutoplay:
    """Which sounds the document asks to start when the scene begins."""

    def document(self):
        return model.AudioDocument(
            sources=[model.AudioSource(name='ambience', autoplay=True),
                     model.AudioSource(name='footstep')],
            emitters=[model.AudioEmitter(name='river', sources=[0, 1]),
                      model.AudioEmitter(name='quiet', sources=[1])])

    def test_it_pairs_each_autoplay_source_with_its_emitter(self):
        pairs = self.document().autoplay()
        assert [(e.name, s.name) for e, s in pairs] == [('river', 'ambience')]

    def test_a_source_on_two_emitters_is_two_sounds(self):
        """One clip in two places is two sounds, not one played twice."""
        document = model.AudioDocument(
            sources=[model.AudioSource(name='drip', autoplay=True)],
            emitters=[model.AudioEmitter(name='left', sources=[0]),
                      model.AudioEmitter(name='right', sources=[0])])
        assert [e.name for e, _ in document.autoplay()] == ['left', 'right']

    def test_a_document_asking_for_nothing_returns_nothing(self):
        assert model.AudioDocument().autoplay() == []

    def test_autoplay_survives_the_round_trip(self):
        document = model.from_gltf(TestFromGltf.DOCUMENT)
        assert model.from_gltf(model.to_gltf(document)).sources[0].autoplay is True


class TestResolvingSources:
    """An emitter names sources by index; the document resolves them."""

    def test_sources_for_an_emitter(self):
        document = model.from_gltf(TestFromGltf.DOCUMENT)
        assert [s.name for s in document.sources_for(document.emitters[0])] == [
            'Clip 1', 'Clip 2']

    def test_audio_for_a_source(self):
        document = model.from_gltf(TestFromGltf.DOCUMENT)
        assert document.audio_for(document.sources[0]).uri == 'audio1.mp3'

    def test_a_source_with_no_audio_resolves_to_nothing(self):
        document = model.AudioDocument(sources=[model.AudioSource()])
        assert document.audio_for(document.sources[0]) is None

    def test_an_out_of_range_index_is_skipped_rather_than_raising(self):
        """Content is not always well formed, and a bad index is not fatal."""
        document = model.AudioDocument(
            sources=[model.AudioSource()],
            emitters=[model.AudioEmitter(sources=[0, 7])])
        assert len(document.sources_for(document.emitters[0])) == 1


class TestCodecExtensions:
    """``OMI_audio_ogg_vorbis`` and ``OMI_audio_opus`` on a source."""

    BLOCK = {
        'audio': [{'uri': 'shot.mp3'}, {'uri': 'shot.ogg'}, {'uri': 'shot.opus'}],
        'sources': [{'audio': 0, 'extensions': {
            'OMI_audio_ogg_vorbis': {'audio': 1},
            'OMI_audio_opus': {'audio': 2}}}],
    }

    def test_a_source_records_what_each_extension_offers(self):
        document = model.from_gltf(self.BLOCK)
        assert document.sources[0].encodings == {
            'OMI_audio_ogg_vorbis': 1, 'OMI_audio_opus': 2}

    def test_a_source_without_them_offers_nothing(self):
        document = model.from_gltf({'sources': [{'audio': 0}]})
        assert document.sources[0].encodings == {}

    def test_the_extensions_survive_a_round_trip(self):
        document = model.from_gltf(self.BLOCK)
        assert model.from_gltf(model.to_gltf(document)) == document

    def test_the_extensions_are_written_where_the_specification_puts_them(self):
        document = model.from_gltf(self.BLOCK)
        assert model.to_gltf(document)['sources'] == [self.BLOCK['sources'][0]]

    def test_a_source_offering_nothing_writes_no_extensions_object(self):
        document = model.AudioDocument(sources=[model.AudioSource(audio=0)])
        assert model.to_gltf(document)['sources'] == [{'audio': 0}]

    def test_the_written_extensions_share_no_state_with_the_model(self):
        document = model.from_gltf(self.BLOCK)
        block = model.to_gltf(document)
        block['sources'][0]['extensions']['OMI_audio_ogg_vorbis']['audio'] = 99
        assert document.sources[0].encodings['OMI_audio_ogg_vorbis'] == 1

    def test_a_documents_extension_names_are_offered_to_an_exporter(self):
        """An exporter has to declare these in ``extensionsUsed``."""
        document = model.from_gltf(self.BLOCK)
        assert document.extensions_used() == (
            'KHR_audio_emitter', 'OMI_audio_opus', 'OMI_audio_ogg_vorbis')

    def test_a_plain_document_declares_only_the_base_extension(self):
        document = model.from_gltf({'sources': [{'audio': 0}]})
        assert document.extensions_used() == ('KHR_audio_emitter',)


class TestResolvingAnEncoding:
    """Which entry of the ``audio`` array a source actually plays."""

    def document(self):
        return model.from_gltf(TestCodecExtensions.BLOCK)

    def test_with_no_codec_support_the_fallback_is_the_answer(self):
        document = self.document()
        assert document.audio_for(document.sources[0]).uri == 'shot.mp3'

    def test_a_supported_codec_wins(self):
        document = self.document()
        chosen = document.audio_for(document.sources[0], [formats.VORBIS])
        assert chosen.uri == 'shot.ogg'

    def test_every_encoding_is_offered_most_preferred_first(self):
        document = self.document()
        options = document.audio_options(document.sources[0], formats.ENCODINGS)
        assert [entry.uri for entry in options] == [
            'shot.opus', 'shot.ogg', 'shot.mp3']

    def test_an_extension_pointing_outside_the_audio_array_is_skipped(self):
        document = model.from_gltf({
            'audio': [{'uri': 'shot.mp3'}],
            'sources': [{'audio': 0, 'extensions': {
                'OMI_audio_ogg_vorbis': {'audio': 7}}}]})
        assert document.audio_indices_for(document.sources[0],
                                          [formats.VORBIS]) == [0]

    def test_a_source_naming_nothing_at_all_resolves_to_nothing(self):
        document = model.AudioDocument(audio=[model.Audio(uri='shot.mp3')],
                                       sources=[model.AudioSource()])
        assert document.audio_options(document.sources[0], formats.ENCODINGS) == []
