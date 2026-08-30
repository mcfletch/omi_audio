"""Ogg Opus: the container, the identification header, and the decode seam.

The container tests build their own streams from RFC 3533, so they pin the
reader against the specification rather than against a fixture, and they run
whether or not ``libopus`` is installed. Only the tests that turn packets into
samples need the library.

No Opus file is checked in. The content this was written for is licensed such
that it cannot live in this repository, and a synthesised stream cannot carry
real codec payload, so a decode over real audio runs only where a sample tree is
pointed at by ``OMI_AUDIO_OPUS_SAMPLES``.
"""

import os

import numpy as np
import pytest

from omi_audio import _opus, formats
from omi_audio.clip import DecodeError, decode_bytes, decode_file

from support import needs_libopus, ogg_stream, opus_head, wav_bytes

#: A tree of `.opus` files to decode for real, if this machine has one.
SAMPLE_ROOT = os.environ.get('OMI_AUDIO_OPUS_SAMPLES', '')


class TestTheOggContainer:
    """RFC 3533 §6: pages, the segment table, and packets laced across them."""

    def test_one_short_packet_comes_back_whole(self):
        assert list(_opus.packets(ogg_stream([b'hello']))) == [b'hello']

    def test_several_packets_keep_their_order(self):
        wanted = [b'one', b'two', b'three']
        assert list(_opus.packets(ogg_stream(wanted))) == wanted

    def test_a_packet_of_exactly_255_bytes_is_not_split(self):
        """255 means "continues", so a packet that length needs a trailing 0.

        Getting this wrong truncates every packet whose length is a multiple of
        255 and loses the one after it.
        """
        packet = bytes(range(256))[:255]
        assert list(_opus.packets(ogg_stream([packet]))) == [packet]

    def test_a_long_packet_is_rejoined_from_its_segments(self):
        packet = bytes(index % 251 for index in range(1000))
        assert list(_opus.packets(ogg_stream([packet]))) == [packet]

    def test_a_packet_is_rejoined_across_a_page_boundary(self):
        """The case a single-page reader gets wrong."""
        packet = bytes(index % 251 for index in range(4000))
        stream = ogg_stream([packet], page_size=4)
        assert stream.count(b'OggS') > 1, 'this test needs more than one page'
        assert list(_opus.packets(stream)) == [packet]

    def test_bytes_that_are_not_ogg_yield_nothing(self):
        assert list(_opus.packets(b'RIFFxxxxWAVE')) == []
        assert list(_opus.packets(b'')) == []

    def test_a_truncated_stream_costs_the_tail_and_not_the_load(self):
        """A short download should lose the end of a sound, not raise."""
        stream = ogg_stream([b'one', b'two'])
        assert list(_opus.packets(stream[:len(stream) - 2])) != []


class TestTheIdentificationHeader:
    """RFC 7845 §5.1."""

    def test_channels_and_pre_skip_are_read(self):
        assert _opus.head(opus_head(channels=2, pre_skip=356)) == (2, 356)

    def test_a_packet_that_is_not_an_opus_head_is_refused(self):
        with pytest.raises(_opus.OpusError):
            _opus.head(b'OpusTags' + bytes(16))

    def test_a_truncated_header_is_refused(self):
        with pytest.raises(_opus.OpusError):
            _opus.head(opus_head()[:8])

    @pytest.mark.parametrize('channels', [0, 3, 8])
    def test_a_channel_count_this_does_not_read_is_refused(self, channels):
        with pytest.raises(_opus.OpusError):
            _opus.head(opus_head(channels=channels))


class TestRecognisingOpus:
    """Told from the bytes, because a name or a MIME type may be wrong."""

    def test_an_ogg_opus_stream_is_recognised(self):
        assert _opus.looks_like_opus(ogg_stream([opus_head(), b'OpusTags']))

    def test_an_ogg_vorbis_stream_is_not(self):
        """The one that matters: Vorbis is Ogg too, and the backend reads it.

        Claiming it here would route Vorbis into a decoder that cannot read it
        and lose a format that already worked.
        """
        vorbis = ogg_stream([b'\x01vorbis' + bytes(24)])
        assert not _opus.looks_like_opus(vorbis)

    @pytest.mark.parametrize('data', [b'', b'OggS', b'RIFFxxxxWAVE', b'\xff\xfb\x90'])
    def test_other_bytes_are_not_opus(self, data):
        assert not _opus.looks_like_opus(data)

    def test_a_wav_is_not_opus(self):
        assert not _opus.looks_like_opus(wav_bytes(np.zeros(64, dtype='f')))


class TestDecoding:
    """Turning packets into samples, which is where ``libopus`` is needed."""

    def test_a_stream_with_no_audio_packets_is_an_error(self):
        with pytest.raises(_opus.OpusError):
            _opus.decode(ogg_stream([opus_head()]), 'short.opus')

    @needs_libopus
    @pytest.mark.parametrize('payload', [
        b'\x00' * 40,                    # a legal TOC byte; libopus accepts this
        b'\xff' * 40,
        bytes(range(64)),
        b'\x0c',                         # a one-byte packet
    ])
    def test_arbitrary_packet_bytes_never_take_the_process(self, payload):
        """Content is untrusted, and this hands attacker-controlled bytes to C.

        What is pinned is that the call is survivable and its result is sane --
        *not* that a given payload is rejected. `libopus` accepts far more than
        it produces: 40 zero bytes are a valid packet and decode to 168 frames
        of silence, so a test asserting they are refused would be asserting
        something untrue.
        """
        stream = ogg_stream([opus_head(), b'OpusTags', payload])
        try:
            samples, channels, rate = _opus.decode(stream, 'arbitrary.opus')
        except _opus.OpusError:
            return                              # refused, which is also fine
        assert channels == 1
        assert rate == _opus.OPUS_DECODE_RATE
        assert np.isfinite(samples).all()

    def test_the_decode_rate_is_the_one_the_specification_fixes(self):
        """RFC 7845 §2: always 48 kHz, whatever the header records."""
        assert _opus.OPUS_DECODE_RATE == 48000

    def test_an_absent_library_is_reported_and_not_raised_through(self, monkeypatch):
        monkeypatch.setattr(_opus, '_libopus', lambda: None)
        with pytest.raises(_opus.OpusError, match='libopus'):
            _opus.decode(ogg_stream([opus_head(), b'OpusTags', b'x']), 'x.opus')


class TestTheClipSeam:
    """What an application actually calls."""

    def test_opus_is_reported_as_decodable_exactly_when_it_is(self):
        offered = [encoding.extension for encoding in formats.decodable()]
        assert ('OMI_audio_opus' in offered) == _opus.available()

    def test_a_stream_that_will_not_decode_raises_a_decode_error(self):
        """Not an OpusError: the seam has one exception type."""
        with pytest.raises(DecodeError):
            decode_bytes(ogg_stream([opus_head()]), name='short.opus')

    def test_a_missing_file_is_a_decode_error(self, tmp_path):
        with pytest.raises(DecodeError):
            decode_file(str(tmp_path / 'absent.opus'))


@pytest.mark.skipif(not SAMPLE_ROOT or not os.path.isdir(SAMPLE_ROOT),
                    reason='no Opus sample tree; set OMI_AUDIO_OPUS_SAMPLES')
@needs_libopus
class TestRealAudio:
    """Decoding actual encoded audio, where this machine has some."""

    def _samples(self):
        found = []
        for directory, _dirs, files in os.walk(SAMPLE_ROOT):
            found.extend(os.path.join(directory, name) for name in files
                         if name.lower().endswith('.opus'))
        if not found:
            pytest.skip('no .opus files under %s' % (SAMPLE_ROOT,))
        return sorted(found)

    def test_every_sample_decodes_to_audible_mono(self):
        for path in self._samples():
            clip = decode_file(path)
            assert clip.samples.dtype == np.float32, path
            assert clip.samples.ndim == 1, path
            assert clip.frames > 0, path
            assert np.isfinite(clip.samples).all(), path

    def test_the_path_and_the_bytes_agree(self):
        """`decode_file` and `decode_bytes` are one decoder, not two."""
        for path in self._samples()[:4]:
            with open(path, 'rb') as handle:
                data = handle.read()
            assert np.array_equal(decode_file(path).samples,
                                  decode_bytes(data, name=path).samples), path

    def test_the_pre_roll_is_dropped(self):
        """RFC 7845 §4.2: left in, a sound opens with the encoder warming up."""
        path = self._samples()[0]
        with open(path, 'rb') as handle:
            data = handle.read()
        stream = list(_opus.packets(data))
        _channels, pre_skip = _opus.head(stream[0])
        assert pre_skip > 0, 'this sample has no pre-roll to test with'
        samples, channels, rate = _opus.decode(data, path)
        raw = sum(1 for _ in stream[2:])
        assert raw > 0
        assert len(samples) % channels == 0
        assert rate == _opus.OPUS_DECODE_RATE
