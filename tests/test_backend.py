"""The optional dependency, and the fact that both halves ask about it once.

Two modules used to keep their own copy of "is ``miniaudio`` installed?", which
is one question with two answers waiting to disagree.  These pin that there is
now one.
"""

import sys

import pytest

from omi_audio import _backend
from omi_audio import clip as clipmodule
from omi_audio import device as devicemodule


@pytest.fixture
def forgotten_import(monkeypatch):
    """Put the module back to the state it is in before anything asks."""
    monkeypatch.setattr(_backend, '_miniaudio', None)
    monkeypatch.setattr(_backend, '_attempted', False)


class TestTheImportHappensOnce:
    def test_an_absent_package_is_reported_rather_than_raised(
            self, forgotten_import, monkeypatch, caplog):
        """``sys.modules[name] = None`` is how the import system spells
        "this is not here", so it exercises the real ``ImportError`` path."""
        monkeypatch.setitem(sys.modules, 'miniaudio', None)
        with caplog.at_level('INFO', logger='omi_audio._backend'):
            assert _backend.backend() is None
        assert _backend.available() is False
        assert any('miniaudio' in record.getMessage() for record in caplog.records)

    def test_a_failed_import_is_not_retried(self, forgotten_import, monkeypatch):
        """A machine without the package pays for one failed import, not one
        per sound."""
        monkeypatch.setitem(sys.modules, 'miniaudio', None)
        assert [_backend.backend() for _ in range(3)] == [None, None, None]
        assert _backend._attempted is True

    def test_the_answer_is_cached_once_it_is_known(self, forgotten_import):
        assert _backend.backend() is _backend.backend()


class TestOneAnswerForBothHalves:
    def test_decoding_and_playback_report_the_same_availability(self):
        assert clipmodule.decoder_available() == devicemodule.miniaudio_available()

    def test_making_the_backend_absent_moves_both_at_once(self, no_backend):
        assert clipmodule.decoder_available() is False
        assert devicemodule.miniaudio_available() is False


def test_the_default_sample_rate_is_the_one_both_modules_use():
    """It lived in two files, and two copies of a constant drift."""
    assert clipmodule.DEFAULT_SAMPLE_RATE is _backend.DEFAULT_SAMPLE_RATE
    assert devicemodule.DEFAULT_SAMPLE_RATE is _backend.DEFAULT_SAMPLE_RATE
