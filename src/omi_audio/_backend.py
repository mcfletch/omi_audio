"""The optional ``miniaudio`` dependency, imported once, in one place.

Two modules need the same fact and would otherwise each keep their own copy of
it: :mod:`~omi_audio.clip` asks whether a file can be decoded, and
:mod:`~omi_audio.device` asks whether a sound card can be reached.  Both are the
question "is ``miniaudio`` installed?", and two answers to one question drift.

The import is deferred rather than done at module load, so the answer is not
frozen before a test -- or an application that vendors its own backend -- can
arrange for it to be different.  It is attempted at most once: a machine without
the package should pay for one failed import, not one per sound.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: Frames per second the whole chain runs at where nothing else is specified.
#: 44.1 kHz is what most content is authored at, so resampling on load is
#: usually a no-op.  It lives here because :mod:`~omi_audio.clip` and
#: :mod:`~omi_audio.device` must agree on it and neither owns the other.
DEFAULT_SAMPLE_RATE = 44100

_miniaudio: Any = None
_attempted = False


def backend() -> Any:
    """The ``miniaudio`` module, or None where it is not installed."""
    global _miniaudio, _attempted
    if not _attempted:
        _attempted = True
        try:
            import miniaudio
        except ImportError as error:
            log.info('miniaudio unavailable: %s', error)
        else:
            _miniaudio = miniaudio
    return _miniaudio


def available() -> bool:
    """Whether the optional backend is installed at all."""
    return backend() is not None
