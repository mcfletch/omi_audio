"""Put real Opus recordings where the decode tests look for them.

``test_opus.py`` builds its own Ogg streams from RFC 3533, which pins the reader
against the specification, but a stream it synthesised carries no codec payload
and so cannot exercise the decoder. That wants a real recording, and a real
recording cannot be committed here: the ones published as references for this
codec are licensed for listening rather than for redistribution under a
BSD-style licence.

So they are fetched instead, into a per-user cache outside every repository:

    python tests/fetch_opus_samples.py

The tests find that cache on their own (``test_opus.SAMPLE_CACHE``); set
``OMI_AUDIO_OPUS_SAMPLES`` to point them somewhere else. Without either they
skip, which is why this exists as a command rather than as something the suite
runs for itself -- a test run that reaches the network is a test run that fails
on an aeroplane.
"""

import os
import sys
import urllib.request

#: What to fetch, as ``{filename: (url, what it is)}``.
#:
#: The Opus project's own published example, which is the reference encoding
#: the format's documentation points at. The recording is 'Paper Lights' by
#: Ehren Starks, CC BY-NC-SA -- fine to fetch and decode, not to redistribute
#: from here, hence the cache.
SAMPLES = {
    'ehren-paper_lights-96.opus': (
        'https://opus-codec.org/static/examples/ehren-paper_lights-96.opus',
        "'Paper Lights' by Ehren Starks, 96 kb/s stereo, CC BY-NC-SA",
    ),
}


def cache_directory() -> str:
    """The directory :mod:`test_opus` looks in when nothing else is named."""
    base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~/.cache')
    return os.path.join(base, 'omi_audio', 'opus-samples')


def fetch(into: str | None = None) -> int:
    """Fetch anything missing; answer how many files are there afterwards."""
    into = into or cache_directory()
    os.makedirs(into, exist_ok=True)
    for name, (url, what) in SAMPLES.items():
        path = os.path.join(into, name)
        if os.path.exists(path):
            print('have %s (%s)' % (name, what))
            continue
        print('fetching %s -- %s' % (name, what))
        # Written beside and moved into place, so an interrupted fetch leaves
        # no half a file for the tests to try to decode.
        partial = path + '.partial'
        with urllib.request.urlopen(url) as source, open(partial, 'wb') as sink:
            sink.write(source.read())
        os.replace(partial, path)
    found = [n for n in os.listdir(into) if n.lower().endswith('.opus')]
    print('%d .opus file(s) in %s' % (len(found), into))
    return len(found)


if __name__ == '__main__':
    raise SystemExit(0 if fetch(*sys.argv[1:2]) else 1)
