# Contributing

Bug reports, questions and patches all welcome. This file is short because the
rules that matter are few.

## Getting set up

```bash
git clone https://github.com/mcfletch/omi_audio
cd omi_audio
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

That runs the whole suite in under a second. `miniaudio` comes in with the
`dev` extra; without it the package still works and simply stays silent, which
is a path the suite exercises on purpose.

## Before you open a pull request

```bash
ruff check .                 # lint
mypy --strict src/omi_audio  # types
pytest --cov --cov-branch    # tests, with the coverage floor
```

or, all of it the way CI does:

```bash
tox -e lint,typecheck,py312-playback,py312-nobackend
```

All three are gates. A merge needs them green.

## What the code is held to

This is a small library that other people are meant to read, so:

- **Tests come first, and they must fail before they pass.** Write the assertion,
  watch it go red, then make it green. A test written after the code often only
  proves the code does what it does.
- **A test must be able to fail for the reason it is named after.** `assert
  handle is not None` in a test called *the source's gain is applied* is not a
  test of the gain. If you are unsure a test is real, break the thing it covers
  and check that it notices.
- **Docstrings explain *why*, not *what*.** The signature says what. The
  docstring says what problem this shape avoids, in prose, and names the failure
  it is designed against. That habit is the most valuable thing in this
  codebase; please keep it up.
- **Documentation ships with the change, not after it.** A new option, a changed
  default, a new file format, a changed public API — none is finished while
  `docs/` still describes the old world. Say in your pull request which
  documentation you changed, and if you changed none, say that and why.
- **`%`-formatting, not f-strings**, matching the surrounding code and required
  by the logging calls (`log.warning('%s', x)` defers the work until somebody is
  listening). `ruff` is configured accordingly.

## Two rules with teeth

### The audio thread

Everything in `mixer.py` from `mix()` down runs on the device's own thread.
There it must **not allocate, block, decode, resolve a path, take a lock, or
log**. An allocation on the audio thread is a garbage collection on the audio
thread, and that is an audible gap.

Pre-allocate in `__init__` and write with NumPy's `out=`. Two tests assert this
under `tracemalloc`; if you make them fail, the answer is to allocate less, not
to raise the threshold.

There is exactly one sanctioned exception, and it is documented where it lives:
the oversized-block path may grow its silence buffer, once per size, on a path
that is already broken.

Numbers crossing *onto* the thread are checked on the control side, where it is
free — see `_finite()`. A NaN gain survives `np.clip` and every voice shares one
buffer, so one bad emitter would otherwise silence the whole scene.

### The document is not trusted

`omi_audio` consumes glTF from third parties. A `uri` is **never** resolved,
opened or interpreted here; that is the application's job, through
`AudioLibrary`. `model.from_gltf()` must never raise and must never let a value
of the wrong type into the model. See [SECURITY.md](SECURITY.md).

## Changing a gain curve

The diagrams in `docs/images/` are generated from `omi_audio.spatial` by
`docs/make_diagrams.py`, and `tests/test_diagrams.py` fails if the committed
files stop matching. So:

```bash
python docs/make_diagrams.py     # or: tox -e diagrams
```

and commit what changes. A picture of a gain curve is a claim about the code.

## Releasing

**Bumping the version is what cuts a release.** `.github/workflows/release.yml`
runs on every push to `main`: it runs the whole test workflow, then reads
`__version__` from `src/omi_audio/__init__.py` and asks PyPI whether that
version exists. If it does, the push is an ordinary CI run and nothing is
published. If it does not, the sdist and wheel are built and uploaded.

So a release is:

1. Update `CHANGELOG.md` under a new heading.
2. Bump `__version__` in `src/omi_audio/__init__.py`.
3. Merge to `main`.

Nothing is published from a red build. PyPI versions are immutable, so a
release number burned on a broken build cannot be taken back — which is why the
publish job waits on every released Python, both backends, lint, types, the
coverage floor and the docs build, and why the version check treats anything
other than a clean 200 or 404 from PyPI as a reason to stop rather than to guess.

The one check that is *not* binding is Python 3.15, which is still a
prerelease. It runs on every push and its failures are visible, but a
regression in a CPython beta is not a reason to hold up a merge or a release.
Make it binding by moving `"3.15"` back into the `test` matrix in
`.github/workflows/test.yml` and deleting the `prerelease` job.

### One-time setup

The workflow needs two things that do not live in this repository, and it
cannot publish until both exist:

**A PyPI trusted publisher** for this project — owner `mcfletch`, repository
`omi_audio`, workflow `release.yml`, and the **environment field left blank**.
The workflow uses no GitHub environment, so its OIDC claim carries none; a
publisher that names one will not match, and the upload is rejected with an
error that does not obviously say why.

Before the first release the project does not exist on PyPI yet, so this has to
be added as a *pending* publisher (PyPI → Your projects → Publishing).

No API token is stored anywhere: the upload authenticates over OIDC. Note that
without an environment there is no place to require a reviewer, so a green push
to `main` with a new version publishes without anyone approving it — the version
bump is the decision.

## Licensing

`omi_audio` is MIT, and everything it depends on is MIT or public domain. **Do
not contribute code copied or translated from a GPL, LGPL, AGPL, SSPL or CC-BY-SA
source**, in whole or in part. The `miniaudio` choice was made on exactly these
grounds — the convenient alternatives (`libsndfile`, PyAV, `pydub`-via-ffmpeg)
are copyleft — and the reasoning is recorded in `pyproject.toml` so it is
not quietly undone.

Behaviour taken from a published *specification* is fine and welcome; cite it
from the docstring, as the existing code cites `KHR_audio_emitter`, the Web
Audio API and ISO/IEC 14772-1.

## A note on how this was written

Much of this package is LLM-written, and it says so in the README. That is not a
reason to hold a contribution to a lower standard — it is the reason the tests,
the review in `docs/reviews/` and the gates above exist.
