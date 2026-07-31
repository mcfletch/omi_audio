# omi_audio documentation

Deep-dive documentation for the `omi_audio` engine. Start with the
[project README](../README.md) for install and quick start, or with
[`examples/orbit.py`](../examples/orbit.py) for something you can run.

## Using it

- **[GAME-INTEGRATION.md](GAME-INTEGRATION.md)** — the frame loop, handle
  lifetime, voice budgets, priority, and the three patterns that cover most
  games. Read this one first if you are building something.
- **[VRML97.md](VRML97.md)** — the field-by-field map from a VRML97 `Sound` node
  onto this API, and what is not implemented.

## How it works

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the two threads, every module's job,
  and the design rules that follow from the split.
- **[DATA-MODEL.md](DATA-MODEL.md)** — `KHR_audio_emitter` as the in-memory
  model, glTF round-tripping, where an emitter *is*, and who is allowed to
  resolve a `uri`.
- **[SPATIALISATION.md](SPATIALISATION.md)** — every gain curve, with a diagram
  of each and the specification it comes from.
- **[MIXING.md](MIXING.md)** — the voice pool, the audio-thread contract, and
  the rules the mixer is not allowed to break.

## Reference

- **[API reference](api/)** — the rendered docstrings. `tox -e docs` builds it;
  CI publishes it as an artifact.

## Notes on the documentation itself

- The flow diagrams are [Mermaid](https://mermaid.js.org/); GitHub renders them
  inline.
- The **gain-curve diagrams in `images/` are generated** from
  `omi_audio.spatial` by [`make_diagrams.py`](make_diagrams.py), and
  `tests/test_diagrams.py` fails if a committed picture stops matching the code.
  Regenerate them with `python docs/make_diagrams.py` after changing a formula.

## Reviews

- **[2026-07-31](reviews/2026-07-31-code-review.md)** — full pre-release review:
  correctness, security, spec conformance, documentation and test-suite audit.
  Its findings were addressed before the first release; it is kept as a record
  of the state it describes, not of the present one.
