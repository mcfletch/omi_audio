# Security

## The threat model, stated plainly

`omi_audio` exists to consume **glTF documents from third parties**. A scene
file is data from somebody you have never met, and everything in it — a `uri`, a
`gain`, an emitter index, a `bufferView` number — is attacker-controlled input if
your application loads content it did not author.

Two properties follow, and they are the ones to hold this library to:

### 1. It never resolves a `uri`

A `uri` in a document is a string. This library does not open it, normalise it,
resolve it against a base directory, or hand it to a decoder. It never sees a
filesystem or a network.

Resolution belongs to the application, through
[`AudioLibrary`](src/omi_audio/library.py), because only the application knows
where its content lives and what it is willing to fetch. `ClipCache` likewise
treats a name as a dictionary key and nothing more.

**This means your resolver is the security boundary.** When you write the
`fetch` callback, that is where the decisions live:

- Reject anything with a scheme you do not serve (`file:`, `http:`, `data:` —
  each is a policy decision, not a default).
- Percent-decode before you compare, because glTF URIs are percent-encoded.
- Confine the result to your content directory — `os.path.realpath` and then
  `os.path.commonpath`, not a string prefix test.
- On Windows, reject UNC paths (`\\host\share\...`): resolving one is an
  outbound SMB connection and, with default settings, an NTLM credential leak.

### 2. Malformed content costs a sound, not a process

`model.from_gltf()` never raises, whatever the document contains, and never
admits a value of the wrong type into the model. `AudioDocument.sources_for()`
and `audio_for()` skip indices that point at nothing. A clip that will not decode
is a warning and a silence.

That is a robustness property, not a sandbox. It says a hostile document cannot
crash your loader or smuggle a string into the mixer; it does not say the
underlying decoder is safe.

## What is not defended

- **The decoder.** Decoding is `miniaudio`, which wraps C decoders for MP3,
  Vorbis, FLAC and WAV. A memory-safety bug in one of those is reachable by any
  audio your application chooses to decode. If you decode content from strangers,
  that risk is real and it is not one this library can remove. `omi_audio` runs
  without `miniaudio` at all — it simply stays silent — which is a legitimate
  configuration for a service that must not decode untrusted audio.
- **Resource exhaustion.** Clips are decoded whole into memory. A document
  naming a very long audio file will use a great deal of it. Bound what you
  fetch, in your resolver, before you supply it.
- **Anything above this library.** It has no network client, no subprocess, no
  deserialisation of pickles, no `eval`, and no plugin loading. If your loader
  has those, they are your perimeter.

## Reporting a vulnerability

Report privately, not as a public issue:

- GitHub's **[private vulnerability reporting](https://github.com/mcfletch/omi_audio/security/advisories/new)**, or
- email **mcfletch@vrplumber.com**.

Please include what you were loading, what happened, and a document or file that
reproduces it if you have one. You should get an acknowledgement within a week.

## Supported versions

This is an alpha (`0.1.0a1`). Fixes go onto the latest release only; there are
no maintained branches to backport to yet.

## A standing caveat

`omi_audio` is **largely LLM-written**, and says so in its README, its package
docstring and its licence. It has a substantial test suite and has been through
a documented review, but it comes with no guarantees. Review it before relying on
it for anything that matters — which, if you are reading this file, you already
were.
