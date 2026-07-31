omi_audio API reference
=======================

Renderer-agnostic spatial audio on the glTF ``KHR_audio_emitter`` model, mixed
in NumPy.

This is the reference: every public name, with the docstring that explains why
it is shaped the way it is.  The narrative documentation -- architecture, the
data model, the gain curves with their diagrams, the mixing contract, and how to
drive all of it from a game -- is in `the docs directory
<https://github.com/mcfletch/omi_audio/tree/main/docs>`_.

Start with :mod:`omi_audio.engine`, which is the one object an application
holds.  The modules below are in the order sound travels through them.

.. contents:: On this page
   :local:
   :depth: 1

The package
-----------

.. automodule:: omi_audio
   :no-members:

..
   Members are deliberately not documented here.  ``omi_audio/__init__.py``
   re-exports the whole public surface for convenience, so documenting it at the
   top level as well would give every class two targets and every cross-reference
   an ambiguity.  Each name is documented once, under the module it lives in.

The data model
--------------

.. automodule:: omi_audio.model

Resolving a document's audio
----------------------------

.. automodule:: omi_audio.library

Clips and decoding
------------------

.. automodule:: omi_audio.clip

Spatialisation
--------------

.. automodule:: omi_audio.spatial

Synthesised sounds
------------------

.. automodule:: omi_audio.synth

The mixer
---------

.. automodule:: omi_audio.mixer

The device seam
---------------

.. automodule:: omi_audio.device

The engine
----------

.. automodule:: omi_audio.engine

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
