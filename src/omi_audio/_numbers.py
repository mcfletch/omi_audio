"""The check every number makes where it crosses onto the audio thread."""

from __future__ import annotations

import math


def finite(value: float) -> float:
    """``value`` as a float, with anything that is not a number replaced by 0.

    Positions come out of a scenegraph, and a scenegraph produces non-finite
    numbers for ordinary reasons: a transform with a zero scale, an
    uninitialised bone, a physics step that blew up, a normalisation of a zero
    vector.  Those become NaN gains, and a NaN multiplied into the mix stays --
    ``np.clip`` does not remove it -- so every voice sharing the output buffer
    goes with it.

    Silence is the substitute: it is what the emitter would sound like if it
    were nowhere, which is what a NaN position says it is.
    """
    number = float(value)
    return number if math.isfinite(number) else 0.0
