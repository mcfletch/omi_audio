"""Routing channels: splitting a signal apart, picking one, and merging them.

A :class:`ChannelSplitter` passes its input on with each channel kept apart
(``discrete``), for nodes that read one channel of it:
``node.connect(splitter, channel=1)``.  A :class:`ChannelSelector` is one
channel of its input, as mono.  A :class:`ChannelMerger` makes one channel of
each of its inputs, in the order they were connected, each mixed to mono.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omi_audio.graph.core import Graph, Input, Node

__all__ = ['ChannelMerger', 'ChannelSelector', 'ChannelSplitter']


class ChannelSplitter(Node):
    """Its input, channels kept apart for nodes that read them one at a time."""

    def __init__(self, gain: float = 1.0, **named: Any) -> None:
        named.setdefault('channel_interpretation', 'discrete')
        super().__init__(gain=gain, **named)

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        np.copyto(out[:block.shape[0], :frames], block)


class ChannelSelector(Node):
    """Channel ``selection`` of its input, counting from 0; silence where there is none."""

    def __init__(self, selection: int = 0, gain: float = 1.0, **named: Any) -> None:
        super().__init__(gain=gain, **named)
        self.selection = int(selection)

    def output_channels(self, _computed: int) -> int:
        return 1

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        if 0 <= self.selection < block.shape[0]:
            np.copyto(out[0, :frames], block[self.selection])
        else:
            out[0, :frames] = 0.0


class ChannelMerger(Node):
    """One channel for each input, in order, each input mixed to mono."""

    def _gather(self, inputs: Sequence[Input], frames: int,
                graph: Graph) -> NDArray[np.float32]:
        count = max(1, min(len(inputs), graph.max_channels))
        block = graph.scratch(count, frames)
        for index, wire in enumerate(inputs[:count]):
            given = wire.signal(frames)
            if given is not None:
                graph.mix_into(given, block[index:index + 1], 'speakers')
        return block

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        np.copyto(out[:block.shape[0], :frames], block)
