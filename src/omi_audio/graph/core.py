"""Nodes, their connections, and the block-by-block rendering of a graph.

A :class:`Node` reads the outputs of the nodes connected to it, mixes them to
the number of channels it works in, processes that block and writes its own
output, which nodes further on read in turn.  A :class:`Graph` renders the
nodes its outputs depend on, each once per block and every input before what
reads it, and sums its outputs into the stereo mix.

Channel counting and mixing follow the W3C Web Audio API ("Channel Up-Mixing
and Down-Mixing"):

=================  ==========================================================
``max``            the node works in as many channels as its widest input
``clamped-max``    as ``max``, but no more than :attr:`Node.channel_count`
``explicit``       exactly :attr:`Node.channel_count`
=================  ==========================================================

and an input with another number of channels is mixed to that number by
:func:`mix_matrix`, as speaker layouts (``speakers``) or channel by channel
(``discrete``).

**Threads.**  :meth:`Graph.render` runs on the audio thread, inside
:meth:`~omi_audio.mixer.Mixer.mix`.  Everything else -- connecting nodes,
setting their parameters, :meth:`Graph.compile` -- is the control thread's.
A node's parameters are plain attributes the render reads once a block, and a
change of wiring takes effect when :meth:`Graph.compile` swaps in a new
rendering plan, which it builds completely before the swap.  Every buffer a
node renders into is allocated when the node joins a graph, not while it
renders.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from omi_audio._backend import DEFAULT_SAMPLE_RATE
from omi_audio._numbers import finite

__all__ = [
    'CHANNEL_COUNT_MODES', 'DEFAULT_MAX_BLOCK', 'DEFAULT_MAX_CHANNELS',
    'INTERPRETATIONS', 'Gain', 'Graph', 'Input', 'Node', 'mix_matrix',
]

#: How a node decides how many channels it works in; see the module docstring.
CHANNEL_COUNT_MODES = ('max', 'clamped-max', 'explicit')

#: How an input is mixed to another number of channels.
INTERPRETATIONS = ('speakers', 'discrete')

#: The largest block a graph renders, in frames; the mixer's own maximum.
DEFAULT_MAX_BLOCK = 4096

#: The most channels a node works in.  Eight holds a 7.1 layout; each node's
#: buffer is this many channels of :data:`DEFAULT_MAX_BLOCK` frames.
DEFAULT_MAX_CHANNELS = 8

_HALF = float(np.sqrt(0.5))

#: The speaker layouts Web Audio names, by channel count, as
#: ``{(inputs, outputs): rows}``: each row is one output channel's weights
#: over the input channels.  Mono is ``M``; stereo ``L R``; quad
#: ``L R SL SR``; 5.1 ``L R C LFE SL SR``.
_SPEAKERS: dict[tuple[int, int], tuple[tuple[float, ...], ...]] = {
    (1, 2): ((1.0,), (1.0,)),
    (1, 4): ((1.0,), (1.0,), (0.0,), (0.0,)),
    (1, 6): ((0.0,), (0.0,), (1.0,), (0.0,), (0.0,), (0.0,)),
    (2, 4): ((1, 0), (0, 1), (0, 0), (0, 0)),
    (2, 6): ((1, 0), (0, 1), (0, 0), (0, 0), (0, 0), (0, 0)),
    (4, 6): ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0),
             (0, 0, 1, 0), (0, 0, 0, 1)),
    (2, 1): ((0.5, 0.5),),
    (4, 1): ((0.25, 0.25, 0.25, 0.25),),
    (6, 1): ((_HALF, _HALF, 1.0, 0.0, 0.5, 0.5),),
    (4, 2): ((0.5, 0.0, 0.5, 0.0), (0.0, 0.5, 0.0, 0.5)),
    (6, 2): ((1.0, 0.0, _HALF, 0.0, _HALF, 0.0), (0.0, 1.0, _HALF, 0.0, 0.0, _HALF)),
    (6, 4): ((1.0, 0.0, _HALF, 0.0, 0.0, 0.0), (0.0, 1.0, _HALF, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
}


def mix_matrix(inputs: int, outputs: int,
               interpretation: str = 'speakers') -> NDArray[np.float64]:
    """The ``(outputs, inputs)`` weights mixing one channel layout into another.

    ``speakers`` mixes between the mono, stereo, quad and 5.1 layouts with the
    Web Audio API's weights; any other pair of counts, and ``discrete``, copies
    channel for channel, leaving extra outputs silent and dropping extra inputs.

    Raises:
        ValueError: for an interpretation that is not one of
            :data:`INTERPRETATIONS`.
    """
    if interpretation not in INTERPRETATIONS:
        raise ValueError('channel interpretation %r is not one of %s'
                         % (interpretation, INTERPRETATIONS))
    if interpretation == 'speakers' and inputs != outputs:
        rows = _SPEAKERS.get((inputs, outputs))
        if rows is not None:
            return np.array(rows, dtype='d')
    return np.eye(outputs, inputs, dtype='d')


#: Each mix as the ``(output, input, weight)`` terms that are not zero, which
#: is what the render loops over.  Filled as layouts are met; there are few.
_TERMS: dict[tuple[int, int, str], tuple[tuple[int, int, float], ...]] = {}


def _terms(inputs: int, outputs: int,
           interpretation: str) -> tuple[tuple[int, int, float], ...]:
    """The non-zero terms of :func:`mix_matrix`, kept once computed."""
    key = (inputs, outputs, interpretation)
    found = _TERMS.get(key)
    if found is None:
        matrix = mix_matrix(inputs, outputs, interpretation)
        found = tuple((int(row), int(column), float(matrix[row, column]))
                      for row, column in zip(*np.nonzero(matrix), strict=True))
        _TERMS[key] = found
    return found


class Input(NamedTuple):
    """One connection into a node: a source, and which of its channels.

    ``channel`` None is every channel the source has; a number is that one
    channel, as a mono input, and silence where the source has no such channel.
    """

    source: Node
    channel: int | None = None

    def signal(self, frames: int) -> NDArray[np.float32] | None:
        """The source's last block as this connection carries it, or None for none."""
        given = self.source.output(frames)
        if self.channel is None:
            return given
        if 0 <= self.channel < given.shape[0]:
            return given[self.channel:self.channel + 1]
        return None


class Node:
    """One processing step: inputs in, a block of channels out.

    A subclass implements :meth:`process`, and :meth:`output_channels` where it
    produces another number of channels than it is given.  A node with no
    inputs is a source.

    :attr:`gain` multiplies the output, and is ramped across a block when it
    changes.  :attr:`enabled` false passes the mixed input through unprocessed
    and at unit gain.  :attr:`playing` false and :attr:`paused` true both make
    the output silent; a paused node also keeps its state where it was.
    """

    def __init__(self, gain: float = 1.0, channel_count: int = 2,
                 channel_count_mode: str = 'max',
                 channel_interpretation: str = 'speakers') -> None:
        if channel_count_mode not in CHANNEL_COUNT_MODES:
            raise ValueError('channel count mode %r is not one of %s'
                             % (channel_count_mode, CHANNEL_COUNT_MODES))
        if channel_interpretation not in INTERPRETATIONS:
            raise ValueError('channel interpretation %r is not one of %s'
                             % (channel_interpretation, INTERPRETATIONS))
        #: What this node reads; change it with :meth:`connect` and
        #: :meth:`disconnect`, then :meth:`Graph.compile`.
        self.inputs: list[Input] = []
        self.channel_count = max(1, int(channel_count))
        self.channel_count_mode = channel_count_mode
        self.channel_interpretation = channel_interpretation
        self.enabled = True
        self.playing = True
        self.paused = False
        #: How many channels the last block had.
        self.channels = 1
        self._gain = finite(gain)
        self._gain_now = self._gain
        self._graph: Graph | None = None
        self._out: NDArray[np.float32] = np.zeros((1, 0), dtype=np.float32)

    def __repr__(self) -> str:
        return '<%s %d in %dch>' % (type(self).__name__, len(self.inputs), self.channels)

    @property
    def gain(self) -> float:
        """Linear multiplier on the output; a non-finite value is silence."""
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = finite(value)

    @property
    def graph(self) -> Graph | None:
        """The graph this node renders in, or None."""
        return self._graph

    @property
    def sample_rate(self) -> int:
        """Frames per second of the graph this node is in."""
        return self._graph.sample_rate if self._graph is not None else DEFAULT_SAMPLE_RATE

    def connect(self, source: Node, channel: int | None = None) -> Node:
        """Read ``source`` (or one channel of it) as an input; returns this node."""
        self.inputs.append(Input(source, channel))
        return self

    def disconnect(self, source: Node | None = None) -> Node:
        """Stop reading ``source``, or every input for None; returns this node."""
        self.inputs = [] if source is None else [
            wire for wire in self.inputs if wire.source is not source]
        return self

    def output(self, frames: int) -> NDArray[np.float32]:
        """The last block rendered, ``(channels, frames)``; a view, overwritten next block."""
        return self._out[:self.channels, :frames]

    def output_channels(self, computed: int) -> int:
        """How many channels the output has, given ``computed`` input channels."""
        return computed

    def prepare(self, graph: Graph) -> None:
        """Allocate what processing needs at ``graph``'s rate and block size.

        Called on the control thread when the node joins a graph.
        """

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        """Write the output for ``block`` into ``out[:channels, :frames]``.

        ``block`` is the mixed input, ``(channels, frames)``.  Runs on the
        audio thread: it must not allocate, block or log.
        """
        raise NotImplementedError

    def bind(self, graph: Graph) -> None:
        """Join ``graph``, allocating the output buffer; :meth:`Graph.compile` calls it."""
        if self._out.shape != (graph.max_channels, graph.max_block):
            self._out = np.zeros((graph.max_channels, graph.max_block), dtype=np.float32)
        self._graph = graph
        self.prepare(graph)

    def detach(self) -> None:
        """Leave the graph this node was in, keeping its buffers."""
        self._graph = None

    def _computed(self, inputs: Sequence[Input], limit: int) -> int:
        """How many channels the mixed input has."""
        if self.channel_count_mode == 'explicit':
            return min(self.channel_count, limit)
        widest = 1
        for wire in inputs:
            count = 1 if wire.channel is not None else wire.source.channels
            widest = max(widest, count)
        if self.channel_count_mode == 'clamped-max':
            widest = min(widest, self.channel_count)
        return min(widest, limit)

    def _gather(self, inputs: Sequence[Input], frames: int,
                graph: Graph) -> NDArray[np.float32]:
        """The inputs summed into the graph's scratch block, mixed to one layout."""
        block = graph.scratch(self._computed(inputs, graph.max_channels), frames)
        for wire in inputs:
            given = wire.signal(frames)
            if given is not None:
                graph.mix_into(given, block, self.channel_interpretation)
        return block

    def render(self, frames: int, inputs: Sequence[Input], graph: Graph) -> None:
        """Render one block: gather, then process, bypass or fall silent."""
        block = self._gather(inputs, frames, graph)
        out = self._out
        if not self.enabled:
            self.channels = block.shape[0]
            np.copyto(out[:self.channels, :frames], block)
            return
        self.channels = min(self.output_channels(block.shape[0]), graph.max_channels)
        if not self.playing or self.paused:
            out[:self.channels, :frames] = 0.0
            return
        self.process(block, out, frames)
        self._apply_gain(out[:self.channels, :frames], frames, graph)

    def _apply_gain(self, out: NDArray[np.float32], frames: int, graph: Graph) -> None:
        """Multiply ``out`` by the gain, ramped from the last block's."""
        target, start = self._gain, self._gain_now
        if target == start:
            if target != 1.0:
                np.multiply(out, target, out=out)
            return
        ramp = graph.ramp(start, target, frames)
        np.multiply(out, ramp, out=out)
        self._gain_now = target


class Gain(Node):
    """Its input, multiplied by :attr:`~Node.gain`.

    Also what a graph's destination is: a gain over everything connected to it,
    added to :meth:`Graph.add_output`.
    """

    def process(self, block: NDArray[np.float32], out: NDArray[np.float32],
                frames: int) -> None:
        np.copyto(out[:block.shape[0], :frames], block)


class _Plan(NamedTuple):
    """What one render does: each node and its inputs in order, then the outputs."""

    steps: tuple[tuple[Node, tuple[Input, ...]], ...]
    outputs: tuple[Node, ...]


class Graph:
    """Nodes rendered a block at a time, their outputs summed into stereo.

    ``max_block`` is the largest block :meth:`render` takes, and
    ``max_channels`` the most channels any node works in; every node's buffer
    is sized from them when it joins.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 max_block: int = DEFAULT_MAX_BLOCK,
                 max_channels: int = DEFAULT_MAX_CHANNELS) -> None:
        self.sample_rate = int(sample_rate)
        self.max_block = int(max_block)
        self.max_channels = max(2, int(max_channels))
        self._outputs: list[Node] = []
        self._members: set[Node] = set()
        self._plan = _Plan((), ())
        size = self.max_block
        #: The mixed input of the node being rendered.
        self._block = np.zeros((self.max_channels, size), dtype=np.float32)
        #: One channel of scratch for weighted mixing.
        self._row = np.zeros(size, dtype=np.float32)
        #: A gain ramp, and the 1..n steps it is scaled from.
        self._ramp_buffer = np.zeros(size, dtype=np.float32)
        self._steps = np.arange(1, size + 1, dtype=np.float32)

    def __repr__(self) -> str:
        return '<%s %d nodes %d outputs @%dHz>' % (
            type(self).__name__, len(self._plan.steps), len(self._plan.outputs),
            self.sample_rate)

    # ------------------------------------------------------------------
    # Control thread
    # ------------------------------------------------------------------

    @property
    def outputs(self) -> tuple[Node, ...]:
        """The nodes whose output is heard."""
        return tuple(self._outputs)

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Every node the outputs depend on, inputs first."""
        return tuple(node for node, _ in self._plan.steps)

    @property
    def active(self) -> bool:
        """Whether rendering would produce anything at all."""
        return bool(self._plan.outputs)

    def add_output(self, node: Node) -> None:
        """Hear ``node``, and compile.

        Raises:
            ValueError: where the wiring has a cycle, or a node is in another
                graph; the graph is left as it was.
        """
        self.set_outputs([*self._outputs, node])

    def remove_output(self, node: Node) -> None:
        """Stop hearing ``node``, and compile."""
        self.set_outputs([output for output in self._outputs if output is not node])

    def set_outputs(self, nodes: Iterable[Node]) -> None:
        """Hear exactly ``nodes``, and compile; see :meth:`add_output`."""
        previous = self._outputs
        self._outputs = list(nodes)
        try:
            self.compile()
        except ValueError:
            self._outputs = previous
            raise

    def compile(self) -> None:
        """Build the rendering plan from the outputs and the current wiring.

        Call after changing any node's inputs.  The plan is built and every
        node's buffers allocated before the plan the audio thread reads is
        replaced.

        Raises:
            ValueError: where the wiring has a cycle, or a node is in another
                graph.
        """
        order = _ordered(self._outputs)
        for node in order:
            if node.graph is not None and node.graph is not self:
                raise ValueError('%r is already in another graph' % (node,))
        for node in order:
            if node.graph is None:
                node.bind(self)
        steps = tuple((node, tuple(node.inputs)) for node in order)
        self._plan = _Plan(steps, tuple(dict.fromkeys(self._outputs)))
        kept = set(order)
        for node in self._members - kept:
            node.detach()
        self._members = kept

    # ------------------------------------------------------------------
    # Audio thread
    # ------------------------------------------------------------------

    def render(self, frames: int, out: NDArray[np.float32] | None = None) -> None:
        """Render one block; add the outputs, as stereo, into ``out`` ``(frames, 2)``.

        Raises:
            ValueError: for a block larger than ``max_block``.
        """
        if frames > self.max_block:
            raise ValueError('block of %d frames exceeds max_block %d'
                             % (frames, self.max_block))
        plan = self._plan
        for node, inputs in plan.steps:
            node.render(frames, inputs, self)
        if out is None:
            return
        for node in plan.outputs:
            given = node.output(frames)
            for row, column, weight in _terms(given.shape[0], 2, 'speakers'):
                target = out[:frames, row]
                if weight == 1.0:
                    np.add(target, given[column], out=target)
                else:
                    row_buffer = self._row[:frames]
                    np.multiply(given[column], weight, out=row_buffer)
                    np.add(target, row_buffer, out=target)

    def scratch(self, channels: int, frames: int) -> NDArray[np.float32]:
        """A zeroed ``(channels, frames)`` block for gathering a node's input.

        One block, shared by every node and reused as each renders.
        """
        block = self._block[:channels, :frames]
        block.fill(0.0)
        return block

    def mix_into(self, given: NDArray[np.float32], block: NDArray[np.float32],
                 interpretation: str) -> None:
        """Add ``given`` into ``block``, mixed to ``block``'s channel count."""
        if given.shape[0] == block.shape[0]:
            np.add(block, given, out=block)
            return
        frames = block.shape[1]
        row_buffer = self._row[:frames]
        for row, column, weight in _terms(given.shape[0], block.shape[0], interpretation):
            if weight == 1.0:
                np.add(block[row], given[column], out=block[row])
            else:
                np.multiply(given[column], weight, out=row_buffer)
                np.add(block[row], row_buffer, out=block[row])

    def ramp(self, start: float, end: float, frames: int) -> NDArray[np.float32]:
        """``frames`` values from just past ``start`` to ``end``, linearly.

        A view of one buffer, overwritten by the next call.
        """
        ramp = self._ramp_buffer[:frames]
        np.multiply(self._steps[:frames], (end - start) / frames, out=ramp)
        np.add(ramp, start, out=ramp)
        return ramp


def _ordered(outputs: Sequence[Node]) -> list[Node]:
    """Every node ``outputs`` depend on, each after all of its inputs.

    Raises:
        ValueError: where a node depends on itself.
    """
    done: set[Node] = set()
    visiting: set[Node] = set()
    order: list[Node] = []
    for output in outputs:
        if output in done:
            continue
        stack: list[tuple[Node, int]] = [(output, 0)]
        visiting.add(output)
        while stack:
            node, index = stack[-1]
            if index < len(node.inputs):
                stack[-1] = (node, index + 1)
                source = node.inputs[index].source
                if source in visiting:
                    raise ValueError('the graph has a cycle through %r' % (source,))
                if source not in done:
                    visiting.add(source)
                    stack.append((source, 0))
                continue
            stack.pop()
            visiting.discard(node)
            done.add(node)
            order.append(node)
    return order
