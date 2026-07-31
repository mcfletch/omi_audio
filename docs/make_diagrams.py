#!/usr/bin/env python3
"""Draw the diagrams in ``docs/images/`` **out of** :mod:`omi_audio.spatial`.

Every curve here is plotted by calling the same function the mixer calls.  That
is the whole reason this script exists rather than a drawing program: a picture
of a gain curve is a claim about the code, and a claim that is redrawn by hand
is a claim that goes quietly wrong the first time somebody touches the formula.
``tests/test_diagrams.py`` regenerates the files and fails if they have moved,
so the pictures in the documentation cannot drift from the code they describe.

Run it after changing anything in :mod:`omi_audio.spatial`::

    python docs/make_diagrams.py

Output is hand-written SVG with no dependency beyond NumPy, which the package
already requires -- adding matplotlib to build the documentation would be a
heavier price than these few hundred lines.

The palette is chosen to read on both a light and a dark page, because GitHub
renders this documentation either way and an ``<img>`` inherits nothing from
the page around it.
"""

from __future__ import annotations

import math
import pathlib
import sys

# Importable from a checkout without installing, since this is a build step.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'src'))

from omi_audio import spatial                                       # noqa: E402

#: Axes, rules and body text: mid grey, legible against white and against black.
INK = '#8a8a8a'
#: Slightly stronger, for labels that have to be read rather than glanced at.
LABEL = '#a8a8a8'
#: One colour per curve, saturated enough to survive either background.
RED, BLUE, GREEN, AMBER, VIOLET = '#e05252', '#3d8fd8', '#3fa860', '#d1902a', '#9a72d0'
#: Fills are the curve colour at low alpha, so a shaded band never hides a rule.
FAINT = 0.16

FONT = 'font-family="ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif"'


# ----------------------------------------------------------------------
# The smallest SVG writer that will do
# ----------------------------------------------------------------------

def svg(width: int, height: int, title: str, body: list[str]) -> str:
    """A complete SVG document.  ``title`` is what a screen reader announces."""
    return '\n'.join([
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" '
        'height="%d" role="img" aria-label="%s">' % (width, height, width, height, title),
        '  <title>%s</title>' % (title,),
        *['  ' + line for line in body],
        '</svg>',
        ''])


def path(points, colour: str, width: float = 2.0, dash: str = '') -> str:
    """A polyline through ``points``, which are already in SVG coordinates."""
    drawn = ' '.join('%s%.2f %.2f' % ('M' if index == 0 else 'L', x, y)
                     for index, (x, y) in enumerate(points))
    style = ' stroke-dasharray="%s"' % (dash,) if dash else ''
    return ('<path d="%s" fill="none" stroke="%s" stroke-width="%.1f" '
            'stroke-linejoin="round" stroke-linecap="round"%s/>'
            % (drawn, colour, width, style))


def area(points, colour: str) -> str:
    """A filled region, for the band between two curves."""
    drawn = ' '.join('%s%.2f %.2f' % ('M' if index == 0 else 'L', x, y)
                     for index, (x, y) in enumerate(points)) + ' Z'
    return '<path d="%s" fill="%s" fill-opacity="%.2f" stroke="none"/>' % (
        drawn, colour, FAINT)


def line(x1: float, y1: float, x2: float, y2: float, colour: str = INK,
         width: float = 1.0, dash: str = '') -> str:
    style = ' stroke-dasharray="%s"' % (dash,) if dash else ''
    return ('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" '
            'stroke-width="%.1f"%s/>' % (x1, y1, x2, y2, colour, width, style))


def dot(x: float, y: float, colour: str, radius: float = 4.0) -> str:
    return '<circle cx="%.2f" cy="%.2f" r="%.1f" fill="%s"/>' % (x, y, radius, colour)


def label(x: float, y: float, text: str, colour: str = LABEL,
          anchor: str = 'start', size: int = 12, weight: str = 'normal') -> str:
    return ('<text x="%.2f" y="%.2f" fill="%s" text-anchor="%s" %s '
            'font-size="%d" font-weight="%s">%s</text>'
            % (x, y, colour, anchor, FONT, size, weight, _escape(text)))


def _escape(text: str) -> str:
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class Axes:
    """A rectangle of the page, and the arithmetic to put data points in it."""

    def __init__(self, left, top, width, height, x_range, y_range):
        self.left, self.top, self.width, self.height = left, top, width, height
        self.x_range, self.y_range = x_range, y_range

    def x(self, value: float) -> float:
        low, high = self.x_range
        return self.left + self.width * (value - low) / (high - low)

    def y(self, value: float) -> float:
        low, high = self.y_range
        return self.top + self.height * (1.0 - (value - low) / (high - low))

    def at(self, x: float, y: float) -> tuple[float, float]:
        return self.x(x), self.y(y)

    def frame(self) -> list[str]:
        bottom, right = self.top + self.height, self.left + self.width
        return [line(self.left, self.top, self.left, bottom),
                line(self.left, bottom, right, bottom)]


# ----------------------------------------------------------------------
# The diagrams
# ----------------------------------------------------------------------

def distance_models() -> str:
    """The three glTF distance curves, on one set of axes.

    ``exponential`` is dashed because at ``rolloffFactor`` 1 -- the default, and
    what the rest of this diagram uses -- it is *numerically identical* to
    ``inverse``: one is ``ref / d`` and the other is ``(d / ref) ** -1``.  Drawn
    solid it would sit exactly on top of the inverse curve and hide it, leaving
    a legend with three entries and a picture with two lines.
    """
    axes = Axes(64, 28, 480, 260, (0.0, 24.0), (0.0, 1.05))
    body = ['<g>']

    # Grid and scales.
    for gain in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = axes.y(gain)
        body.append(line(axes.left, y, axes.left + axes.width, y, INK, 0.5,
                         dash='' if gain == 0.0 else '3 4'))
        body.append(label(axes.left - 8, y + 4, '%.2f' % (gain,), INK, anchor='end'))
    for metres in (0, 6, 12, 18, 24):
        x = axes.x(metres)
        body.append(label(x, axes.top + axes.height + 18, '%d' % (metres,), INK,
                          anchor='middle'))

    reference, maximum = 2.0, 16.0
    samples = [index * 24.0 / 400.0 for index in range(401)]
    curves = [('inverse', RED, 'inverse (default)', ''),
              ('exponential', BLUE, 'exponential', '7 5'),
              ('linear', GREEN, 'linear', '')]
    for name, colour, _, dash in curves:
        points = [axes.at(distance, spatial.distance_gain(
            distance, name, ref_distance=reference, max_distance=maximum,
            rolloff_factor=1.0)) for distance in samples]
        body.append(path(points, colour, 2.2, dash=dash))

    # The two distances that shape every curve.
    for value, text, colour in ((reference, 'refDistance', AMBER),
                                (maximum, 'maxDistance', VIOLET)):
        x = axes.x(value)
        body.append(line(x, axes.top, x, axes.top + axes.height, colour, 1.2, dash='4 4'))
        body.append(label(x + 5, axes.top + 12, text, colour))

    # Legend, and the one sentence the picture exists to make obvious.
    for index, (_, colour, shown, dash) in enumerate(curves):
        y = 316 + index * 18
        body.append(line(64, y - 4, 92, y - 4, colour, 2.2, dash=dash))
        body.append(label(100, y, shown, LABEL))
    body.append(label(300, 316, 'Only linear reaches silence, and it does so',
                      LABEL))
    body.append(label(300, 334, 'at maxDistance. The other two fall for ever', LABEL))
    body.append(label(300, 352, 'and never use maxDistance at all.', LABEL))
    body.append(label(64, 380,
                      'At rolloffFactor 1 — the default, and what is drawn here —',
                      LABEL))
    body.append(label(64, 398,
                      'inverse and exponential are the same curve, so the dashed',
                      LABEL))
    body.append(label(64, 416, 'line lies exactly on the solid one.', LABEL))

    body.append(label(304, axes.top + axes.height + 40, 'distance from the listener (m)',
                      INK, anchor='middle'))
    body.append(label(20, 158, 'gain', INK, anchor='middle'))
    body.extend(axes.frame())
    body.append('</g>')
    return svg(576, 432, 'The three KHR_audio_emitter distance models', body)


def cone() -> str:
    """The directional cone: the angles are diameters, drawn as diameters."""
    inner_angle, outer_angle, outer_gain = math.pi / 3, math.pi * 0.85, 0.15
    body = ['<g>']

    # ---- left: the cone in plan, seen from above the emitter ----
    origin_x, origin_y, reach = 168.0, 176.0, 128.0

    def wedge(half_angle: float, colour: str) -> str:
        """The sector within ``half_angle`` either side of the axis."""
        points = [(origin_x, origin_y)]
        steps = 48
        for index in range(steps + 1):
            angle = -half_angle + 2.0 * half_angle * index / steps
            points.append((origin_x + reach * math.sin(angle),
                           origin_y - reach * math.cos(angle)))
        return area(points, colour)

    body.append(wedge(outer_angle / 2.0, AMBER))
    body.append(wedge(inner_angle / 2.0, GREEN))
    for half, colour in ((inner_angle / 2.0, GREEN), (outer_angle / 2.0, AMBER)):
        for sign in (-1, 1):
            body.append(line(origin_x, origin_y,
                             origin_x + sign * reach * math.sin(half),
                             origin_y - reach * math.cos(half), colour, 1.6))
    body.append(line(origin_x, origin_y, origin_x, origin_y - reach - 16, INK, 1.0,
                     dash='4 4'))
    body.append(label(origin_x, origin_y - reach - 24, 'the emitter’s −Z axis',
                      INK, anchor='middle'))
    body.append(dot(origin_x, origin_y, RED, 5))
    body.append(label(origin_x + 10, origin_y + 16, 'emitter', RED))
    body.append(label(origin_x, origin_y - 78, 'no attenuation', GREEN, anchor='middle'))
    body.append(label(origin_x - 96, origin_y - 26, 'ramp', AMBER, anchor='middle'))
    body.append(label(origin_x + 96, origin_y - 26, 'ramp', AMBER, anchor='middle'))
    body.append(label(origin_x, origin_y + 60, 'coneOuterGain', VIOLET, anchor='middle'))
    body.append(label(168, 330,
                      'The angles are angular DIAMETERS, so each boundary',
                      LABEL, anchor='middle'))
    body.append(label(168, 348, 'sits at half of the value the document gives.',
                      LABEL, anchor='middle'))

    # ---- right: the same thing as a curve ----
    axes = Axes(392, 40, 160, 216, (0.0, math.pi), (0.0, 1.08))
    for gain in (0.0, 0.5, 1.0):
        y = axes.y(gain)
        body.append(line(axes.left, y, axes.left + axes.width, y, INK, 0.5,
                         dash='' if gain == 0.0 else '3 4'))
        body.append(label(axes.left - 8, y + 4, '%.1f' % (gain,), INK, anchor='end'))
    points = [axes.at(angle, spatial.cone_gain(angle, inner_angle, outer_angle,
                                               outer_gain))
              for angle in [index * math.pi / 300.0 for index in range(301)]]
    body.append(path(points, BLUE, 2.2))
    for half, colour in ((inner_angle / 2.0, GREEN), (outer_angle / 2.0, AMBER)):
        x = axes.x(half)
        body.append(line(x, axes.top, x, axes.top + axes.height, colour, 1.2, dash='4 4'))
    body.append(label(axes.x(inner_angle / 2.0), axes.top - 8, 'inner/2', GREEN,
                      anchor='middle'))
    body.append(label(axes.x(outer_angle / 2.0), axes.top - 24, 'outer/2', AMBER,
                      anchor='middle'))
    body.append(label(axes.left + axes.width, axes.y(outer_gain) - 8, 'coneOuterGain',
                      VIOLET, anchor='end'))
    body.append(label(axes.left, axes.top + axes.height + 18, '0', INK, anchor='middle'))
    body.append(label(axes.left + axes.width, axes.top + axes.height + 18, 'π',
                      INK, anchor='end'))
    body.append(label(axes.left + axes.width / 2, axes.top + axes.height + 36,
                      'angle off the axis (rad)', INK, anchor='middle'))
    body.extend(axes.frame())
    body.append('</g>')
    return svg(576, 370, 'The KHR_audio_emitter directional cone', body)


def ellipsoids() -> str:
    """VRML97's two ellipsoids, each with a focus at the sound."""
    fields = dict(min_front=6.0, min_back=2.0, max_front=16.0, max_back=5.0)
    body = ['<g>']
    origin_x, origin_y, scale = 220.0, 200.0, 9.5

    def surface(front: float, back: float) -> list[tuple[float, float]]:
        """The focal conic, from :func:`~omi_audio.spatial.ellipsoid_reach`."""
        points = []
        for index in range(361):
            theta = index * 2.0 * math.pi / 360.0
            reach = spatial.ellipsoid_reach(front, back, math.cos(theta))
            points.append((origin_x + scale * reach * math.sin(theta),
                           origin_y - scale * reach * math.cos(theta)))
        return points

    outer = surface(fields['max_front'], fields['max_back'])
    inner = surface(fields['min_front'], fields['min_back'])
    body.append(area(outer, AMBER))
    body.append(area(inner, GREEN))
    body.append(path(outer, AMBER, 1.8))
    body.append(path(inner, GREEN, 1.8))

    body.append(line(origin_x, origin_y + scale * fields['max_back'] + 20,
                     origin_x, origin_y - scale * fields['max_front'] - 22, INK, 1.0,
                     dash='4 4'))
    body.append(label(origin_x + 6, origin_y - scale * fields['max_front'] - 28,
                      'direction', INK))
    body.append(dot(origin_x, origin_y, RED, 5))
    body.append(label(origin_x + 10, origin_y + 17,
                      'location — a focus of both', RED))

    for value, colour, text in ((fields['min_front'], GREEN, 'minFront'),
                                (fields['max_front'], AMBER, 'maxFront')):
        y = origin_y - scale * value
        body.append(line(origin_x, y, origin_x + 96, y, colour, 1.0, dash='2 3'))
        body.append(label(origin_x + 100, y + 4, text, colour))
    for value, colour, text in ((fields['min_back'], GREEN, 'minBack'),
                                (fields['max_back'], AMBER, 'maxBack')):
        y = origin_y + scale * value
        body.append(line(origin_x, y, origin_x - 96, y, colour, 1.0, dash='2 3'))
        body.append(label(origin_x - 100, y + 4, text, colour, anchor='end'))

    body.append(label(origin_x, origin_y - 30, 'full volume', GREEN, anchor='middle'))
    body.append(label(origin_x, origin_y - 92, '0 dB → −20 dB', AMBER,
                      anchor='middle'))
    body.append(label(origin_x, 336, 'silent beyond the outer surface', INK,
                      anchor='middle'))

    # The ramp itself, so "linear in decibels" is visible rather than asserted.
    axes = Axes(412, 60, 140, 200, (0.0, 1.0), (0.0, 1.05))
    for gain in (0.0, 0.5, 1.0):
        y = axes.y(gain)
        body.append(line(axes.left, y, axes.left + axes.width, y, INK, 0.5,
                         dash='' if gain == 0.0 else '3 4'))
        body.append(label(axes.left - 8, y + 4, '%.1f' % (gain,), INK, anchor='end'))
    inner_reach = spatial.ellipsoid_reach(fields['min_front'], fields['min_back'], 1.0)
    outer_reach = spatial.ellipsoid_reach(fields['max_front'], fields['max_back'], 1.0)
    ramp = []
    for index in range(201):
        across = index / 200.0
        distance = inner_reach + across * (outer_reach - inner_reach)
        ramp.append(axes.at(across, spatial.ellipsoid_gain(distance, 1.0, **fields)))
    body.append(path(ramp, AMBER, 2.2))
    body.append(label(axes.left, axes.top - 12, 'the ramp is linear in dB,', LABEL))
    body.append(label(axes.left, axes.top - 30, 'which is why it curves here', LABEL))
    body.append(label(axes.left, axes.top + axes.height + 18, 'inner', GREEN))
    body.append(label(axes.left + axes.width, axes.top + axes.height + 18, 'outer',
                      AMBER, anchor='end'))
    body.extend(axes.frame())
    body.append('</g>')
    return svg(576, 350, 'The two ellipsoids of a VRML97 Sound node', body)


def pan() -> str:
    """Equal-power panning, and the fold that puts "behind" in front."""
    body = ['<g>']
    axes = Axes(64, 36, 300, 216, (-math.pi, math.pi), (0.0, 1.08))
    for gain in (0.0, 0.5, 1.0):
        y = axes.y(gain)
        body.append(line(axes.left, y, axes.left + axes.width, y, INK, 0.5,
                         dash='' if gain == 0.0 else '3 4'))
        body.append(label(axes.left - 8, y + 4, '%.1f' % (gain,), INK, anchor='end'))
    azimuths = [-math.pi + index * 2.0 * math.pi / 400.0 for index in range(401)]
    for channel, colour, text in ((0, BLUE, 'left'), (1, RED, 'right')):
        points = [axes.at(azimuth, spatial.equal_power_pan(azimuth)[channel])
                  for azimuth in azimuths]
        body.append(path(points, colour, 2.2))
        body.append(label(axes.x(math.pi) + 6,
                          axes.y(spatial.equal_power_pan(math.pi)[channel]) + 4,
                          text, colour))
    for value, text in ((-math.pi / 2, 'hard left'), (0.0, 'ahead'),
                        (math.pi / 2, 'hard right')):
        x = axes.x(value)
        body.append(line(x, axes.top, x, axes.top + axes.height, INK, 0.8, dash='4 4'))
        body.append(label(x, axes.top - 8, text, INK, anchor='middle'))
    body.append(label(axes.left, axes.top + axes.height + 18, '−π', INK))
    body.append(label(axes.left + axes.width, axes.top + axes.height + 18, 'π',
                      INK, anchor='end'))
    body.append(label(axes.left + axes.width / 2, axes.top + axes.height + 38,
                      'azimuth (rad), positive to the right', INK, anchor='middle'))
    body.extend(axes.frame())

    body.append(label(64, 300, 'Behind the listener the curves come back: a source',
                      LABEL))
    body.append(label(64, 318, 'behind-and-right pans right, because two speakers',
                      LABEL))
    body.append(label(64, 336, 'cannot put a sound behind anybody. Elevation is not',
                      LABEL))
    body.append(label(64, 354, 'represented at all — azimuth spends the only degree',
                      LABEL))
    body.append(label(64, 372, 'of freedom a pair of ear gains has.', LABEL))

    # The arc itself: left**2 + right**2 is 1 at every angle.
    centre_x, centre_y, radius = 470.0, 236.0, 96.0
    body.append(line(centre_x - radius - 14, centre_y, centre_x + radius + 14, centre_y,
                     INK, 1.0))
    quarter = [(centre_x + radius * spatial.equal_power_pan(a)[1],
                centre_y - radius * spatial.equal_power_pan(a)[0])
               for a in [-math.pi / 2 + index * math.pi / 200.0 for index in range(201)]]
    body.append(path(quarter, VIOLET, 2.2))
    body.append(dot(centre_x, centre_y, INK, 3))
    body.append(label(centre_x, centre_y + 18, 'left² + right² = 1', VIOLET,
                      anchor='middle'))
    body.append(label(centre_x, 106, 'a quarter circle, so panning', LABEL,
                      anchor='middle'))
    body.append(label(centre_x, 124, 'never changes the loudness', LABEL,
                      anchor='middle'))
    body.append('</g>')
    return svg(576, 390, 'Equal-power stereo panning', body)


def emitter_types() -> str:
    """Global against positional: which one moves when the listener does."""
    body = ['<g>']

    for left, title, moving in ((16, 'global', False), (296, 'positional', True)):
        centre_x, centre_y = left + 132, 150
        body.append('<rect x="%d" y="24" width="264" height="250" rx="8" fill="none" '
                    'stroke="%s" stroke-width="1" stroke-dasharray="3 5"/>'
                    % (left, INK))
        body.append(label(centre_x, 46, title, LABEL, anchor='middle', size=14,
                          weight='600'))
        # The listener, at two moments.
        for offset, alpha in ((-58, '1'), (58, '0.45')):
            body.append('<g opacity="%s">' % (alpha,))
            body.append(dot(centre_x + offset, centre_y + 54, BLUE, 6))
            body.append(label(centre_x + offset, centre_y + 76, 'listener', BLUE,
                              anchor='middle'))
            body.append('</g>')
        body.append(line(centre_x - 44, centre_y + 54, centre_x + 44, centre_y + 54,
                         BLUE, 1.0, dash='3 4'))

        if moving:
            body.append(dot(centre_x + 66, centre_y - 46, RED, 6))
            body.append(label(centre_x + 66, centre_y - 58, 'emitter', RED,
                              anchor='middle'))
            for offset in (-58, 58):
                body.append(line(centre_x + offset, centre_y + 54,
                                 centre_x + 66, centre_y - 46, RED, 1.0, dash='2 4'))
            body.append(label(centre_x, centre_y + 100,
                              'fixed in the world; the distance,', LABEL,
                              anchor='middle'))
            body.append(label(centre_x, centre_y + 118,
                              'cone and pan all change', LABEL, anchor='middle'))
        else:
            for offset in (-58, 58):
                body.append(dot(centre_x + offset, centre_y - 46, RED, 6))
            body.append(label(centre_x, centre_y - 66, 'emitter', RED, anchor='middle'))
            body.append(label(centre_x, centre_y + 100,
                              'fixed to the listener; heard the', LABEL,
                              anchor='middle'))
            body.append(label(centre_x, centre_y + 118,
                              'same wherever they stand', LABEL, anchor='middle'))

    body.append(label(288, 300,
                      'A global emitter has no PositionalProperties at all — the '
                      'extension forbids them,', LABEL, anchor='middle'))
    body.append(label(288, 318,
                      'and a scene (which has no transform) may carry only this kind.',
                      LABEL, anchor='middle'))
    body.append('</g>')
    return svg(576, 336, 'Global and positional emitters', body)


DIAGRAMS = {
    'distance-models.svg': distance_models,
    'cone.svg': cone,
    'ellipsoids.svg': ellipsoids,
    'pan.svg': pan,
    'emitter-types.svg': emitter_types,
}


def write(directory: pathlib.Path) -> list[pathlib.Path]:
    """Draw every diagram into ``directory``; return what was written."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, draw in sorted(DIAGRAMS.items()):
        target = directory / name
        target.write_text(draw(), encoding='utf-8')
        written.append(target)
    return written


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    for target in write(here / 'images'):
        print(target.relative_to(here.parent))


if __name__ == '__main__':
    main()
