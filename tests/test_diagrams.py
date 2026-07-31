"""The documentation's diagrams are generated, and must stay that way.

A picture of a gain curve is a claim about the code.  Redrawing one by hand is
how a claim goes quietly wrong the first time somebody edits a formula, so
``docs/make_diagrams.py`` plots every curve by calling the same functions the
mixer calls -- and these tests fail if the committed files stop matching what
that script produces.

Whoever changes :mod:`omi_audio.spatial` therefore has exactly one extra step,
and the failure tells them what it is::

    python docs/make_diagrams.py
"""

import importlib.util
import pathlib
import re
import xml.etree.ElementTree as ElementTree

import pytest

DOCS = pathlib.Path(__file__).resolve().parent.parent / 'docs'
IMAGES = DOCS / 'images'


def _generator():
    """``docs/make_diagrams.py``, imported by path: it is a build step, not a module."""
    spec = importlib.util.spec_from_file_location('make_diagrams',
                                                  DOCS / 'make_diagrams.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_diagrams = _generator()
NAMES = sorted(make_diagrams.DIAGRAMS)


@pytest.mark.parametrize('name', NAMES)
class TestEveryDiagram:
    def test_the_committed_file_is_what_the_script_produces(self, name):
        """If this fails, the picture and the code disagree about the curve."""
        expected = make_diagrams.DIAGRAMS[name]()
        assert (IMAGES / name).read_text(encoding='utf-8') == expected, (
            '%s is out of date; run: python docs/make_diagrams.py' % (name,))

    def test_it_is_well_formed_xml(self, name):
        """A browser is forgiving and a screen reader is not."""
        ElementTree.fromstring((IMAGES / name).read_text(encoding='utf-8'))

    def test_it_carries_a_title_for_a_screen_reader(self, name):
        root = ElementTree.fromstring((IMAGES / name).read_text(encoding='utf-8'))
        assert root.get('aria-label')
        assert root.find('{http://www.w3.org/2000/svg}title') is not None

    def test_nothing_is_drawn_outside_the_canvas(self, name):
        """Off-canvas geometry is the failure a generated diagram makes, and
        the only one nobody notices until the page is looked at."""
        text = (IMAGES / name).read_text(encoding='utf-8')
        root = ElementTree.fromstring(text)
        width, height = (float(value) for value in root.get('viewBox').split()[2:])
        outside = [point for point in _points(root)
                   if not (-4 <= point[0] <= width + 4 and -4 <= point[1] <= height + 4)]
        assert outside == [], '%s draws %d points off the canvas' % (name, len(outside))

    def test_no_curve_is_hidden_underneath_another(self, name):
        """Two curves on the same path, both solid, means one is invisible.

        It happens for real reasons rather than by carelessness: at
        ``rolloffFactor`` 1 the ``inverse`` and ``exponential`` distance models
        are numerically identical, so plotting both drew a legend with three
        entries over a picture with two lines.  Coincident geometry is fine --
        the coincidence is often the point -- but it has to be *visible*, which
        means the strokes must differ in more than colour.
        """
        root = ElementTree.fromstring((IMAGES / name).read_text(encoding='utf-8'))
        curves = [element for element in root.iter()
                  if element.tag.endswith('}path') and element.get('fill') == 'none']
        for index, first in enumerate(curves):
            for second in curves[index + 1:]:
                if first.get('d') != second.get('d'):
                    continue
                assert first.get('stroke-dasharray') != second.get('stroke-dasharray'), (
                    '%s draws two identical curves in the same style; the one '
                    'underneath cannot be seen' % (name,))

    def test_it_is_referenced_from_the_documentation(self, name):
        """A diagram nobody links to is a diagram nobody sees."""
        pages = [page.read_text(encoding='utf-8') for page in DOCS.glob('*.md')]
        assert any(name in page for page in pages), '%s is not shown anywhere' % (name,)


def _points(root):
    """Every coordinate the document draws at, in SVG user units."""
    found = []
    for element in root.iter():
        drawn = element.get('d')
        if drawn:
            found += [(float(x), float(y)) for x, y in
                      re.findall(r'[ML]\s*(-?[\d.]+)\s+(-?[\d.]+)', drawn)]
        for horizontal, vertical in (('x1', 'y1'), ('x2', 'y2'), ('cx', 'cy'),
                                     ('x', 'y')):
            if element.get(horizontal) is not None and element.get(vertical) is not None:
                found.append((float(element.get(horizontal)),
                              float(element.get(vertical))))
    return found


def test_every_generated_file_is_committed():
    """The script and the directory must hold the same set of names."""
    assert sorted(path.name for path in IMAGES.glob('*.svg')) == NAMES
