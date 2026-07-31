"""The demo has to keep running, or it is documentation that lies.

``examples/orbit.py`` is the first thing most evaluators will run and the
backbone of the game-integration guide, so it is held to the same standard as
the rest: it is driven here, briefly, on a silent device.
"""

import importlib.util
import pathlib

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / 'examples'


def _example(name):
    """An example script, imported by path: they are programs, not a package."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / ('%s.py' % (name,)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


orbit = _example('orbit')


class TestOrbit:
    def test_it_runs_to_completion_with_no_device_at_all(self, capsys):
        assert orbit.main(['--silent', '--seconds', '0.15', '--rate', '8000']) == 0

    def test_it_reports_the_output_it_chose(self, capsys):
        orbit.main(['--silent', '--seconds', '0.05', '--rate', '8000'])
        assert 'audio: silent' in capsys.readouterr().out

    def test_the_sound_it_describes_is_actually_audible(self, capsys):
        """The demo prints a level; a demo that printed silence would be worse
        than no demo, because it would look like the library not working."""
        orbit.main(['--silent', '--seconds', '0.05', '--rate', '8000'])
        printed = capsys.readouterr().out
        peak = float(printed.rsplit('peaks at ', 1)[1].split()[0])
        assert peak > 0.05, 'the demo mixed silence'

    def test_the_emitter_it_builds_is_a_valid_positional_one(self):
        emitter = orbit.emitter()
        assert emitter.positional_audio is True
        assert emitter.positional.refDistance > 0.0

    @pytest.mark.parametrize('left,right,expect_left,expect_right', [
        (1.0, 0.0, 24, 0),
        (0.0, 1.0, 0, 24),
        (0.5, 0.5, 12, 12),
    ])
    def test_the_meter_shows_which_ear_the_sound_is_in(self, left, right,
                                                       expect_left, expect_right):
        shown = orbit.bar(left, right)
        near, far = shown.split('|')
        assert (near.count('#'), far.count('#')) == (expect_left, expect_right)
