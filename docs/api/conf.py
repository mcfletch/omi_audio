"""Sphinx configuration for the rendered API reference.

The docstrings are the best asset this package has, and until this existed
nothing rendered them: they are written in Sphinx style -- ``:class:``,
``:func:``, ``:mod:``, ``Raises:`` blocks -- which reads as noise on GitHub and
as prose here.

Build it with tox, which pins the dependencies::

    tox -e docs

or directly, from a checkout with ``omi_audio[docs]`` installed::

    sphinx-build -W -b html docs/api docs/_build/html
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src'))

from omi_audio import __version__                                  # noqa: E402

project = 'omi_audio'
author = 'Mike C. Fletcher'
copyright = '2026, Mike C. Fletcher'                               # noqa: A001
release = __version__
version = __version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',          # the `Raises:` and `References:` blocks
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# The narrative documentation lives beside this directory as Markdown, and is
# linked from the index rather than duplicated into reStructuredText.
exclude_patterns = ['_build']

autodoc_member_order = 'bysource'   # the order the module tells its own story in
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = 'furo'
html_title = 'omi_audio %s' % (release,)
html_static_path = []

# A warning is a broken cross-reference, and a broken cross-reference in an API
# reference is a lie.  `tox -e docs` runs with -W, so they fail the build.
nitpicky = False
