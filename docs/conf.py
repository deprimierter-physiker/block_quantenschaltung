# Configuration file for the Sphinx documentation builder.
#
# NOTE: the package lives under src/, which is not on sys.path when Sphinx runs from
# docs/. autodoc imports the module to read its docstrings, so point it there.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'Block_quantenschaltung_pascal_marius'
copyright = '2026, Pascal and Marius'
author = 'Pascal and Marius'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autosummary','sphinx.ext.napoleon']

# -- autosummary / autodoc ---------------------------------------------------
# autosummary_generate makes Sphinx write the per-object stub pages into
# docs/_autosummary (already listed in .gitignore) instead of requiring them by hand.
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'   # render the annotations into the parameter list

# -- Napoleon (Google-style docstrings) --------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



# Include notebooks
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

nbsphinx_execute = 'never'