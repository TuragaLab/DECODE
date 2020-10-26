# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# import pytorch_sphinx_theme
import sphinx_rtd_theme

# import guzzle_sphinx_theme
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import decode

project = 'DECODE'
copyright = '2020, Artur Speiser, Lucas-Raphael Mueller et al.'
author = 'Artur Speiser, Lucas-Raphael Mueller et al.'

# The full version, including alpha/beta/rc tags
release = '0.9.a'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'recommonmark',
]

# napoleon
napoleon_use_param = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# map file extension to respective types
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------
# Adds an HTML table visitor to apply Bootstrap table classes
# html_translator_class = 'guzzle_sphinx_theme.HTMLTranslator'
# html_theme_path = guzzle_sphinx_theme.html_theme_path()
# html_theme = 'guzzle_sphinx_theme'

# Register the theme as an extension to generate a sitemap.xml
# extensions.append("guzzle_sphinx_theme")


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_rtd_theme"
# html_theme_path = ["/Users/lucasmueller/Repositories/pytorch_sphinx_theme"]

html_theme_options = {
    # Set the name of the project to appear in the sidebar
    "project_nav_name": "DECODE",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# which class content to include
autoclass_content = 'init'  # init
