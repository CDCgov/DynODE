# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

import tomli

sys.path.insert(0, os.path.abspath("../../src"))

with open("../../pyproject.toml", "rb") as file:
    pyproject_data = tomli.load(file)

project = "DynODE"
copyright = pyproject_data["tool"]["poetry"]["license"]
author = "".join(pyproject_data["tool"]["poetry"]["authors"])
release = pyproject_data["tool"]["poetry"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
    "myst_parser",
]
autosummary_generate = True
intersphinx_mapping = {
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
# exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/CDCgov/DynODE",
    "use_repository_button": True,
    "show_navbar_depth": 2,
}
html_static_path = ["_static"]
html_css_files = [
    "style.css",
]
