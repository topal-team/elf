# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = "ELF"
copyright = "2025, ELF developers"
author = "Adrien Aguila--Multner, Julia Gusak"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	"sphinx.ext.autodoc",
	"sphinx.ext.autosummary",
	"sphinx.ext.napoleon",
	"sphinx.ext.viewcode",
	"sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_default_options = {
	"members": True,
	"undoc-members": True,
	"show-inheritance": True,
	"member-order": "bysource",
	"special-members": "__init__",
}

# Show enum values and their docstrings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Show __init__ docstring under class
autoclass_content = "both"

# Napoleon settings for better docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
	"python": ("https://docs.python.org/3", None),
	"torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"
html_theme_options = {
	"navigation_depth": 4,
	"collapse_navigation": False,
	"sticky_navigation": True,
	"includehidden": True,
	"titles_only": False,
	"style_external_links": True,
	"prev_next_buttons_location": "bottom",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["theme-toggle.js"]

# Add custom sidebar (RTD theme specific)
# html_sidebars = {"**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]}

# Modern HTML options
html_show_sphinx = False
html_show_copyright = True

sys.path.insert(0, os.path.abspath(".."))
