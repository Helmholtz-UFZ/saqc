# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

# clean sphinx generated stuff
clean:
	rm -rf _build _static _api
	rm -f *.automodsumm
	rm -f func_modules/*.automodsumm
	rm -f intro_modules/*.automodsumm
	mkdir _static

# trigger (saqc) customized documentation pipeline
doc:
	# generate fake modules to b documented by sphinx
	python make_doc_module.py -p "saqc/funcs" -t "docs/intro_modules" -sr ".." -mo "intro_doc"
	python make_doc_module.py -p "saqc/funcs" -t "docs/func_modules" -sr ".." -mo "registered_doc"
	# make rest files from fake modules
	python make_doc_rst.py -p "docs/intro_modules" -t "sphinx-doc/intro_modules" -sr ".."
	python make_doc_rst.py -p "docs/func_modules" -t "sphinx-doc/func_modules" -sr ".."
	# make rest folders from markdown folders
	python make_md_to_rst.py -p "sphinx-doc/getting_started_md" -sr ".."
	python make_md_to_rst.py -p "sphinx-doc/how_to_doc_md" -sr ".."
	# make the html build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	# format docstring module domain strings to correct adresses
	python make_html_headings_proppa.py -b "sphinx-doc/_build/html/_api" -p "docs/func_modules" -sr ".."
	python make_html_headings_proppa.py -b "sphinx-doc/_build/html/_api" -p "docs/intro_modules" -sr ".."
    # clear fake modules/intermediate rest files
#	rm -r ../docs
	rm -r getting_started_md_m2r
	rm -r how_to_doc_md_m2r

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

