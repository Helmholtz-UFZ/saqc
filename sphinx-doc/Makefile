# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# sphinx build controll
FUNCTIONS     = Functions
# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

# clean sphinx generated stuff
clean:
	rm -rf _build _static
	rm -f *.automodsumm
	rm -f func_modules/*.automodsumm
	rm -f intro_modules/*.automodsumm
	rm -rf ../$(FUNCTIONS)
	mkdir _static
	rm -r getting_started_md_m2r how_to_doc_md_m2r

# trigger (saqc) customized documentation pipeline
doc:
	# generate parent fake module for the functions to be documented
	python make_doc_module.py -p "saqc/funcs" -t "$(FUNCTIONS)" -sr ".."
	# make rest files from fake module
	python make_doc_rst.py -p "$(FUNCTIONS)" -t "sphinx-doc/$(FUNCTIONS)" -sr ".."
	# make rest folders from markdown folders
	python make_md_to_rst.py -p "sphinx-doc/getting_started_md" -sr ".."
	python make_md_to_rst.py -p "sphinx-doc/how_to_doc_md" -sr ".."
	# make the html build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
    # clear fake modules/intermediate rest files
	#rm -r getting_started_md_m2r
	#rm -r how_to_doc_md_m2r

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

