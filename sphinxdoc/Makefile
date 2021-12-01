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

test:
	for k in $(MDLIST); do echo sphinxdoc/"$$k"; done

# clean sphinx generated stuff
clean:
	rm -rf _build _static
	mkdir _static
	rm -f *.automodsumm
	rm -f *.automodapi
	rm -f moduleAPIs/*.automodsumm
	rm -f moduleAPIs/*.automodapi
	rm -f */*.automodsumm

# trigger (saqc) customized documentation pipeline
doc:
	# generate parent fake module for the functions to be documented
	python scripts/make_doc_module.py -p "saqc/funcs" -sr ".."
	# generate environment table from dictionary
	python scripts/make_env_tab.py
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

