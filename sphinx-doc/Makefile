# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# sphinx build controll:
# Function documentations will be acessible through: FUNCTIONS.saqc
FUNCTIONS     = Functions
CORE          = Core
# List of folders containing markdown content to include into the build
MDLIST		  = getting_started_md how_to_doc_md cook_books_md misc_md
# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

test:
	for k in $(MDLIST); do echo sphinx-doc/"$$k"; done

# clean sphinx generated stuff
clean:
	rm -rf _build _static
	rm -rf ../$(FUNCTIONS)
	rm -rf ../$(CORE)
	mkdir _static
	# for k in $(MDLIST); do rm -r "$$k"_m2r; done
	rm -f *.automodsumm
	rm -f *.automodapi
	rm -f moduleAPIs/*.automodsumm
	rm -f moduleAPIs/*.automodapi

# trigger (saqc) customized documentation pipeline
doc:
	# generate parent fake module for the functions to be documented
	python scripts/make_doc_module.py -p "saqc/funcs" -t "$(FUNCTIONS)" -sr ".."
	# generate environment table from dictionary
	python scripts/make_env_tab.py
	# make rest folders from markdown folders
	# for k in $(MDLIST); do python scripts/make_md_to_rst.py -p sphinx-doc/"$$k" -sr ".."; done
	# make the html build
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

