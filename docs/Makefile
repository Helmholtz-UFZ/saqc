# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
	for k in $(MDLIST); do echo docs/"$$k"; done

# clean sphinx generated stuff
clean:
	rm -rf _build _static
	mkdir _static
	rm -f *.automodsumm
	rm -f *.automodapi
	rm -f modules/*.automodsumm
	rm -f modules/*.automodapi
	rm -f */*.automodsumm
	rm -f -r coredoc

# make doctest, make documentation, make clean
doc:
	# generate environment table from dictionary
	python scripts/make_env_tab.py
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -f *.automodsumm
	rm -f *.automodapi
	rm -f modules/*.automodsumm
	rm -f modules/*.automodapi
	rm -f */*.automodsumm

# make documentation
doconly:
	# generate environment table from dictionary
	python scripts/make_env_tab.py
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# make test, clean up
testonly:
    # generate parent fake module for the functions to be documented
	python scripts/make_env_tab.py
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -f *.automodsumm
	rm -f *.automodapi
	rm -f modules/*.automodsumm
	rm -f modules/*.automodapi
	rm -f */*.automodsumm
	rm -f -r coredoc

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
