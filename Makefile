# Constants
DIR := $(abspath .)
DEPS := kshramt_py


MY_PYTHON ?= python3.3
PYTHON := $(MY_PYTHON)
MY_PYFLAKES ?= pyflakes-3.3
PYFLAKES := $(MY_PYFLAKES)

PYTHON_FILES := $(shell git ls-files '**/*.py')
PYTHON_TESTED_FILES := $(addsuffix .tested,$(PYTHON_FILES))

# Configurations
.SUFFIXES:
.DELETE_ON_ERROR:
.PRECIOUS: %.sha256 %.sha256.new
.ONESHELL:
export SHELL := /bin/bash
export SHELLOPTS := pipefail:errexit:nounset:noclobber


sha256 = $(1:%=%.sha256)
unsha256 = $(1:%.sha256=%)


# Tasks

.PHONY: download-deps arrange-deps all all-impl check check-impl build build-impl


define INTERFACE_TARGET_TEMPLATE =
$(1):
	$$(MAKE) download-deps
	$$(MAKE) $(1)-impl
endef
$(foreach f,all check build,$(eval $(call INTERFACE_TARGET_TEMPLATE,$(f))))


all-impl: arrange-deps


check-impl: arrange-deps $(PYTHON_TESTED_FILES)


build-impl: arrange-deps
	readonly tmp_dir="$$(mktemp -d)"
	git ls-files | xargs -I{} echo cp --parents ./{} "$$tmp_dir"
	git ls-files | xargs -I{} cp --parents ./{} "$$tmp_dir"
	mkdir -p "$${tmp_dir}"/eq
	cp eq/kshramt.py "$${tmp_dir}"/eq
	cd "$$tmp_dir"
	$(MY_PYTHON) setup.py sdist
	mkdir -p dist
	mv -f dist/* $(DIR)/dist/
	rm -fr "$${tmp_dir}"


arrange-deps: eq/kshramt.py
download-deps: $(DEPS:%=dep/%.updated)


# Files

eq/kshramt.py: $(call sha256,dep/kshramt_py/kshramt.py)
	mkdir -p $(@D)
	cat $(call unsha256,$<) >| $@

# Rules

%.py.tested: %.py.sha256
	$(PYFLAKES) $(call unsha256,$<)
	$(PYTHON) $(call unsha256,$<)
	touch $@


dep/%.updated: config/dep/%.ref.sha256 dep/%.synced
	cd $(@D)/$*
	git fetch origin
	git checkout "$$(cat ../../$(call unsha256,$<))"
	cd -
	if [[ -r dep/$*/Makefile ]]; then
	   $(MAKE) -C dep/$*
	fi
	touch $@

dep/%.synced: config/dep/%.uri.sha256 | dep/%
	cd $(@D)/$*
	git remote rm origin
	git remote add origin "$$(cat ../../$(call unsha256,$<))"
	cd -
	touch $@

$(DEPS:%=dep/%): dep/%:
	git init $@
	cd $@
	git remote add origin "$$(cat ../../config/dep/$*.uri)"


%.sha256.new: %
	sha256sum $< >| $@


%.sha256: %.sha256.new
	cmp -s $< $@ || cat $< >| $@
