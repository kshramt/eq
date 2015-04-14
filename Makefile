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

.PHONY: all deps check build
all: deps
deps: $(DEPS:%=dep/%.updated) eq/kshramt.py

check: deps $(PYTHON_TESTED_FILES)

build: deps
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


# Files

eq/kshramt.py: $(call sha256,dep/kshramt_py/kshramt.py)
	mkdir -p $(@D)
	cat $(call unsha256,$<) >| $@

# Rules

%.py.tested: %.py.sha256
	$(PYFLAKES) $(call unsha256,$<)
	$(PYTHON) $(call unsha256,$<)
	touch $@


define DEPS_RULE_TEMPLATE =
dep/$(1)/%.sha256.new: dep/$(1)/% | dep/$(1).updated
	sha256sum $$< >| $$@
endef
$(foreach f,$(DEPS),$(eval $(call DEPS_RULE_TEMPLATE,$(f))))

dep/%.updated: config/dep/%.ref.sha256 dep/%.synced
	cd $(@D)/$*
	git fetch origin
	git merge "$$(cat ../../$(call unsha256,$<))"
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
