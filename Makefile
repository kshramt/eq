# Constants
DIR := $(abspath .)
DEPS := kshramt_py


MY_PYTHON ?= python3.3
MY_RUBY ?= ruby2.0
MY_PYFLAKES ?= pyflakes-3.3

# Configurations
.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:
export SHELL := /bin/bash
export SHELLOPTS := pipefail:errexit:nounset:noclobber


# Tasks

.PHONY: all deps test test_coverage build
all: deps
deps: $(DEPS:%=dep/%.updated) eq/kshramt.py

test: deps
	for f in $$(git ls-files **/*.py)
	do
	   echo
	   echo "$$f"
	   coverage run -a "$$f"
	   $(MY_PYFLAKES) "$$f"
	done

test_coverage: test
	coverage html

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

eq/kshramt.py: dep/kshramt_py/kshramt.py
	mkdir -p $(@D)
	cp -a $< $@

# Rules

define DEPS_RULE_TEMPLATE =
dep/$(1)/%: | dep/$(1).updated ;
endef
$(foreach f,$(DEPS),$(eval $(call DEPS_RULE_TEMPLATE,$(f))))

dep/%.updated: config/dep/%.ref dep/%.synced
	cd $(@D)/$*
	git fetch origin
	git merge "$$(cat ../../$<)"
	cd -
	if [[ -r dep/$*/Makefile ]]; then
	   $(MAKE) -C dep/$*
	fi
	touch $@

dep/%.synced: config/dep/%.uri | dep/%
	cd $(@D)/$*
	git remote rm origin
	git remote add origin "$$(cat ../../$<)"
	cd -
	touch $@

$(DEPS:%=dep/%): dep/%:
	git init $@
	cd $@
	git remote add origin "$$(cat ../../config/dep/$*.uri)"
