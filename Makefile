# Constants
MY_PYTHON ?= python3.3
MY_RUBY ?= ruby2.0
MY_PYFLAKES ?= pyflakes-3.3

# Configurations
.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:
.SECONDARY:
export SHELL := /bin/bash
export SHELLOPTS := pipefail:errexit:nounset:noclobber

# Tasks
.PHONY: all test test_coverage up_prepare up

all: test

test:
	for f in $$(git ls-files **/*.py)
	do
	   echo
	   echo "$$f"
	   coverage run -a "$$f"
	   $(MY_PYFLAKES) "$$f"
	done

test_coverage: test
	coverage html

up: up_prepare
	cd ../pypi/eq
	$(MY_PYTHON) setup.py register sdist upload
	rm -f ~/.pypirc

up_prepare:
	cd ../pypi/eq
	git pull origin master

# Files

# Rules
