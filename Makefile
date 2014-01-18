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
.PHONY: all test test_coverage

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

# Files

# Rules
