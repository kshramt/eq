# Constants
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
.PHONY: all test

all: test

test:
	for f in $$(git ls-files **/*.py)
	do
	   coverage run -a $$f
	done
	coverage html

	for f in $$(git ls-files **/*.py)
	do
	   $(MY_PYFLAKES) $$f
	done

# Files

# Rules
