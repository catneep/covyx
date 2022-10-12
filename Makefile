SHELL := bash
PATH := ./venv/bin:${PATH}
PYTHON = python3

run: clean venv install

.PHONY: venv
venv:
	test -d venv || $(PYTHON) -m venv venv

.PHONY: install
install: venv
	source activate && pip install -q -r requirements.txt

clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
