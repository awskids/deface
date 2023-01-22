help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	@rm -f .coverage
	@rm -fr coverage-reports/
	@rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 deface tests

format:
	black deface tests

test:
	PYTHONPATH=".:deface/:tests/" pytest -s -rPx tests/test_spike.py

version := $(shell cat VERSION)

latest-distro := dist/topicanalysis-$(version).tar.gz

build-debug:
	echo "Product Version $(version)"
	echo "? $(latest-distro)"

build:
	rm -fr dist
	python -m build

build-latest:
	rm -fr dist
	python -m build -s
	cp $(latest-distro) dist/topicanalysis-latest.tar.gz