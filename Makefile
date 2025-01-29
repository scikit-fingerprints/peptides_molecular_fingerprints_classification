.PHONY: setup

setup: ## Install development dependencies, pre-commit hooks and poetry plugin
	# check if poetry is installed
	poetry --version >/dev/null 2>&1 || (echo "Poetry is not installed. Please install it from https://python-poetry.org/docs/#installation" && exit 1)
	poetry sync --with dev
	poetry run pre-commit install
