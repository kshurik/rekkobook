pylint:
	pylint data_prep models utils

isort:
	isort data_prep models utils --jobs=0

black:
	black data_prep models utils

flake:
	flake8 data_prep models utils

fmt: isort black

clean_cache: ## Remove __pycache__ folders
	@find . | grep __pycache__ | xargs rm -rf