# What comes after ML model development?
This chapter describes what is production code and how to write it using compelementary tools
such `poetry`, `Makefile` and code styling modules like `pylint`, `black`, `isort`, `flake8`.
Finally, we will define architecture of our production microservice of RecSys we developed in [Chapter 1](https://rekkobook.com/chapter1/full_pipeline.html)


In any Machine Learning you will come across writing a production procedure. What is that?
Production code refers to the code that is used to run a machine learning model in a real-world environment.
This code is often written in a programming language such as Python or Java and is executed on a server
or virtual machine that is accessible to end-users. Thus, to get our model from notebook to users
we need to write production code to use it in production environment.


The purpose of production code is to ensure that a machine learning model is able to perform
as expected in a production environment. This includes handling real-world data, scaling to meet
demand, and providing accurate and timely predictions. Writing production code for machine learning
requires a different set of skills than writing code for research or experimentation. In production,
code must be robust, scalable, and reliable.


This pipeline include several steps such as:
- Replicate development environment for full reproducibility of results: python and library versions;
- Data preparation: collect necessary input, calculate runtime features;
- Inference & postprocessing: make final predictions, do any postprocessing of predictions if necessary (scaling, etc.)

In addition, strong coding requirements must be met which can be tracked and fixed using:
- Use a consistent coding style and naming convention
- Write clear and concise comments
- Include error handling and logging
- Use version control to track changes and collaborate with others
- Write unit tests to ensure that the code works as expected

Therefore, various tools beside model and python code is used. Here, let's make an overview of these tools
that used in development today.


# Developement tools
## Poetry
The first tool to use when creating production service or even any ML project to have full reproducibility
of the results is library version control. Previously, `requirements.txt` was very popular amid developers
and Data Scientists. It is pretty straightforward in usage -- add name and version of the library with a new line

```
pandas==1.1.5
numpy==1.23.5
etc.
```
However, more advanced config management has been discovered [`poetry`](https://python-poetry.org/) which allows
to use it for dependencies management and configuration file for popular devtools (we will discuss it later).
The main features are:
- Dependecies management - resolves all dependencies for each library in the correct order. When using requirements.txt,
developers have to manually manage the installation and updating of dependencies. This can be time-consuming
and error-prone, especially when dealing with complex projects that have multiple dependencies. In contrast,
Poetry automates most of the dependency management process, making it easier to install, update, and remove dependencies.;
- Isolated virtualenv - automatically creates virtual environment for reproducibility with given Python version
and libraries (no need to execute by hand, just one line of command in CLI);
- CLI interaction to manage configuration file - intuitive commands to add / remove / install dependencies;
- BONUS: some paramters for `pylint` styling guide, etc. can be set within the same config!


Now, let's discuss details about how to use it. To initialize config we need to run installation
```
curl -sSL https://install.python-poetry.org | python3 -
```

Then, we can initialize poetry config by executing (assuming we want to create dependency management for pre-existing project)
```
poetry init
```
After we run, several optional questions will be asked to create the `pyproject.toml` file -- main configuration file.
You can either fill it (for instacem set python version) or just press enter to use suggested default values.

Next, we add package names as following
```
poetry add pandas
```
or speicific version
```
poetry add pandas==1.5.1
```
Further it creates virtual environment with all packages and you can use it to run your scripts / notebooks
or run by using bash commands
```
poetry run python inference.py
```

Also, you can set various parameters by utilizing full power of TOML extensions. In below there is an example of 
a full poetry set up generated for the illustration purpose.

- `[tool.poetry]` - meta information about the project
- `[[tool.poetry.dependencies]]` - main dependencies used in production
- `[tool.poetry.group.dev.dependencies]` - development packages only needed for test and code style guides
- `[build-system]` - poetry system parameters
- `[tool.isort], [tool.black], [tool.pylint]` - parameters to set up custom code style and checks.
We will discuss them in the next part.

```
[tool.poetry]
name = "rekko_handbook"
version = "0.1.0"
description = "RecSys Handbook"
authors = ["khalilbekov92@gmail.com"]
readme = "README.md"
packages = [{include = "rekko_handbook"}]

[tool.poetry.dependencies]
python = ">=3.9,<3.12"
loguru = "0.6.0"
pandas = "1.5.3"
presto-python-client = "0.8.3"
clickhouse-driver = "0.2.5"
dynaconf = "^3.1.12"
torch = "^2.0.0"
transformers = "^4.27.3"
tqdm = "^4.65.0"
scipy = "^1.10.1"
scikit-learn = "^1.2.2"
boto3 = "^1.26.106"
python-dotenv = "^1.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.2.2"
pylint = "^2.17.1"
flake8 = "^6.0.0"
black = "^23.1.0"
isort = "^5.12.0"
pytest-html = "^3.2.0"

[tool.isort]
line_length = 120
multi_line_output = 3
include_trailing_comma = true

[tool.black]
line-length = 120
target-version = ['py39']
skip-string-normalization = true

[build-system]
requires = ["poetry-core>=1.4.1"]
build-backend = "poetry.core.masonry.api"

[tool.pylint.'FORMAT']
min-similarity-lines = 10
fail-under = 9.7
py-version = 3.9
good-names=[
    'bp',
    'db',
    'df'
]
max-line-length = 120
disable = [
    'locally-disabled', 'suppressed-message',
    'missing-module-docstring', 'missing-class-docstring',
    'missing-function-docstring', 'too-few-public-methods',
    'wrong-import-position', 'import-outside-toplevel',
    'fixme', 'too-many-locals', 'too-many-arguments',
    'too-many-instance-attributes', 'c-extension-no-member'
]
```

Overall, poetry provides easy and efficient way to reproduce development environment,
resolve dependencies and gives opportunity to use customizaation in some devtools.

## Styling guide and code quality
TBD


## Makefile
TBD


