# What comes after ML model development?
This chapter describes what is production code and how to write it using compelementary tools
such `poetry`, `Makefile` and code styling modules like `pylint`, `black`, `isort`, `flake8`.
Finally, we will define architecture of our production microservice of RecSys we developed in [Chapter 1](https://rekkobook.com/chapter1/full_pipeline.html)


In any Machine Learning you will come across writing a production procedure. What is that?
Production code refers to the code that is used to run a machine learning model in a real-world environment.
This code is often written in a programming language such as Python and is executed on a server
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
You can either fill it (for instance set python version) or just press enter to use suggested default values.

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
- `[tool.poetry.dependencies]` - main dependencies used in production
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
resolve dependencies and gives opportunity to use customization in some devtools.

## Styling guide and code quality
As we have mentioned earlier, production code must be robust, scalable & reliable. Tracking
the quality of the code by reviews only is exhaustive and almost impossible to achieve -- we
are prone to errors too. However, most common mistakes like typos and convieniet bugs can
be checked automatically and fixed. Here, we will describe main Python tools to achieve that
- `[pylint](https://pypi.org/project/pylint/)` - code analyzer which does not run your code.
It checks for errors, coding standard, makes clear sugesstions on how to improve it
and even grade it on the scale of 0 to 10;

```{image} ./img/pylint.png
:alt: fishy
:class: bg-primary mb-1
:width: 400px
:align: center
```

- `[black](https://pypi.org/project/black/)` - in Python community there is well-known
PEP-8 standard to follow. This library is a code formatter which is PEP 8 compliant
opinionated formatter. It formats all code in given directory / files inplace fast and efficiently;

Before formatting with black
```{code-cell} ipython3
import pandas as pd
import numpy as np

def some_kaif_function(input_df: pd.DataFrame, param_1: int, param_2: str, path: str):
    """
    some example function
    """
    input_df['check_bool'] = input_df.loc[(input_df[param_2] >= param_1) & (input_df[param_2] < param_1 - 1)]

    return input_df
```

After formatting with black
```{code-cell} ipython3
import pandas as pd
import numpy as np

def some_kaif_function(input_df: pd.DataFrame, param_1: int, param_2: str, path: str):
    """
    some example function
    """
    input_df["check_bool"] = input_df.loc[
        (input_df[param_2] >= param_1) & (input_df[param_2] < param_1 - 1)
    ]
    
    return input_df

```

- `[isort](https://pycqa.github.io/isort/)` - a library to make appropiate imports:
alphabetical order and group by types to sections. Below is the example from official homepage

Before the isort
```{code-cell} ipython3
from my_lib import Object

import os

from my_lib import Object3

from my_lib import Object2

import sys

from third_party import lib15, lib1, lib2, lib3, lib4, lib5, lib6, lib7, lib8, lib9, lib10, lib11, lib12, lib13, lib14

import sys

from __future__ import absolute_import

from third_party import lib3

print("Hey")
print("yo")
```

After isort
```{code-cell} ipython3
from __future__ import absolute_import

import os
import sys

from third_party import (lib1, lib2, lib3, lib4, lib5, lib6, lib7, lib8,
                         lib9, lib10, lib11, lib12, lib13, lib14, lib15)

from my_lib import Object, Object2, Object3

print("Hey")
print("yo")
```

## Makefile
Makefiles are important tools for managing and automating the building and
deployment of software projects. In Python development, Makefiles are especially
useful in managing the compilation, packaging, and testing of code. A Makefile
is essentially a script that defines a set of rules and actions to be performed
when certain conditions are met. These rules can be used to automate repetitive
tasks, such as compiling code, running tests, and generating documentation.

They are particularly useful for large projects with many files and dependencies,
as they allow developers to quickly and easily build, test, and deploy their code
without having to manually type in commands for each step.

For example, let's say you have a Python project with multiple modules, each with
their own dependencies. You would need to install these dependencies and set up
the environment in order to run the project. With a Makefile, you can automate this
process by defining targets for each step of the build,
such as "install dependencies", "set up environment", "run tests", etc. 

Here's an example Makefile for a Python project:

```
# install dependencies
install:
    pip install -r requirements.txt

# set up environment
setup:
    virtualenv env
    source env/bin/activate

# run tests
test:
    python -m unittest discover -s tests

# clean up environment
clean:
    rm -rf env/
    @find . | grep __pycache__ | xargs rm -rf

# let's add formatiing stuff here as well

pylint:
	pylint app config tests # folder names

isort:
	isort app config tests --jobs=0

black:
	black app config tests

fmt: isort black
```

This Makefile defines four targets: `install`, `setup`, `test`, and `clean`.
- `make install` will install all the dependencies listed in `requirements.txt`;
- `make setup` will create a virtual environment and activate it;
- `make test` will run all the tests in the `tests` directory;
- `make clean` will remove the virtual environment and any cached Python files;
- `make pylint` will run code formatting check;
- `make isort` will do proper import of modules;
- `make black` will run formatting inplace;
- `make fmt` will call `make isort` & `make black` for full formatting

Overall, Makefiles are important for Python development because they automate
the many routine processes, making it easier and faster for developers
to build, test, and deploy their code.

# Architecture for our RecSys Project
TBD