(chapter2_part1)=

# What comes after ML model development?
This chapter describes what is production code and how to write it using complementary tools
such as `poetry`, `Makefile` and code styling modules like `pylint`, `black`, `isort` and `flake8`.
Finally, we will define the architecture of our production microservice of RecSys we developed in
[Chapter 1](https://rekkobook.com/chapter1/full_pipeline.html)


After developing a machine learning (ML) model, the next step is to write production code
that enables the model to function in a real-world environment. Production code is typically
written in a programming language such as Python and executed on a server or virtual machine
that is accessible to end users. The goal of production code is to ensure that the ML model
performs as expected in a production environment, including handling real-world data, scaling
to meet demand, and providing accurate and timely predictions.


To ensure that production code is robust, scalable, and reliable, different skills are required
than those needed for research or experimentation. The production code pipeline involves several steps,
including:
- Replicating the development environment for full reproducibility of results;
- Data preparation such as collecting necessary input and calculating runtime features;
- Inference and postprocessing that includes making final predictions and any necessary
postprocessing of predictions, such as scaling


In addition, strong coding requirements must be met which can be tracked and fixed using:
- Use a consistent coding style and naming convention
- Write clear and concise comments
- Include error handling and logging
- Use version control to track changes and collaborate with others
- Write unit tests to ensure that the code works as expected

Therefore, various tools besides model and Python code are used. Here, let's make an overview of these tools
that is used in development today.


# Developement tools
## Poetry
The first tool to use when creating a production service or even any ML project to have full reproducibility
of the results is library version control. Previously, `requirements.txt` was very popular among developers
and Data Scientists. It is pretty straightforward in usage -- add the name and version of the library with a new line

```
pandas==1.1.5
numpy==1.23.5
etc.
```
However, more advanced config management has been discovered [`poetry`](https://python-poetry.org/) which allows
to use it for dependencies management and configuration file for popular dev tools (we will discuss it later).
The main features are:
- Dependencies management - resolves all dependencies for each library in the correct order. When using requirements.txt,
developers have to manually manage the installation and updating of dependencies. This can be time-consuming
and error-prone, especially when dealing with complex projects that have multiple dependencies. In contrast,
Poetry automates most of the dependency management process, making it easier to install, update, and remove dependencies.;
- Isolated virtualenv - automatically creates a virtual environment for reproducibility with a given Python version
and libraries (no need to execute by hand, just one line of command in CLI);
- CLI interaction to manage configuration files - intuitive commands to add/remove/install dependencies;
- BONUS: some parameters for the `pylint` styling guide, etc. can be set within the same config!


Now, let's discuss details about how to use it. To initialize the config we need to run the installation
```
curl -sSL https://install.python-poetry.org | python3 -
```

Then, we can initialize the poetry config by executing (assuming we want to create dependency management for the pre-existing project)
```
poetry init
```
After we run, several optional questions will be asked to create the `pyproject.toml` file -- the main configuration file.
You can either fill it (for instance set the Python version) or just press enter to use suggested default values.

Next, we add package names as following
```
poetry add pandas
```
or specific version
```
poetry add pandas==1.5.1
```
Further, it creates a virtual environment with all packages and you can use it to run your scripts/notebooks
or run by using bash commands
```
poetry run python inference.py
```

Also, you can set various parameters by utilizing the full power of TOML extensions. Below there is an example of 
a full poetry set-up generated for illustration purposes.

- `[tool.poetry]` - meta-information about the project
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

Overall, poetry provides an easy and efficient way to reproduce development environment,
resolve dependencies and allows using customization in some dev tools.

## Styling guide and code quality
As we have mentioned earlier, production code must be robust, scalable & reliable. Tracking
the quality of the code by reviews only is exhaustive and almost impossible to achieve -- we
are prone to errors too. However, most common mistakes like typos and convenient bugs can
be checked automatically and fixed. Here, we will describe the main Python tools to achieve that
- `[pylint](https://pypi.org/project/pylint/)` - code analyzer which does not run your code.
It checks for errors, and the coding standard makes clear suggestions on how to improve it
and even grade it on a scale of 0 to 10;

```{image} ./img/pylint.png
:alt: fishy
:class: bg-primary mb-1
:width: 400px
:align: centre
```

- `[black](https://pypi.org/project/black/)` - in the Python community there is well-known
PEP-8 standard to follow. This library is a code formatter which is PEP 8-compliant
opinionated formatter. It formats all code in given directory/files in place fast and efficiently;

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

- `[isort](https://pycqa.github.io/isort/)` - a library to make appropriate imports:
alphabetical order and group by types to sections. Below is the example from the official homepage

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
its dependencies. You would need to install these dependencies and set up
the environment to run the project. With a Makefile, you can automate this
process by defining targets for each step of the build,
such as "install dependencies", "set up environment", "run tests", etc. 

Here's an example Makefile for a Python project:

```
# install dependencies
install:
    pip install -r requirements.txt

# set up the environment
setup:
    virtualenv env
    source env/bin/activate

# run tests
test:
    python -m unittest discover -s tests

# clean up the environment
clean:
    rm -rf env/
    @find . | grep __pycache__ | xargs rm -rf

# let's add formatting stuff here as well

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
- `make pylint` will run a code formatting check;
- `make isort` will do a proper import of modules;
- `make black` will run formatting in place;
- `make fmt` will call `make isort` & `make black` for full formatting

Overall, Makefiles are important for Python development because they automate
the many routine processes, making it easier and faster for developers
to build, test, and deploy their code.

# Architecture for our RecSys Project
Let's recall the two-level architecture of a recommender system consists of two main components:
the candidate generator and the ranker. The candidate generator is responsible for 
selecting a set of items that are likely to be of interest to the user, and the ranker
takes those items and ranks them in order of predicted relevance to the user. Keeping
that in mind, the architecture for the light version will be as follows:

```{image} ./img/recsys_architecture.png
:alt: fishy
:class: bg-primary mb-1
:width: 400px
:align: centre
```

Here is a detailed description of each component:
1. `Client-user interface` to interact with the product;

2. `First-level model a.k.a candidate Generator`: The candidate generator is the first level in the
recommender system architecture. Its role is to select a set of items that are likely to be of
interest to the user. It is usually designed to be very fast and efficient, as it needs to process
a large number of potential items in a short amount of time.
In the architecture above service described, the candidate generator receives a `user_id` as an
input and uses a first-level model to select a set of candidate items that are likely to be of
interest to that user. The output of the candidate generator is a set of movie ids;

3. `FeatureStore` - a feature store is a centralized repository of features used in machine learning models.
It enables easy sharing and reuse of features across different models and teams. Instead of storing features in
separate databases, a feature store allows for efficient storage and retrieval of feature for inference.
In this project, we will be using parquet files stored in GDrive as an example, rather than traditional
databases. This will allow for easy integration and understanding key points of the project
instead of technical stuff with data storage;

4. `ReRanker`: The ranker is the second level in the recommender system architecture. Its role
is to take the set of candidate items generated by the candidate generator and rank them in order
of predicted relevance to the user. The ranker is usually a more complex and computationally
expensive model than the candidate generator, as it needs to process a smaller set of potential
items but with more detailed information

In our production architecture, the ranker takes the set of candidate items generated
by the first-level model and uses a feature store to enrich the data with additional information
about the items and the user. This additional information could include item features such as genre,
release date, or popularity, as well as user features such as demographics, past purchases,
or browsing history. The ranker then applies a more sophisticated machine learning model to predict
the relevance of each item to the user and sorts the items in order of predicted relevance.
The output of the ranker is a sorted list of items that are likely to be of interest to the user.

# How to structure the project
Structuring an ML service is essential for a well-organized, maintainable, and scalable project.
A well-structured project is easy to understand and easy to modify when requirements change.

There are four base modules for structuring an ML service:
- configs;
- data preparation;
- models;
- utils

Configs hold the parameters and settings of the project, data preparation contains the code
for data processing and feature engineering, models contain the code for training, evaluating,
and deploying models, and utils have common functions and classes that are frequently used throughout the project.


`Configs` are an essential part of any ML project as they hold the settings and parameters of
the project. Dynaconf is a popular Python package that simplifies working with configurations.
Dynaconf enables developers to store configuration parameters in a file or an environment
variable and read them in their code. An example of working with Dynaconf you can find in our project [here](https://github.com/kshurik/rekkobook/tree/chapter2/api_example/supplements/recsys/configs). It is a set of TOML files with parameters that
are combined together using the Dynaconf loaders


The `data preparation` module is responsible for loading and processing the input data before
feeding it into the ML models. This module may contain functions for tasks such as data cleaning,
normalization, feature engineering, and splitting the data into training and validation sets.
Depending on the nature of the data and the specific ML problem being tackled, this module can be
quite complex and may require its own submodules. 


The `models` module contains the actual ML models inference used to make predictions. This module may include
different models for different tasks, as well as the code necessary to train and evaluate these models.
In addition, this module may also include functions for loading and saving pre-trained models,
as well as for fine-tuning models with new data.


Finally, the `utils` module contains commonly used functions or classes shared across
different parts of the service. This module may include functions for logging, visualization,
file I/O, or any other utility functions that are not specific to either the data preparation or modeling modules.


Thus, by separating the code into these four modules, the ML service can be organized in a logical
and modular way, making it easier to maintain and extend over time.