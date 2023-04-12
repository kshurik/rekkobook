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