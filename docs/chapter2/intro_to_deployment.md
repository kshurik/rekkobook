# What comes after ML model development?
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
