# Docker and containerization of a two-level recsys microservice
In this chapter, we will explore Docker and containerization technique and move to an example with 
a two-level recsys microservice. We will start by discussing what Docker is and why it is useful
for building and deploying software applications. Then, we will go on to show how to build a
 Docker container for a recsys project we worked on previously in API development module.

## What is Docker?
Docker is a containerization platform that allows developers to create and deploy software
applications in an isolated, portable, and consistent environment. To simply put, it gives
you a tool to fully reproduce all the the stuff you did locally and maintain it throughout
the needed time. Docker containers are lightweight, standalone, and executable packages
that include everything needed to run an application, including code, libraries, dependencies,
and system tools. Containers are similar to virtual machines, but they are more lightweight
and efficient, and they run directly on the host operating system. 

Docker provides several benefits for building and deploying software applications, such as:
- Portability: Docker containers are self-contained and can be run on any system that supports 
Docker, regardless of the underlying hardware or operating system.

- Isolation: Docker containers are isolated from each other and from the host operating system,
which helps prevent conflicts and dependencies issues.

- Efficiency: Docker containers are lightweight and consume fewer resources than virtual machines,
which makes them faster to start and stop, and more scalable.

- Consistency: Docker containers are built from the same image, which ensures consistency across
different environments and deployments.

## Building a Docker container for a RecSys project
Let's remember, a two-level recsys microservice consists of two models: a candidate generator
and a reranker. The candidate generator is responsible for generating a list of candidate items
that are relevant to the user's id, while the reranker is responsible for reordering the list based
on the user's preferences. The microservice also includes a feature store (parquet file), which stores
the features used by the models, and an API endpoint (run.py), which allows clients to query
the microservice and receive recommendations.

To build a Docker container for a two-level recsys microservice, we need to create a `Dockerfile`,
which is a script that contains instructions for building the container. The Dockerfile specifies
the base image, the dependencies, the application code, and the startup command.

Here is an example Dockerfile for our project:
```
# sse an official Python image
FROM python:3.9-slim-buster

# set variable for working directory
ARG work_dir=/app
WORKDIR $work_dir

# copy the poetry config to the container
COPY pyproject.toml poetry.lock $work_dir/

# install poetry
RUN pip install poetry

# install dependencies
RUN poetry export --with=dev --without-hashes --output requirements.txt \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . $work_dir

# Expose the port used by the microservice
EXPOSE 8080

# load the models and start the microservice
CMD ["python", "run.py"]
```

Let's go through each line of the Dockerfile:

- `FROM python:3.9-slim-buster`: specifies the base image for the container, which is the official Python 3.9 slim version;

- `ARG work_dir=/app`: creates variable work_dir to reuse in further commands;

- `WORKDIR $work_dir`: sets the working directory inside the container to `/app` from taken `work_dir`;

- `COPY pyproject.toml poetry.lock $work_dir/`: copies the poetry configs from the host into the container;

- `RUN poetry export --with=dev --without-hashes --output requirements.txt && pip3 install --no-cache-dir -r requirements.txt`: installs the dependencies specified in poetry configs;

- `COPY . $work_dir`: copies the rest of the application code from the host into the container;

- `EXPOSE 8080`: specifies that the microservice will listen on port;

- `CMD ["python", "run.py"]` - runs container and the models. Ready to receive requests

