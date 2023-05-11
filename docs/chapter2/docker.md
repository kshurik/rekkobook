(chapter2_part3)=

# Containerization with Docker of a two-level recsys microservice
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
# use an official Python image
FROM python:3.9-slim-buster

# install c++ compilers
RUN apt-get update && \
    apt-get -y install gcc

# set variable for working directory
ARG work_dir=/app
WORKDIR $work_dir

# copy the poetry config to the container
COPY pyproject.toml $work_dir/

# install poetry and Cmake
RUN pip install poetry Cmake

# install dependencies
RUN poetry export --with=dev --without-hashes --output requirements.txt \
    && pip3 install --no-cache-dir -r requirements.txt

# workaround to install lightfm
RUN pip install lightfm==1.17 --no-use-pep517

# copy the rest of the app code to the container
COPY . $work_dir

# expose the port used by the microservice
EXPOSE 8080

# load the models and start the app
CMD ["python", "run.py"]
```

Let's go through each line of the Dockerfile:

- `FROM python:3.9-slim-buster` - specifies the base image for the container, which is the official Python 3.9 slim version;

- `RUN apt-get update && apt-get -y install gcc` - install compilators for C++ (some libraries need that usually);

- `ARG work_dir=/app` - creates variable work_dir to reuse in further commands;

- `WORKDIR $work_dir`- sets the working directory inside the container to `/app` from taken `work_dir`;

- `COPY pyproject.toml $work_dir/` - copies the poetry configs from the host into the container;

- `RUN poetry export --with=dev --without-hashes --output requirements.txt && pip3 install --no-cache-dir -r requirements.txt` - installs the dependencies specified in poetry configs;

- `RUN pip install lightfm==1.17 --no-use-pep517` - install LightFM library - it has issues for not integrating PEP-517 compliance and this
is a workaround for that;

- `COPY . $work_dir` - copies the rest of the application code from the host into the container;

- `EXPOSE 8080` - specifies that the microservice will listen on port;

- `CMD ["python", "run.py"]` - runs the container and the models. Ready to receive requests

Now that we have written a Dockerfile, it's time to get familiar with some of the most frequently
used Docker commands to run it. The containerization process allows us to isolate our application's
dependencies, making it easier to manage and deploy. With docker commands, we can build, run,
and manage containers for our applications. Some of the most frequently used docker commands include:

Sure, here's a brief overview of the most frequently used Docker commands:
1. `docker run` starts a container from an image.
2. `docker build` builds a Docker image from a Dockerfile.
3. `docker ps` lists all running containers.
4. `docker stop` stops a running container.
5. `docker rm` remove a stopped container.
6. `docker images` lists all the Docker images on the system.
7. `docker pull` pulls an image from a registry (like Docker Hub).
8. `docker push` pushes an image to a registry (like Docker Hub).
9. `docker exec` executes a command inside a running container.
10. `docker logs` shows the logs of a running container.

These commands form the foundation of most Docker workflows and are the most frequently used.

So, what do we need to do to run our Dockerfile? For that, you need to follow these steps:

- Ensure you have installed Docker Engine from [here](https://docs.docker.com/engine/install/);
- Open the app;
- Go to project root dir in the command line with `cd {path_name}` command. For me, it looks like `cd ~/Desktop/private/rekko_handbook/supplements/recsys` (to check if you are in the right directory, run `ls` which lists files);
- `docker build --platform=linux/amd64 . -t recsys_api` run in the command line - `--platform=linux/amd64` is required for M1 chips,
otherwise, it fails to install catboost, `recsys_api` is a project name and you can set it any. This way you build 
your Docker image which will be running in the next line by a command;
- `docker run -p 8080:8080 recsys_api` - runs a Docker image on port 8080 and the project name we defined earlier
(note: you can set any available ports and project name -- just make sure it is consistent with your image)

Now, if all went well, you will see some kind of this output - indicating your app is up and running.
```
<jemalloc>: MADV_DONTNEED does not work (memset will be used instead)
<jemalloc>: (This is the expected behaviour if you are running under QEMU)
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
 * Running on http://172.17.0.2:8080
```
Therefore, you can take a local URL and test it in a browser or by request module. In the browser, just put `http://127.0.0.1:8080/get_recommendation?user_id=176549&top_k=100`

Congrats! You are all done with your RecSys containerization. Full example to run you can find [here](https://github.com/kshurik/rekkobook/tree/chapter2/docker_example)