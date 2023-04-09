# FROM ubuntu:20.04

# ARG DOCKER_USER_HOME=/home
# WORKDIR ${DOCKER_USER_HOME}
# COPY . .

# RUN apt-get update; \
#     apt-get -y install vim python3 python3-pip; \
#     pip3 install poetry; \
#     poetry install

# ENTRYPOINT ["tail", "-f"]
# ENTRYPOINT ["tail", "-f", "/dev/null"]
# ENTRYPOINT poetry run python3 application.py


FROM python:3.9-slim-bullseye

ENV POETRY_VERSION 1.4.1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV PIP_NO_CACHE_DIR 1

RUN apt-get update && apt-get install -y \
    gcc

RUN pip install "poetry==${POETRY_VERSION}"

ENV VIRTUAL_ENV /opt/venv
ENV HOME /home

WORKDIR ${HOME}
COPY . .

ENTRYPOINT ["tail", "-f", "/dev/null"]