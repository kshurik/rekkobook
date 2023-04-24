#!/bin/bash
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker-compose up -d --build
