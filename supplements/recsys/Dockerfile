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

# install poetry
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
CMD ["python", "api.py"]
#docker run -p 8080:8080 project_name -- run container
#docker build --platform=linux/amd64 . -t recsys_api