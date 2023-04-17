FROM python:3.9-slim-bullseye

# ENV HOME /home

WORKDIR /home

COPY . .
# RUN pip3 install -r requirements.txt


ENTRYPOINT ["tail", "-f", "/dev/null"]
# ENTRYPOINT ["python3", "app/api.py"]














# FROM python:3.9-slim-bullseye

# # ENV PYTHONDONTWRITEBYTECODE 1
# # ENV PYTHONBUFFERED 1
# # ENV PIP_NO_CACHE_DIR 1
# # ENV PYTHONPATH /home
# # ENV HOME /home

# RUN apt-get update && apt-get install -y \
#     gcc

# # WORKDIR ${HOME}
# COPY . .


# RUN pip3 install -r requirements.txt

# # ENTRYPOINT python3 app/api.py
# ENTRYPOINT ["tail", "-f", "/dev/null"]