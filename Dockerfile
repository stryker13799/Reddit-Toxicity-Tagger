FROM python:3.8-slim-buster

## Updating the os
RUN apt-get update

## Creating the working directory
WORKDIR /APP

## Copy everything to the working directory
COPY . /APP

## Running the requirements
RUN python -m pip install -r requirements.txt

## Exposing the port
EXPOSE 5000

## Running the app file
CMD ["python","app.py"]