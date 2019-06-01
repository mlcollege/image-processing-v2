FROM ubuntu:16.04
MAINTAINER adakolar@gmail.com

USER root
ENV SHELL /bin/bash
EXPOSE 8000

RUN apt-get update && apt-get install -yq --no-install-recommends software-properties-common, git vim sudo && apt-get clean
RUN add-apt-repository -y ppa:jonathonf/python-3.6 && apt-get update && apt-get install -yq python3.6

RUN ln -fs /usr/bin/python3.6 /usr/bin/python
RUN python -m pip install --upgrade pip setuptools wheel

ADD . /image_processing_workshop
WORKDIR /image_processing_workshop
RUN python setup.py develop
RUN jupyter nbextension enable --py widgetsnbextension
