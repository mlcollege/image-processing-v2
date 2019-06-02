FROM ubuntu:16.04
MAINTAINER adakolar@gmail.com

USER root
ENV SHELL /bin/bash
EXPOSE 8000 6006

RUN apt-get update && apt-get install -yq --no-install-recommends software-properties-common git vim sudo wget && apt-get clean
RUN add-apt-repository -y ppa:jonathonf/python-3.6 && apt-get update && apt-get install -yq python3.6 python3.6-dev

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN ln -fs /usr/bin/python3.6 /usr/bin/python
RUN ln -fs /usr/local/bin/pip3.6 /usr/local/bin/pip
RUN pip install --upgrade pip setuptools wheel

ADD . /image_processing_workshop
WORKDIR /image_processing_workshop
RUN python setup.py develop
RUN jupyter nbextension install --py widgetsnbextension --user
