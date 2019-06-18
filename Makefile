NAME=mlcollege/image-processing
BUILD_TIME=$(shell date -u +'%Y%m%dT%H%M%SZ')

all: download build run

run:
	docker run -it --shm-size 8G -p 9997:8000 -p 6006:6006 -v $(shell pwd)/notebooks:/image_processing_workshop/notebooks $(NAME) ./run.sh

run_terminal:
	docker run -it --shm-size 8G -p 9997:8000 -p 6006:6006 -v $(shell pwd)/notebooks:/image_processing_workshop/notebooks $(NAME) /bin/bash

stop:
	docker stop $(NAME)

build:
	docker build -t $(NAME) .

push:
	docker tag $(NAME):latest $(NAME):$(BUILD_TIME)
	docker push $(NAME):latest
	docker push $(NAME):$(BUILD_TIME)

pull:
	docker pull $(NAME):latest
