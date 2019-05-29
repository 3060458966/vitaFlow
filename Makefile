#!/usr/bin/env bash

# Setting Version
APP_VERSION=0.1
IMAGE_VERSION=0.1123


help:		### Help Command
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

clear:		### Clears Shell
	clear


version:		### To prints the Application Version
	@echo "Application Version "${APP_VERSION}
	set -e; \
	export IMAGE_VERSION=${IMAGE_VERSION}
	@echo "Image Version "${IMAGE_VERSION}

preprocess:		### OCR Pipeline - Pre-processing
	@echo "Starting Document Localisation - doc2text"
	python vitaflow/annotate_server/doc2text.py


binarisation:		### OCR Pipeline - Pre-processing
	@echo "Starting Document Binarisation - textcleaner"
	python vitaflow/annotate_server/binarisation.py


text2lineimages:		### OCR Pipeline - Pre-processing
	@echo "Starting Document to text line image converter"
	python vitaflow/annotate_server/image_to_textlines.py


tesseract:		### OCR Pipeline - OCR with Tessaract
	@echo "Starting OCR using Tesseract"
	python vitaflow/annotate_server/ocr.py


calmari:		### OCR Pipeline - OCR with Calamari OCR
	@echo "Starting OCR using Calamari"
	python vitaflow/annotate_server/ocr_calamari.py


ocr_pipeline: data_cleanup preprocess binarisation text2lineimages tesseract calmari		### OCR Pipeline - Run complete pipeline
	@echo "Starting OCR Pipeline(All Step)"


data_cleanup:		### OCR Pipeline - Clean all sub folder
	@echo "Starting "
	rm -rf vitaflow/annotate_server/static/data/east/*
	rm -rf vitaflow/annotate_server/static/data/images/*
	rm -rf vitaflow/annotate_server/static/data/binarisation/*
	rm -rf vitaflow/annotate_server/static/data/text_images/*
	rm -rf vitaflow/annotate_server/static/data/text_data/*


show_input:		### OCR Pipeline - Run complete pipeline
	ls -l vitaflow/annotate_server/static/data/preprocess/


east:		### OCR Pipeline - Run complete pipeline
	@echo "East Model "
	mkdir -p vitaflow/annotate_server/static/data/east/
	mkdir -p vitaflow/annotate_server/static/data/images/
	python vitaflow/playground/east/pb_predict.py \
    --images_dir="vitaflow/annotate_server/static/data/preprocess"\
    --output_dir="vitaflow/annotate_server/static/data/east"\
    --model="vitaflow/annotate_server/static/data/east_models/east/EASTModel/exported/1558013588"
	cp vitaflow/annotate_server/static/data/east/* vitaflow/annotate_server/static/data/images


crop2box:
	python vitaflow/annotate_server/crop_to_box.py


text2file:
	python vitaflow/annotate_server/text_file_stitch.py


east_ocr_pipeline:	data_cleanup east binarisation crop2box tesseract calmari text2file ### EAST OCR Pipeline - Run complete pipeline

###################################################################
########################################################## DOCKER #
###################################################################

#version: ## Prints Build version
#	set -e; \
#	export IMAGE_VERSION=${IMAGE_VERSION}
#	@echo "Image Version "${IMAGE_VERSION}

base: ## building base docker image
	docker build --rm -f Dockerfile-base -t vitaflow-base .


build: ## building docker image
	docker build --rm -t vitaflow:${IMAGE_VERSION} .

run: # run the VitaFlow Docker
	@echo "Running vitaflow/vitaflow-app - RUN"
	docker run -d --name vitaflow vitaflow:${IMAGE_VERSION} /bin/bash

rm: ## rm
	docker rm -f vitaflow

exec:
	# docker exec -it vitaflow /bin/bash
	docker run -i -t --name vitaflow vitaflow:${IMAGE_VERSION} bash

ps:
	docker ps -a

im:
	# help - https://docs.docker.com/engine/reference/commandline/build/#options
	docker images
