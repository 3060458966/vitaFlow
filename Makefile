#!/usr/bin/env bash

# Setting Version
APP_VERSION=0.1


help:		### Help Command
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

clear:		### Clears Shell
	clear


version:		### To prints the Application Version
	@echo "Application Version "${APP_VERSION}


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
	python vitaflow/annotate_server/ocr_calamari.py

data_cleanup:		### OCR Pipeline - Clean all sub folder
	@echo "Starting "
	rm -rf vitaflow/annotate_server/static/data/images/*
	rm -rf vitaflow/annotate_server/static/data/binarisation/*
	rm -rf vitaflow/annotate_server/static/data/text_images/*

show_input:		### OCR Pipeline - Run complete pipeline
	ls -l vitaflow/annotate_server/static/data/preprocess/
