# Enviromental variables
BUILD_CHANNEL?=local
OS=$(shell uname)
VERSION=v1.12.0
GIT_REVISION = $(shell git rev-parse HEAD | tr -d '\n')
TAG_VERSION?=$(shell git tag --points-at | sort -Vr | head -n1)
SHELL := /usr/bin/env bash 

# Default values for upladoing training script
ORG_ID = 53e2a500-3fa0-4ad6-9844-098769838d87
SCRIPT_NAME = data_augmentation_test
SCRIPT_VERSION = 0.3

# Make commands
default: clean build

build:
	python3 setup.py sdist --formats=gztar

clean: 
	rm -rf dist training.egg-info

upload-training-script:
	viam training-script upload \
	--path=dist/training-0.1.tar.gz \
	--framework=tflite \
	--type=single_label_classification \
	--org-id=$(ORG_ID) \
	--script-name=$(SCRIPT_NAME) \
	--version=$(SCRIPT_VERSION)