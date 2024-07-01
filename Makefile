# Enviromental variables
BUILD_CHANNEL?=local
OS=$(shell uname)
VERSION=v1.12.0
GIT_REVISION = $(shell git rev-parse HEAD | tr -d '\n')
TAG_VERSION?=$(shell git tag --points-at | sort -Vr | head -n1)
SHELL := /usr/bin/env bash 

# Default values for upladoing training script
ORG_ID = <ADD_YOUR_ORG_ID_HERE>
SCRIPT_NAME = red_candy_augmented
SCRIPT_VERSION = 0.1

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