FROM nvidia/cuda:10.0-base
# FROM ubuntu:18.04

LABEL maintainer "Abhishek"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

### General packages ###
RUN apt-get update && apt-get install -y \
	wget \
	git \
	exuberant-ctags \
	sudo \
	locate \
	curl \
	unzip \
	tree \
	python-pip \
	python3-pip \
	vim

RUN pip install --upgrade pip && \
	pip3 install --upgrade pip

### Install Miniconda3
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

### Packages ###
RUN conda install -y \
	tensorflow-gpu \
	keras

RUN conda install -y \
	numpy \
	rasterio \
	geopandas \
	fiona \
	shapely \
	matplotlib \
	pandas \
	scipy \
	scikit-learn \
	scikit-image

RUN pip install \
	pudb
# 	google-cloud-datastore \
# 	google-cloud-storage

### Google cloud SDK
# RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
#     echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
#     curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
#     apt-get update -y && apt-get install google-cloud-sdk -y

