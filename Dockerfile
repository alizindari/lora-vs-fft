FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    software-properties-common \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip for Python 3.11
RUN python3.11 -m pip install --upgrade pip

# Install LLaMA Factory with deepspeed support
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && \
    cd LLaMA-Factory && \
    python3.11 -m pip install -e . && \
    python3.11 -m pip install -r requirements/metrics.txt && \
    python3.11 -m pip install -r requirements/deepspeed.txt

# Install lm-eval-harness
RUN python3.11 -m pip install lm-eval

# Set working directory to LLaMA-Factory
WORKDIR /opt/LLaMA-Factory

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}