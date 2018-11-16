FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER Christian Schroeder de Witt

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

#Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get update && apt-get install -y python3-pip

# RUN mkdir /install
# WORKDIR /install

#### -------------------------------------------------------------------
#### install tensorflow
#### -------------------------------------------------------------------
RUN pip3 install --upgrade tensorflow-gpu
RUN pip3 install sacred pymongo pyyaml
RUN mkdir /install 
WORKDIR /install
RUN git clone https://github.com/openai/multiagent-particle-envs.git maenv && cd maenv && pip3 install -e .

EXPOSE 8888

WORKDIR /maddpg
