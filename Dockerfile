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
#RUN apt-get update
#RUN apt-get -y install python3
#RUN apt-get update && apt-get install -y python3-pip
#Install python3 pip3
RUN apt-get update
#RUN apt-get -y install python3
#RUN apt-get install -y python-apt --reinstall
RUN add-apt-repository ppa:jamesh/snap-support && apt-get update && apt install -y patchelf
#RUN add-apt-repository ppa:jonathonf/python-3.6 -y && apt-get update && apt-get install -y python3.6
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y python3.6 python3.6-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py
RUN pip3 install --upgrade pip
RUN apt-get install -y python-apt --reinstall

# RUN mkdir /install
# WORKDIR /install

#### -------------------------------------------------------------------
#### install tensorflow
#### -------------------------------------------------------------------
RUN pip3 install --upgrade tensorflow-gpu
RUN pip3 install sacred pymongo pyyaml
RUN mkdir /install 
WORKDIR /install
#RUN git clone https://github.com/openai/multiagent-particle-envs.git maenv && cd maenv && pip3 install -e .
RUN git clone https://github.com/schroeder-dewitt/multiagent-particle-envs.git maenv && cd maenv && pip3 install -e .

#### -------------------------------------------------------------------
#### install mujoco
#### -------------------------------------------------------------------
RUN apt install -y libosmesa6-dev libglew1.5-dev

WORKDIR /beieng
RUN chmod -R 777 /beieng
RUN chmod -R 777 /usr/local

RUN useradd -d /beieng -u 13290 beieng
USER beieng

#RUN apt install -y libosmesa6-dev libglew1.5-dev
RUN mkdir -p /beieng/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /beieng/.mujoco \
    && rm mujoco.zip \
    && cd /beieng/.mujoco && mv mujoco200_linux mujoco200

RUN export PATH=$PATH:$HOME/.local/bin
COPY ./mujoco_key.txt /beieng/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /beieng/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MJKEY_PATH /beieng/.mujoco/mjkey.txt
ENV MUJOCO_PY_MUJOCO_PATH /beieng/.mujoco/mujoco200

#RUN pip3 install gym[mujoco] --upgrade
RUN pip3 install mujoco-py
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/beieng/.mujoco/mujoco200/bin" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/beieng/.mujoco/mujoco200/bin" >> ~/.profile
RUN python3 -c "import mujoco_py"


# set pythonpath
RUN echo "export PYTHONPATH=/beieng/pymarl" >> ~/.bashrc
RUN echo "export PYTHONPATH=/beieng/pymarl" >> ~/.profile

RUN pip3 install gym==0.10.8

EXPOSE 8888

WORKDIR /maddpg

