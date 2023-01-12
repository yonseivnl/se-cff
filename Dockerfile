FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get upgrade -y
RUN apt-get install -y vim git
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get install -y tmux
RUN apt-get install -y wget
RUN apt-get install -y htop

ENV PYTHONIOENCODING=UTF-8

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN bash Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -b
RUN rm -f Miniconda3-py39_22.11.1-1-Linux-x86_64.sh
RUN conda init bash

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools==57.0.0
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN conda install -y -c numba numba==0.53.1
RUN conda install -y -c conda-forge h5py==2.10.0 blosc-hdf5-plugin==1.0.0 scikit-video==1.1.11 tqdm==4.61.1 prettytable==2.1.0
RUN python3 -m pip install yacs==0.1.8 pytz==2021.1 tensorboard==2.5.0 opencv-python==4.5.2.54 einops==0.3.2 matplotlib

ARG DEBIAN_FRONTEND=teletype
