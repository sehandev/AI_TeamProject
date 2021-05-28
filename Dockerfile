FROM nvidia/cuda:11.1.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt update && apt install -y \
   curl \
   ca-certificates \
   sudo \
   git \
   bzip2 \
   libx11-6 \
   gcc \
   g++ \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
   && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh \
   && chmod +x ~/miniconda.sh \
   && ~/miniconda.sh -b -p ~/miniconda \
   && rm ~/miniconda.sh \
   && conda install -y python==3.8.3 \
   && conda clean -ya

# CUDA 11.1.1-specific steps
RUN conda install -y -c conda-forge cudatoolkit=11.1.1 \
   && conda clean -ya

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /home/user/.jupyter
RUN chmod 777 /home/user/.jupyter

RUN conda install -y -n base -c conda-forge widgetsnbextension ipywidgets \
   && conda clean -ya

WORKDIR /workspace

ENTRYPOINT ["jupyter", "lab"]