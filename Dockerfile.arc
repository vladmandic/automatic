# SD.Next Dockerfile
# docs: <https://github.com/vladmandic/automatic/wiki/Docker>

# Intel ARC "Dockerfile.arc" usage: docker build -f Dockerfile.arc .

# base image
FROM ubuntu:24.04

# metadata
LABEL org.opencontainers.image.vendor="SD.Next"
LABEL org.opencontainers.image.authors="vladmandic"
LABEL org.opencontainers.image.url="https://github.com/vladmandic/automatic/"
LABEL org.opencontainers.image.documentation="https://github.com/vladmandic/automatic/wiki/Docker"
LABEL org.opencontainers.image.source="https://github.com/vladmandic/automatic/"
LABEL org.opencontainers.image.licenses="AGPL-3.0"
LABEL org.opencontainers.image.title="SD.Next"
LABEL org.opencontainers.image.description="SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models"
LABEL org.opencontainers.image.base.name="https://hub.docker.com/ubuntu/ubuntu:24.04"
LABEL org.opencontainers.image.version="latest"

# minimum install
RUN apt-get -y update && apt-get -y install git build-essential google-perftools curl ca-certificates wget gpg software-properties-common

# install intel reops and packages
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | \
        tee /etc/apt/sources.list.d/intel-gpu-noble.list \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update && apt-get -y install libze-intel-gpu1 libze1 intel-opencl-icd clinfo intel-gsc libze-dev intel-ocloc python3.12 python3-pip python3-venv libgl1 libglib2.0-0 libgomp1 \
    && apt-get -y full-upgrade \
    && rm -rf /var/lib/apt/lists/*

# optionally...
# install the deadsnakes PPA: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
# apt-get install -t python3.11 python3.11-venv
# make python 3.11 the default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && python3 -V

# this is "dumb" - but the docker is the isolation in this case...
RUN rm /usr/lib/python3.*/EXTERNALLY-MANAGED

RUN ["/usr/sbin/ldconfig"]

# copy sdnext
COPY . /app
WORKDIR /app

# if using webui.sh and venv and a different version of python:
# make python 3.11 the default by setting the variable in the webui-user.sh
#RUN echo "export PYTHON=/usr/bin/python3.11" > webui-user.sh

# stop pip and uv from caching
ENV PIP_NO_CACHE_DIR=true
ENV PIP_ROOT_USER_ACTION=ignore
ENV UV_NO_CACHE=true
# disable model hashing for faster startup
ENV SD_NOHASHING=true
# set data directories
ENV SD_DATADIR="/mnt/data"
ENV SD_MODELSDIR="/mnt/models"
ENV SD_DOCKER=true
# use the US IPEX download server. Comment/remove for the CN download server.
ENV SD_IPEX_USE_US_SERVER=true
# network debugging
ENV SD_EN_DEBUG=true
# debug 
ENV SD_PROCESS_DEBUG=true

# tcmalloc is not required but it is highly recommended
ENV LD_PRELOAD=libtcmalloc.so.4  

# setup and activate the venv
#RUN python3 -m venv venv
#ENV VIRTUAL_ENV=/app/venv
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/venv/lib:/opt/rocm/lib
#ENV PATH=$VIRTUAL_ENV/bin:$PATH:/opt/rocm/bin

# this will make the docker images quite large
# some requirements are used very early launch
#RUN pip install -r requirements.txt


# this will only work with "--use-ipex" if the host system has a supported Intel GPU
# it will also increase the size of the docker image
# sdnext will run all necessary pip install ops and then exit
#RUN ["python3", "/app/launch.py", "--debug", "--uv", "--use-ipex", "--test", "--log",  "sdnext-setup.log"]
# preinstall additional packages to avoid installation during runtime

# actually run sdnext
CMD ["python3", "/app/launch.py", "--debug", "--listen", "--api-log", "--log", "sdnext.log"]

#CMD ["/bin/bash", "webui.sh", "--use-ipex"]

# expose port
EXPOSE 7860

# healthcheck function
# HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 CMD curl --fail http://localhost:7860/sdapi/v1/status || exit 1

# stop signal
STOPSIGNAL SIGINT
