# SD.Next Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
LABEL org.opencontainers.image.vendor="SD.Next"
LABEL org.opencontainers.image.authors="vladmandic"
LABEL org.opencontainers.image.url="https://github.com/vladmandic/automatic/"
LABEL org.opencontainers.image.documentation="https://github.com/vladmandic/automatic/wiki/Docker"
LABEL org.opencontainers.image.source="https://github.com/vladmandic/automatic/"
LABEL org.opencontainers.image.licenses="AGPL-3.0"
LABEL org.opencontainers.image.title="SD.Next"
LABEL org.opencontainers.image.description="SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models"
LABEL org.opencontainers.image.base.name="https://hub.docker.com/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
WORKDIR /
COPY . .
# stop pip and uv from caching
ENV PIP_NO_CACHE_DIR=true
ENV PIP_ROOT_USER_ACTION=ignore
ENV UV_NO_CACHE=true
ENV SD_INSTALL_DEBUG=true
# disable model hashing for faster startup
ENV SD_NOHASHING=true
# set data directories
ENV SD_DATADIR="/mnt/data"
ENV SD_MODELSDIR="/mnt/models"
# minimum install
RUN ["apt-get", "-y", "update"]
RUN ["apt-get", "-y", "install", "git", "build-essential", "google-perftools"]
RUN ["ldconfig"]
# tcmalloc is not required but it is highly recommended
ENV LD_PRELOAD=libtcmalloc.so.4  
# sdnext will run all necessary pip install ops and then exit
RUN ["python", "launch.py", "--debug", "--uv", "--use-cuda", "--log", "sdnext.log", "--test", "--optional"]
# preinstall additional packages to avoid installation during runtime
# actually run sdnext
CMD ["python", "launch.py", "--debug", "--skip-all", "--listen", "--quick", "--api-log", "--log", "sdnext.log"]
# expose port
EXPOSE 7860
# TBD add healthcheck function
HEALTHCHECK NONE
STOPSIGNAL SIGINT
