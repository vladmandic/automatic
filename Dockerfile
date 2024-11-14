# SD.Next Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# TBD add org info
LABEL org.opencontainers.image.authors="vladmandic"
WORKDIR /
COPY . .
# stop pip and uv from caching
ENV PIP_NO_CACHE_DIR=true
ENV UV_NO_CACHE=true
# disable model hashing for faster startup
ENV SD_NOHASHING=true
# set data directories
ENV SD_DATADIR="/mnt/data"
ENV SD_MODELSDIR="/mnt/models"
# install dependencies
RUN ["apt-get", "-y", "update"]
RUN ["apt-get", "-y", "install", "git"]
# sdnext will run all necessary pip install ops and then exit
RUN ["python", "launch.py", "--debug", "--uv", "--use-cuda", "--log", "sdnext.log", "--test"]
# preinstall additional packages to avoid installation during runtime
RUN ["uv", "pip", "install", "-r", "requirements-extra.txt", "--system"]
# actually run sdnext
CMD ["python", "launch.py", "--debug", "--skip-all", "--listen", "--quick", "--api-log", "--log", "sdnext.log"]
# expose port
EXPOSE 7860
# TBD add healthcheck function
HEALTHCHECK NONE
STOPSIGNAL SIGINT
