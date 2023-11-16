FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ARGS
ARG INSTALLDIR="/webui" \
  RUN_UID=1000 \
  DEBIAN_FRONTEND=noninteractive \
  GIT_SHA
ENV INSTALLDIR=$INSTALLDIR \
  RUN_UID=$RUN_UID \
  DATA_DIR=$INSTALLDIR/data \
  TZ=Etc/UTC \
  GIT_SHA=$GIT_SHA

# Install dependencies (apt)
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  wget git libgl1 libglib2.0-0 \
  # necessary for extensions
  ffmpeg libglfw3-dev libgles2-mesa-dev pkg-config libcairo2 libcairo2-dev

# Setup user which will run the service
RUN useradd -m -u $RUN_UID webui-user
USER webui-user

# Copy Local Files to Container
COPY --chown=webui-user . $INSTALLDIR

# Setup venv and pip cache
RUN mkdir -p $INSTALLDIR/cache/pip \
  && mkdir -p $DATA_DIR

ENV PIP_CACHE_DIR=$INSTALLDIR/cache/pip

# Install dependencies (pip, wheel)
RUN pip install -U pip wheel

WORKDIR $INSTALLDIR
# Start container as root in order to enable bind-mounts
USER root

RUN ${INSTALLDIR}/entrypoint.sh --test --upgrade \
  --no-download \
  --skip-torch

ENTRYPOINT ["/bin/bash", "-c", "${INSTALLDIR}/entrypoint.sh \"$0\" \"$@\""]

CMD ["--listen", "--no-download", "--docs"]