ARG BASE_IMG
FROM ${BASE_IMG}

USER root
WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY ./ ./

RUN ./docker/setup.sh

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

ENTRYPOINT ["./docker/entrypoint.sh"]