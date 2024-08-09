ARG BASE_IMG
FROM ${BASE_IMG}

USER root
WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY ./ ./

RUN ./docker/setup.sh

ENTRYPOINT ["./docker/entrypoint.sh"]