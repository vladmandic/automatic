FROM rocm/dev-ubuntu-22.04:6.0.2

USER root
WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY ./ ./

RUN ./docker/setup.sh

CMD ["./webui.sh", "--uv", "--listen", "--use-rocm"]