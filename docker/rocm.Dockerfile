FROM rocm/dev-ubuntu-22.04:6.0.2

USER root
WORKDIR /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY ./ ./

RUN chmod -R 755 ./

RUN ./docker/setup.sh

ENV venv_dir=/python/venv

CMD ["./webui.sh", "--uv", "--listen", "--use-rocm"]