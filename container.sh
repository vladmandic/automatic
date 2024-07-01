#!/usr/bin/env bash

cd -- "$(dirname -- "$0")"

if [[ -z "${PYTHON}" ]]
then
    PYTHON="python3"
fi
if [[ -z "${venv_dir}" ]]
then
    venv_dir="venv"
fi

if [[ ! -d "${venv_dir}" ]]
then
    "${PYTHON}" -m venv "${venv_dir}"
fi
source "${venv_dir}"/bin/activate

PYTHON="${venv_dir}/bin/python3"
img=$(exec "${PYTHON}" ./docker/docker.py "$@" | tail -n 1)

echo

docker build -t sd-next -f ./docker/Dockerfile --build-arg "BASE_IMG=$img" .
docker rm "SD-Next"
docker run -it --device /dev/dri -v SD-Next:/workspace -v SD-Next_Venv:/python/venv -v SD-Next_Cache:/root/.cache -p 7860:7860 --gpus=all --name "SD-Next" sd-next
