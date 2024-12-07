# Docker

SD.Next includes basic [Dockerfile](https://github.com/vladmandic/automatic/blob/dev/Dockerfile) for use with **nVidia GPU** equipped systems  
Other system may require different configurations and base images, but principle remains  

Goal of containerized SD.Next is to provide a fully stateless environment that can be easily deployed and scaled  

SD.Next docker template is based on [official base image](https://hub.docker.com/r/pytorch/pytorch/tags) with `torch==2.5.1` with `cuda==12.4`

SD.Next docker image is currently not published in docker hub or any other repository since typically each user or organization will have their own customizations and requirements and build process is very simple and fast  

## Prerequisites

!!! info

    If you already have functional Docker on your host, you can skip this section  
    For manualy steps see appendix at the end of the document  

- Docker itself  
  <https://docs.docker.com/get-started/get-docker/>
- nVidia Container ToolKit to enable GPU support  
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

## Build Image

!!! note

    Building SDNext docker image is normally only required once and takes between few seconds (using cached image) to ~1.5min (initial build) to complete  
    First build will also need to download the base image, which can take a while depending on your connection  
    If you make changes to `Dockerfile` or update SD.Next, you will need to rebuild the image  

!!! info

    Build process should be done on a system where SD.Next was started at least once to download all required submodules before docker copy process  

```shell
docker build \
  --debug \
  --tag sdnext/sdnext-cuda \
  <path_to_sdnext_folder>

docker image inspect sdnext/sdnext-cuda  
```

```log
    [+] Building 93.3s (12/12) FINISHED                                                     docker:default
    [internal] load build definition from Dockerfile                                              0.0s
    transferring dockerfile: 2.25kB                                                               0.0s
    [internal] load metadata for docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime          0.0s
    [internal] load .dockerignore                                                                 0.0s
    transferring context: 366B                                                                    0.0s
    CACHED [1/7] FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime                     0.0s
    [internal] load build context                                                                 1.3s
    transferring context: 417.02MB                                                                1.3s
    [2/7] RUN ["apt-get", "-y", "update"]                                                         4.4s
    [3/7] RUN ["apt-get", "-y", "install", "git", "build-essential", "google-perftools", "curl"  20.7s
    [4/7] RUN ["/usr/sbin/ldconfig"]                                                              0.3s
    [5/7] COPY . /app                                                                             0.8s
    [6/7] WORKDIR /app                                                                            0.0s
    [7/7] RUN ["python", "/app/launch.py", "--debug", "--uv", "--use-cuda", "--log", "sdnext.lo  63.9s
    exporting to image                                                                            3.1s
    exporting layers                                                                              3.1s
    writing image sha256:5b2571c1f2a71f7a6d5ce4b1de1ec0e76cd4f670a1ebc17de79c333fb7fffd46         0.0s
    naming to docker.io/sdnext/sdnext-cuda                                                        0.0s
```

Base image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` is 6.14GB  
And full SD.Next resulting image is ~8.8GB and contains all required dependencies  

!!! warning

    If you have build errors, run with `--progress=plain` to get full build log

## Run Container

!!! note

    - Republishes port from container to host directly  
      You may need to remap ports if you have multiple containers running on the same host  
    - Maps local server folder `/server/data` to be used by the container as data root  
      This is where all state items and outputs will be read from and written to  
    - Maps local server folder `/server/models` to be used by the container as model root  
      This is where models will be read from and written to  

```shell
docker run \
  --name sdnext-container \
  --rm \
  --gpus all \
  --publish 7860:7860 \
  --mount type=bind,source=/server/models,target=/mnt/models \
  --mount type=bind,source=/server/data,target=/mnt/data \
  --detach \
  sdnext/sdnext-cuda
```

Typical SDNext container will start in ~10sec and will be ready to accept connections on port `7860`

### State

As mentioned, the goal of SD.Next docker deployment is fully stateless operations.  
By default, SD.Next docker containers is stateless: any data stored inside the container is lost when the container stops.  

All state items and outputs will be read from and written to `/server/data`  
This includes:
- Configuration files: `config.json`, `ui-config.json`
- Cache information: `cache.json`, `metadata.json`
- Outputs of all generated images: `outputs/`

### Persistence

If you plan to customize SD.Next deployment with additional extensions,  
you may want to create and map docker volume to avoid constaint reinstalls on each startup.  

### Healthchecks

By default, SD.Next docker container does not include healthchecks, but they can be enabled.
Simply remove comment from `HEALTHCHECK` line in `Dockerfile` and rebuild the image.  

## Extra

Additional docker commands that may be useful

!!! tip

    Clean Up

```shell
docker image ls --all
docker image rm <id>
docker builder prune --force  
```

!!! tip

    List Containers

```shell
docker container ls --all
docker ps --all
```

!!! tip

    View Log

```shell
    docker container logs --follow <id>
```

!!! tip

    Stop Container

```shell
    docker container stop <id>
```

!!! tip

    Test GPU

```shell
docker info  
docker run --name cudatest --rm --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark  
```

!!! tip

    Test Torch

```shell
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime  
docker run --name pytorch --rm --gpus all -it pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime  
```

## Manual Install

> Docker

```shell
wget https://download.docker.com/linux/ubuntu/dists/noble/pool/stable/amd64/containerd.io_1.7.23-1_amd64.deb
wget https://download.docker.com/linux/ubuntu/dists/noble/pool/stable/amd64/docker-ce_27.3.1-1~ubuntu.24.04~noble_amd64.deb
wget https://download.docker.com/linux/ubuntu/dists/noble/pool/stable/amd64/docker-ce-cli_27.3.1-1~ubuntu.24.04~noble_amd64.deb
wget https://download.docker.com/linux/ubuntu/dists/noble/pool/stable/amd64/docker-buildx-plugin_0.17.1-1~ubuntu.24.04~noble_amd64.deb
wget https://download.docker.com/linux/ubuntu/dists/noble/pool/stable/amd64/docker-compose-plugin_2.29.7-1~ubuntu.24.04~noble_amd64.deb
sudo dpkg -i *.deb

sudo groupadd docker
sudo usermod -aG docker $USER
systemctl status docker
systemctl status containerd
```

> nVidia Container ToolKit

```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```
