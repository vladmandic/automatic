# Stable Diffusion - Automatic (Docker)

Docker image of [Stable Diffusion - Automatic](https://github.com/vladmandic/automatic).

Now it supports only on NVIDIA GPU.

## Requirement

* [Docker](https://docs.docker.com/engine/install/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Setup

    docker compose build

## Usage

    docker compose up

After startup, you can access to http://localhost:7860.

## Configure

Create `docker-compose.override.yml` to change the command arguments or data location.

```yaml
services:
  nvidia:
    command: |
      --data-dir=/webui/repo/data
      --listen
      --noupdate
    volumes:
      - /mnt/c/Projects/sd_webui_data:/webui/repo/data
      - /mnt/c/Projects/sd_webui_voutput:/webui/repo/outputs
```
