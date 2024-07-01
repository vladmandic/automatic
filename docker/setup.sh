#!/usr/bin/env bash

echo Updating apt...
apt update -y > /dev/null

echo Upgrading apt...
apt upgrade -y > /dev/null

echo Installing apt packages...
apt install -y git \
    python3.10 \
    pythonpy \
    python3.10-venv \
    python3-pip \
    curl \
    wget \
    aria2 > /dev/null