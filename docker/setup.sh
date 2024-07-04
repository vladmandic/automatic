#!/usr/bin/env bash

echo
echo ======================
echo == Updating apt-get ==
echo ======================
echo
apt-get update -y

echo
echo =========================
echo == Setting up apt-fast ==
echo =========================
echo
apt-get install curl wget aria2 -y
/bin/bash -c "$(curl -sL https://git.io/vokNn)"
echo -e "_MAXNUM=16\n_MAXCONPERSRV=16\n_SPLITCON=64" > /etc/apt-fast.conf

echo
echo =========================
echo == Installing packages ==
echo =========================
echo
apt-fast install -y git \
    python3.10 pythonpy python3.10-venv python3-pip \
    curl wget aria2 \
    libcairo2 libcairo2-dev libgl1 libglib2.0-0 libgoogle-perftools4 libtcmalloc-minimal4 \
    ffmpeg pkg-config