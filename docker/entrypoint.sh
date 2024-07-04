#!/usr/bin/env bash

cd "$(dirname "$(dirname "$0")")"

echo
echo "======================="
echo "== SD Next Container =="
echo "======================="
echo

if [ $# -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi