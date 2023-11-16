#!/usr/bin/env bash
set -Eeuo pipefail

# Ensure that --data-dir is set
if [ -z $DATA_DIR ]; then
  echo "Must specify data directory for application state"
  exit 1
fi

# Ensure that potentially bind-mounted directories are owned by the user that runs the service
chown -R $RUN_UID:$RUN_UID $DATA_DIR
# Create directory for temporary files and assign it to the user that runs the service
mkdir -p /tmp/gradio
chown -R $RUN_UID:$RUN_UID /tmp/gradio

# Run service as specified (non-root) user
exec runuser -u $(id -un $RUN_UID) -- python "$INSTALLDIR"/launch.py --data-dir="$DATA_DIR" "$@"
