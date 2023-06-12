#!/usr/bin/env bash
set -Eeuo pipefail
source "$INSTALL_DIR/venv/bin/activate"

# Ensure that --data-dir is set
if [ -z "$DATA_DIR" ]; then
  echo "Must specify data directory for application state"
  exit 1
fi

# Ensure that potentially bind-mounted directories are owned by the user that runs the service
chown -R $UUID:$UUID "$DATA_DIR"

# Run service as specified (non-root) user
exec runuser -u $(id -un $UUID) -- python "$INSTALL_DIR/launch.py" --data-dir="$DATA_DIR" $@

