#!/usr/bin/env bash
set -Eeuo pipefail

: "${OLD_PID:?OLD_PID is required}"
: "${CONTROLLER:?CONTROLLER is required}"

while kill -0 "$OLD_PID" 2>/dev/null; do
  sleep 5
done

exec "$CONTROLLER"
