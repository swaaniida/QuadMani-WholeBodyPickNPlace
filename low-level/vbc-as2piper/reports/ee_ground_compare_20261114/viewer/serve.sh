#!/usr/bin/env bash
set -euo pipefail
umask 0002
cd "$(dirname "$0")"
echo "3D replay: http://127.0.0.1:${PORT:-8774}"
exec python3 -m http.server "${PORT:-8774}" --bind 127.0.0.1
