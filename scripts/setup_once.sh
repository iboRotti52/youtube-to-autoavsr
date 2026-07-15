#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 bulunamadi. Once Python 3.11 kur."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  python3.11 -m venv .venv
fi

".venv/bin/python" -m pip install -e .
".venv/bin/yt2avsr" setup-external --config configs/default.yaml
".venv/bin/yt2avsr" setup-retinaface --config configs/default.yaml
".venv/bin/yt2avsr" setup-whisper --config configs/default.yaml

echo "Hazir. Bundan sonra calistirmak icin: ytavsr"
