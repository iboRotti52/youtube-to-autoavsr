#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/yt2avsr" ]]; then
  echo ".venv hazir degil. Once calistir:"
  echo "$ROOT/scripts/setup_once.sh"
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  exec ".venv/bin/yt2avsr" process-both-sources --config configs/default.yaml
fi

exec ".venv/bin/yt2avsr" "$@"
