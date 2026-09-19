#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/yt2avsr" ]]; then
  echo ".venv hazir degil. Once calistir:"
  echo "$ROOT/scripts/setup_once.sh"
  exit 1
fi

KNOWN_COMMANDS="^(process|process-playlist|process-local|process-sources|process-both-sources|check-downloader|setup-external|setup-retinaface|setup-whisper|push-data|pull-data|sync-processed|manifest|inspect)$"

if [[ "$#" -eq 0 ]]; then
  exec ".venv/bin/yt2avsr" process-both-sources --config configs/default.yaml
elif [[ "$1" =~ $KNOWN_COMMANDS ]]; then
  exec ".venv/bin/yt2avsr" "$@"
else
  exec ".venv/bin/yt2avsr" process-both-sources --config configs/default.yaml "$@"
fi

