#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELL_RC="${HOME}/.zshrc"
MARKER_START="# >>> youtube-to-autoavsr >>>"
MARKER_END="# <<< youtube-to-autoavsr <<<"

touch "$SHELL_RC"

if grep -qF "$MARKER_START" "$SHELL_RC"; then
  echo "Terminal komutu zaten ekli: ytavsr"
  exit 0
fi

cat >> "$SHELL_RC" <<EOF

$MARKER_START
alias ytavsr='$ROOT/scripts/run.sh'
alias ytavsr-setup='$ROOT/scripts/setup_once.sh'
$MARKER_END
EOF

echo "Eklendi:"
echo "  ytavsr       -> kaynak dosyalarini isler"
echo "  ytavsr-setup -> kurulumu bir kere hazirlar"
echo "Yeni terminal ac veya calistir: source ~/.zshrc"
