#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Detect shell rc file
SHELL_TARGETS=()
if [[ -n "${ZSH_VERSION:-}" ]] || [[ "${SHELL:-}" =~ zsh$ ]] || [[ -f "${HOME}/.zshrc" ]]; then
  SHELL_TARGETS+=("${HOME}/.zshrc")
fi
if [[ -f "${HOME}/.bashrc" ]] || [[ "${SHELL:-}" =~ bash$ ]]; then
  SHELL_TARGETS+=("${HOME}/.bashrc")
fi
if [[ ${#SHELL_TARGETS[@]} -eq 0 ]]; then
  SHELL_TARGETS+=("${HOME}/.zshrc")
fi

MARKER_START="# >>> youtube-to-autoavsr >>>"
MARKER_END="# <<< youtube-to-autoavsr <<<"

for SHELL_RC in "${SHELL_TARGETS[@]}"; do
  touch "$SHELL_RC"
  if grep -qF "$MARKER_START" "$SHELL_RC"; then
    echo "Terminal komutu zaten ekli: $SHELL_RC"
    continue
  fi

  cat >> "$SHELL_RC" <<EOF

$MARKER_START
alias ytavsr='$ROOT/scripts/run.sh'
alias ytavsr-setup='$ROOT/scripts/setup_once.sh'
$MARKER_END
EOF
  echo "Eklendi ($SHELL_RC):"
  echo "  ytavsr       -> kaynak dosyalarini isler"
  echo "  ytavsr-setup -> kurulumu bir kere hazirlar"
done

echo "Yeni terminal ac veya calistir: source ~/.zshrc (veya source ~/.bashrc)"
