#!/usr/bin/env bash
set -euo pipefail

target=/homes/lzhang/.bashrc
backup=/homes/lzhang/.bashrc.codex-backup-20260812
guard='case $- in *i*) ;; *) return ;; esac'

cp -p "$target" "$backup"
if [[ "$(head -n 1 "$target")" != "$guard" ]]; then
  sed -i "1i $guard" "$target"
fi
head -n 3 "$target"
