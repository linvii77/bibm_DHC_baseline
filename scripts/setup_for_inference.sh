#!/bin/bash
# Link weights/ into logs/ layout expected by code/test.py
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for fold in 1 2 3; do
  src="weights/fold${fold}/ckpts/best_model.pth"
  dst="logs/Task_synapse_20p/dhc/fold${fold}/ckpts/best_model.pth"
  if [ ! -f "$src" ]; then
    echo "Missing $src — download weights first (see REPRODUCE.md)"
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  if [ ! -f "$dst" ]; then
    ln -sf "$(pwd)/$src" "$dst"
    echo "Linked $dst -> $src"
  fi
done

echo "Ready for: python code/test.py --task synapse --exp Task_synapse_20p/dhc/fold1 -g 0 --cps AB"
