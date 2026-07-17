#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CONFIG_YML [OUTPUT_MP4]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source .venv-ns/bin/activate

export CUDA_HOME="$SCRIPT_DIR/.venv-ns/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SCRIPT_DIR/.cache/torch_extensions_cu130"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MAX_JOBS=4

config="$1"
output="${2:-renders/interpolated.mp4}"
mkdir -p "$(dirname "$output")"

ns-render interpolate \
  --load-config "$config" \
  --output-path "$output" \
  --pose-source train \
  --order-poses False \
  --interpolation-steps 2 \
  --frame-rate 30 \
  --rendered-output-names rgb \
  --image-format png \
  --output-format video
