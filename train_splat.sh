#!/usr/bin/env bash
set -euo pipefail

cd /home/nhogg/splat
source .venv-ns/bin/activate

export CUDA_HOME=/home/nhogg/splat/.venv-ns/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=/home/nhogg/splat/.cache/torch_extensions_cu130
export MAX_JOBS=4

ns-train splatfacto-big \
  --experiment-name capture2-half \
  --output-dir outputs \
  --max-num-iterations 30000 \
  --steps-per-save 2000 \
  --vis tensorboard \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.datamanager.cache-images-type uint8 \
  colmap \
  --data datasets/capture2 \
  --colmap-path sparse/0 \
  --downscale-factor 2
