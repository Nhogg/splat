#!/usr/bin/env bash
set -euo pipefail

cd /home/nhogg/splat
source .venv-ns/bin/activate
export CUDA_HOME=/home/nhogg/splat/.venv-ns/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=/home/nhogg/splat/.cache/torch_extensions_cu130
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MAX_JOBS=4

ns-train splatfacto-big \
  --experiment-name capture2-zed-stereo-max \
  --output-dir outputs \
  --max-num-iterations 80000 \
  --steps-per-save 10000 \
  --vis tensorboard \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.datamanager.cache-images-type uint8 \
  --pipeline.datamanager.max-thread-workers 4 \
  --pipeline.datamanager.train-cameras-sampling-strategy fps \
  --pipeline.datamanager.fps-reset-every 1692 \
  --pipeline.model.enable-collider False \
  --pipeline.model.camera-optimizer.mode off \
  --pipeline.model.use-bilateral-grid True \
  --pipeline.model.use-scale-regularization True \
  --pipeline.model.densify-grad-thresh 0.0003 \
  --pipeline.model.stop-split-at 50000 \
  --pipeline.model.rasterize-mode antialiased \
  nerfstudio-data \
  --data datasets/capture2-zed-stereo \
  --downscale-factor 1 \
  --orientation-method none \
  --center-method none \
  --auto-scale-poses False \
  --train-split-fraction 0.99
