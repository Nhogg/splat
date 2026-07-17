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
  --experiment-name capture2-half-poseopt \
  --output-dir outputs \
  --max-num-iterations 60000 \
  --steps-per-save 5000 \
  --vis tensorboard \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.datamanager.cache-images-type uint8 \
  --pipeline.datamanager.max-thread-workers 4 \
  --pipeline.datamanager.train-cameras-sampling-strategy fps \
  --pipeline.datamanager.fps-reset-every 736 \
  --pipeline.model.enable-collider False \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  --pipeline.model.camera-optimizer.trans-l2-penalty 0.01 \
  --pipeline.model.camera-optimizer.rot-l2-penalty 0.001 \
  --pipeline.model.use-scale-regularization True \
  --pipeline.model.stop-split-at 40000 \
  colmap \
  --data datasets/capture2 \
  --colmap-path sparse/0 \
  --downscale-factor 2
