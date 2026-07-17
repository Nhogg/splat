# ZED Gaussian Splatting

Tools for extracting ZED SVO recordings, generating metric camera poses and
depth, training Gaussian splats with Nerfstudio/gsplat, and rendering results.

## Requirements

- Linux with Python 3.11
- An NVIDIA GPU and working NVIDIA driver
- ZED SDK 5.4 installed on the machine
- No `sudo` access is required for the Python environments

The ZED SDK itself is a system prerequisite. Ask an administrator to install it
if `/usr/local/zed` is unavailable.

## Install

```bash
git clone <repository-url> ~/splat
cd ~/splat
```

### ZED and reconstruction environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install requests opencv-python pycolmap==4.1.0
python get_python_api.py
```

`get_python_api.py` detects the installed ZED SDK and Python version, downloads
the matching PyZED wheel, and installs it into `.venv`. Verify it with:

```bash
python -c "import pyzed.sl, pycolmap, cv2; print('ZED environment ready')"
```

### Nerfstudio and gsplat environment

```bash
cd ~/splat
python3.11 -m venv .venv-ns
source .venv-ns/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install nerfstudio==1.1.5 gsplat==1.4.0
python -m pip install nvidia-cuda-nvcc==13.0.88
```

The CUDA compiler is installed inside the virtual environment, so compiling
gsplat for newer GPUs such as the RTX 5090 does not require a system CUDA
toolkit or `sudo`.

Set the runtime environment:

```bash
source ~/splat/.venv-ns/bin/activate
export CUDA_HOME="$HOME/splat/.venv-ns/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$HOME/splat/.cache/torch_extensions_cu130"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MAX_JOBS=4
```

The first gsplat run compiles and caches its CUDA kernels under
`~/splat/.cache/torch_extensions_cu130`. Verify the installation with:

```bash
python -c "import torch, gsplat; print(torch.cuda.get_device_name()); print(torch.__version__)"
```

## Render a trained splat

Pass a Nerfstudio training `config.yml` and output filename to the rendering
script:

```bash
cd ~/splat
./render_splat.sh \
  outputs/capture2-zed-depth/splatfacto/<run-timestamp>/config.yml \
  renders/room.mp4
```

The script activates `.venv-ns`, loads the checkpoint associated with the
configuration, and renders an interpolated camera path to an MP4.

An exported `splat.ply` alone cannot be rendered by this script because
Nerfstudio also needs the configuration and checkpoint. Use a standalone
Gaussian PLY viewer when only the exported PLY is available.
