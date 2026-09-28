#!/usr/bin/env bash
# Optional, isolated shape-only experiment. Does not modify the main .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
repo=vendor/Hunyuan3D-2
revision=f8db63096c8282cb27354314d896feba5ba6ff8a
mkdir -p vendor
if [[ ! -d "$repo" ]]; then
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$repo"
    git -C "$repo" checkout "$revision"
fi
if [[ "$(git -C "$repo" rev-parse HEAD)" != "$revision" ]]; then
    echo "Existing Hunyuan checkout differs from the tested revision: $revision" >&2
    exit 1
fi
python3 -m venv vendor/hunyuan-venv
py=vendor/hunyuan-venv/bin/python
"$py" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$py" -m pip install numpy==2.2.6 diffusers==0.32.2 transformers==4.49.0 \
    huggingface-hub==0.34.4 accelerate==1.6.0 einops omegaconf trimesh \
    pymeshlab scikit-image opencv-python-headless safetensors
"$py" scripts/fetch_hunyuan_models.py
"$py" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name()); print(torch.ones(4,device="cuda").sum().item())'
