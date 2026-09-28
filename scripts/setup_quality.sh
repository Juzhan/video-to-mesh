#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
./scripts/setup_local.sh
# CPU inference keeps this optional segmentation model independent of CUDA toolkit versions.
.venv/bin/pip install --timeout 120 torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
if [ ! -d vendor/sam2/.git ]; then
  git clone https://github.com/facebookresearch/sam2.git vendor/sam2
  git -C vendor/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
fi
if [ "$(git -C vendor/sam2 rev-parse HEAD)" != "2b90b9f5ceec907a1c18123530e92e794ad901a4" ]; then
  echo 'SAM 2 checkout differs from the validated revision; keep it untouched and resolve explicitly.' >&2
  exit 1
fi
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation -e vendor/sam2
mkdir -p models
if [ ! -f models/sam2.1_hiera_tiny.pt ]; then
  curl -fL --retry 2 --max-time 180 \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
    -o models/sam2.1_hiera_tiny.pt.partial
  mv models/sam2.1_hiera_tiny.pt.partial models/sam2.1_hiera_tiny.pt
fi
echo 'SAM 2.1 ready. All generated geometry still requires quality review.'
echo '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69  models/sam2.1_hiera_tiny.pt' | sha256sum --check
