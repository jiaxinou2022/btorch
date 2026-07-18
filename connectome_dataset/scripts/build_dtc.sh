#!/bin/bash
# Build the DTC-SpMM torch CUDA extension for the local GPU arch (sm_120 by default).
# DTC-SpMM depends on Sputnik + glog; we sidestep the full builds by compiling Sputnik's
# SpMM translation unit alongside (via cpp/third_party/sputnik_shim) and standing in a
# minimal glog (cpp/third_party/glog_shim). A small patch adds the weighted entry point.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
ROOT="external/DTC-SpMM/DTC-SpMM"
[ -d "$ROOT" ] || { echo "DTC-SpMM not fetched; run scripts/fetch_external.sh"; exit 1; }
[ -d "external/sputnik" ] || { echo "Sputnik not fetched; run scripts/fetch_external.sh"; exit 1; }

# Apply the weighted-SpMM patch (idempotent: skip if already applied).
if ! grep -q "run_DTCSpMM_weighted" "$ROOT/DTCSpMM.cpp"; then
  echo "applying dtc_weighted.patch ..."
  git -C external/DTC-SpMM apply "$REPO/scripts/patches/dtc_weighted.patch"
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
MM="${MICROMAMBA:-micromamba}"; ENV="${ENV:-ml-py312}"

ABS_ROOT="$REPO/$ROOT"
( cd "$ABS_ROOT" && "$MM" run -n "$ENV" python - "$REPO" <<'PY'
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
repo = sys.argv[1]
sys.argv = [sys.argv[0], "build_ext", "--inplace"]  # -> .so lands here in ROOT
setup(name="DTCSpMM", ext_modules=[CUDAExtension("DTCSpMM",
    ["./DTCSpMM.cpp", "./DTCSpMM_kernel.cu",
     f"{repo}/cpp/third_party/sputnik_shim/sputnik_cuda_spmm.cu"],
    include_dirs=[f"{repo}/external/sputnik", f"{repo}/cpp/third_party/glog_shim"],
    extra_compile_args={"cxx": ["-DGLOG_USE_GLOG_EXPORT"],
        "nvcc": ["--expt-relaxed-constexpr", "-DGLOG_USE_GLOG_EXPORT", "-std=c++17"]})],
    cmdclass={"build_ext": BuildExtension})
PY
)
echo "done. DTCSpMM.*.so in $ROOT"
