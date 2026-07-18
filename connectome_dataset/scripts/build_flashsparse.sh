#!/bin/bash
# Build FlashSparse's FS_SpMM + FS_Block torch CUDA extensions for the local GPU arch.
# They compile for Blackwell (sm_120) with TORCH_CUDA_ARCH_LIST=12.0. The .so files are
# left in-place; the provider adds their dirs to sys.path (or set FLASHSPARSE_ROOT).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="external/FlashSparse/FlashSparse"
[ -d "$ROOT" ] || { echo "FlashSparse not fetched; run scripts/fetch_external.sh"; exit 1; }

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
MM="${MICROMAMBA:-micromamba}"; ENV="${ENV:-ml-py312}"

"$MM" run -n "$ENV" python - "$ROOT" <<'PY'
import os, sys, subprocess
root = sys.argv[1]
for d, name, srcs in [
    ("Block", "FS_Block", ["./example.cpp"]),
    ("SpMM",  "FS_SpMM",  ["./src/benchmark.cpp", "./src/spmmKernel.cu"]),
]:
    code = (
        "from setuptools import setup\n"
        "from torch.utils.cpp_extension import BuildExtension, CUDAExtension\n"
        f"setup(name={name!r}, script_args=['build_ext','--inplace'],\n"
        f"      ext_modules=[CUDAExtension(name={name!r}, sources={srcs!r})],\n"
        "      cmdclass={'build_ext': BuildExtension})\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=os.path.join(root, d), check=True)
    print(f"built {name} in {d}")
PY
echo "done. FS_SpMM.*.so and FS_Block.*.so are in $ROOT/{SpMM,Block}"
