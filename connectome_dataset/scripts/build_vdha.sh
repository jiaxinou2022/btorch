#!/bin/bash
# Build the VDHA SpMSpV kernel into cpp/build-vdha/libconnectome_vdha.so for the local GPU
# arch (Blackwell sm_120 by default). Requires a CUDA toolkit / GPU node (no GPU on the
# login node — run this via sbatch on the debug partition). The out-of-tree provider
# (examples/connectome_bench_vdha) ctypes-loads the resulting .so.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${CUDA_ARCH:-120}"          # sm_120 = RTX 5090 (Blackwell)
OUT="cpp/build-vdha"
MM="${MICROMAMBA:-micromamba}"; ENV="${ENV:-ml-py312}"
DTYPES="${VDHA_DTYPES:-float half}"
mkdir -p "$OUT"

build_one() {  # value-type  suffix
  echo "--- building $2 (VDHA_VAL=$1) ---"
  "$MM" run -n "$ENV" nvcc -O3 -std=c++14 -Xcompiler -fPIC -shared \
      -gencode "arch=compute_${ARCH},code=sm_${ARCH}" -DVDHA_VAL="$1" \
      cpp/baselines/vdha/vdha.cu -o "$OUT/libconnectome_vdha_$2.so" \
    && echo "built $OUT/libconnectome_vdha_$2.so (sm_${ARCH})" || echo "[warn] $2 build failed"
}

for dt in $DTYPES; do
  case "$dt" in
    float) build_one float  fp32 ;;
    half)  build_one __half fp16 ;;
    *) echo "[warn] unknown VDHA_DTYPES entry '$dt'";;
  esac
done

# Back-compat default symlink -> fp32.
[ -f "$OUT/libconnectome_vdha_fp32.so" ] && ln -sf libconnectome_vdha_fp32.so "$OUT/libconnectome_vdha.so"
