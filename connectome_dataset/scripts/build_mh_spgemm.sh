#!/bin/bash
# Build MH-SpGEMM into cpp/build-mh_spgemm/libconnectome_mh_spgemm_<prec>.so for the local GPU
# arch (Blackwell sm_120 by default), one shared lib per value type. Needs the vendored
# sources (scripts/fetch_external.sh copies them from ~/src/MH-SpGEMM). Requires a CUDA
# toolkit / GPU node — run via sbatch on debug. The provider loads the lib matching the
# requested precision.
#   MH_DTYPES="float half"   # default; add "double" for an fp64 build
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${CUDA_ARCH:-120}"          # sm_120 = RTX 5090 (Blackwell)
SRC="external/MH-SpGEMM"
OUT="cpp/build-mh_spgemm"
MM="${MICROMAMBA:-micromamba}"; ENV="${ENV:-ml-py312}"
# fp32 only by default: the vendored kernel's __half path aborts on some graphs (fp64-centric
# shared-hash/atomics), so fp16 is not advertised. Build it for experimentation with MH_DTYPES="float half".
DTYPES="${MH_DTYPES:-float}"
[ -d "$SRC/inc" ] || { echo "MH-SpGEMM not vendored; run scripts/fetch_external.sh"; exit 1; }
mkdir -p "$OUT"

# Silence the vendored kernel's per-phase debug prints (they spam thousands of lines).
sed -i '/printf(.*global row/d' "$SRC/inc/MH_spgemm.cuh"
# Make the value type a build parameter: guard the hardcoded `#define VALUE_TYPE double` so
# a command-line -DVALUE_TYPE=... wins. Idempotent.
grep -q "ifndef VALUE_TYPE" "$SRC/inc/common.h" \
  || sed -i 's/^#define VALUE_TYPE double/#ifndef VALUE_TYPE\n#define VALUE_TYPE double\n#endif/' "$SRC/inc/common.h"
# CSR::operator== calls std::fabs on VALUE_TYPE; that is ambiguous for __half (fp16 build).
# Cast to double — harmless for float/double, and this checker is never called by the wrapper.
grep -q 'fabs((double)' "$SRC/src/CSR.cu" || sed -i \
  -e 's/std::fabs(val\[j\] - C_tmp.val\[j\])/std::fabs((double)(val[j] - C_tmp.val[j]))/g' \
  -e 's/std::fabs(val\[j\])/std::fabs((double)val[j])/g' \
  "$SRC/src/CSR.cu"

srcs=(cpp/baselines/mh_spgemm/mh_spgemm_c.cu "$SRC/src/CSR.cu" "$SRC/src/Tool.cu" "$SRC/src/Timing.cpp" "$SRC/src/utils.cpp")

build_one() {  # dtype  suffix  extra-nvcc-args...
  local dtype="$1" sfx="$2"; shift 2
  echo "--- building $sfx (VALUE_TYPE=$dtype) ---"
  "$MM" run -n "$ENV" nvcc -O3 -w -std=c++14 \
      -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
      -Xcompiler -fPIC -Xcompiler -fopenmp -shared \
      -I"$SRC/inc" -DVALUE_TYPE="$dtype" "$@" \
      "${srcs[@]}" -o "$OUT/libconnectome_mh_spgemm_${sfx}.so" \
    && echo "built $OUT/libconnectome_mh_spgemm_${sfx}.so (sm_${ARCH})" \
    || echo "[warn] $sfx build failed"
}

for dt in $DTYPES; do
  case "$dt" in
    float)  build_one float  fp32 ;;
    double) build_one double fp64 ;;
    half)   build_one __half fp16 -include cuda_fp16.h ;;  # force fp16 header into every TU
    *) echo "[warn] unknown MH_DTYPES entry '$dt'";;
  esac
done

# Back-compat default symlink -> fp32.
[ -f "$OUT/libconnectome_mh_spgemm_fp32.so" ] && ln -sf libconnectome_mh_spgemm_fp32.so "$OUT/libconnectome_mh_spgemm.so"
