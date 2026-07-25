#!/bin/bash
# Fetch the vendored external kernel repos used by the optional CUDA baselines.
# They are gitignored (large third-party trees); run this once per checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
clone() {  # repo_url  dir  [extra git args...]
  local url="$1" dir="$2"; shift 2
  if [ -d "external/$dir/.git" ] && [ -n "$(ls -A "external/$dir" 2>/dev/null | grep -v '^\.git$')" ]; then
    echo "external/$dir present"; return 0
  fi
  echo "cloning $dir ..."
  # per-repo failure must not abort the caller — one repo may be unreachable
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$@" "$url" "external/$dir" \
    || { echo "[warn] failed to clone $dir; skipping"; rm -rf "external/$dir"; return 0; }
}
clone https://github.com/google-research/sputnik.git sputnik --single-branch
# FlashSparse bundles large data; grab just the kernel/cmake via a blobless sparse checkout.
clone https://github.com/ParCIS/FlashSparse.git FlashSparse --single-branch --branch main --filter=blob:none
clone https://github.com/HPMLL/DTC-SpMM_ASPLOS24.git DTC-SpMM --single-branch --branch main --filter=blob:none

# MH-SpGEMM is a local checkout (not a public URL here); mirror its sources in (no .git /
# build artifacts). Override the source with MH_SPGEMM_ROOT.
MH_SRC="${MH_SPGEMM_ROOT:-$HOME/src/MH-SpGEMM}"
if [ ! -d external/MH-SpGEMM/inc ] && [ -d "$MH_SRC/inc" ]; then
  echo "copying MH-SpGEMM from $MH_SRC ..."
  mkdir -p external/MH-SpGEMM
  rsync -a --exclude='.git' --exclude='obj' --exclude='spgemm' --exclude='*.o' \
    "$MH_SRC"/ external/MH-SpGEMM/ 2>/dev/null \
    || cp -r "$MH_SRC"/inc "$MH_SRC"/src external/MH-SpGEMM/
else
  echo "external/MH-SpGEMM present or source $MH_SRC missing"
fi
echo "done. build with: cmake -B cpp/build-cuda -S cpp -DCONNECTOME_BENCH_BUILD_SPUTNIK=ON"
