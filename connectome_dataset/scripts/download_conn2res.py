#!/usr/bin/env python3
"""
Download connectivity data from the conn2res toolbox Zenodo repository.

Source: Suárez et al. (2024) Connectome-based reservoir computing with the conn2res toolbox.
        PLOS Computational Biology. DOI: 10.1371/journal.pcbi.1011370
Zenodo: https://zenodo.org/records/10205004

Downloads data.zip and extracts the following connectomes into data/conn2res/:
  drosophila/   — Drosophila brain regions (49 nodes, Chiang et al. 2011)
  macaque_modha/ — Macaque long-range pathways (242 nodes, Modha & Singh 2010)
  human/         — 6 consensus human structural connectomes (1015 parcels, ~25k edges each)
  mouse/         — Mouse acronym lookup (conn.csv not in public release)

Note: The large individual-subject connectivity.npy (551 MB, 70 subjects) is NOT downloaded.
      Only the group-level consensus matrices (consensus_0..5.npy, ~8 MB each) are kept.
"""

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

ZENODO_URL = "https://zenodo.org/api/records/10205004/files/data.zip/content"
DEST_DIR = _REPO / "data" / "conn2res"

# Files to SKIP (too large or not needed)
SKIP_FILES = {"connectivity.npy"}


def download_and_extract() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading conn2res data from Zenodo (~19 MB)...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        urllib.request.urlretrieve(ZENODO_URL, tmp_path)
        size_mb = tmp_path.stat().st_size / 1024 / 1024
        print(f"Downloaded {size_mb:.1f} MB → extracting...")

        with zipfile.ZipFile(tmp_path) as zf:
            members = zf.namelist()
            extracted = 0
            skipped = 0
            for member in members:
                fname = Path(member).name
                if fname in SKIP_FILES:
                    print(f"  [SKIP large] {member}")
                    skipped += 1
                    continue
                # Strip the leading "data/" from the zip path
                relative = Path(member)
                parts = relative.parts
                if len(parts) > 1 and parts[0] == "data":
                    out_path = DEST_DIR / Path(*parts[1:])
                else:
                    out_path = DEST_DIR / relative

                if out_path.suffix == "":  # directory
                    out_path.mkdir(parents=True, exist_ok=True)
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists():
                    print(f"  [SKIP exists] {out_path.name}")
                    skipped += 1
                    continue

                with zf.open(member) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  {out_path.relative_to(DEST_DIR)}")
                extracted += 1

        print(f"\nDone: {extracted} files extracted, {skipped} skipped.")
        print(f"Location: {DEST_DIR}")

    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    download_and_extract()
