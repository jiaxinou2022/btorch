#!/usr/bin/env python3
"""
Download Fish1 (zebrafish larva) connectome graph and node data.
Source: https://fish1-release.storage.googleapis.com

This downloads the TEN_analysis.zip which contains:
  - agglomerated_segments_and_soma_ids.csv  (neuron segments)
  - incoming_synapses.csv                   (incoming synapse lists per neuron)
  - outgoing_synapses.csv                   (outgoing synapse lists per neuron)
  - synapse_sizes.csv                       (synapse size metadata)
  - DMVIDs.csv, DMV_InputsIDs.csv           (cell type annotations)
  - TEN_cats.csv                            (TEN category annotations)
  - README.txt                              (documentation)

Also downloads HMI_analysis.zip which contains:
  - cave_somas_in_big_box.csv               (soma positions and metadata)
  - em_zfish1_dataframe.xlsx                (neuron type spreadsheet)

And the programmatic access notebook.
"""

import urllib.request
from pathlib import Path
import zipfile

_REPO = Path(__file__).resolve().parent.parent

BASE_DIR = _REPO / "data" / "fish1"
BASE_URL = "https://storage.googleapis.com/fish1-release/paper_data"

FILES = {
    "TEN_analysis.zip": f"{BASE_URL}/TEN_analysis.zip",
    "HMI_analysis.zip": f"{BASE_URL}/HMI_analysis.zip",
    "ProgrammaticInteractionWithFish1Cave.ipynb": f"{BASE_URL}/ProgrammaticInteractionWithFish1Cave.ipynb",
}


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [SKIP] Already exists: {dest.name}")
        return True
    try:
        print(f"  [DOWNLOAD] {dest.name}")
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        print(f"       Done ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    success = 0

    for fname, url in FILES.items():
        dest = BASE_DIR / fname
        if download_file(url, dest):
            success += 1

    # Extract TEN_analysis.zip
    ten_zip = BASE_DIR / "TEN_analysis.zip"
    if ten_zip.exists():
        print("\n  [EXTRACT] TEN_analysis.zip")
        with zipfile.ZipFile(ten_zip, 'r') as z:
            z.extractall(BASE_DIR)
        print("       Done")

    # Extract HMI_analysis.zip (only CSVs and data files)
    hmi_zip = BASE_DIR / "HMI_analysis.zip"
    if hmi_zip.exists():
        print("\n  [EXTRACT] HMI_analysis.zip (data files only)")
        with zipfile.ZipFile(hmi_zip, 'r') as z:
            for member in z.namelist():
                if member.startswith("HMI_analysis/data/") or member == "HMI_analysis/README.md":
                    z.extract(member, BASE_DIR)
        print("       Done")

    print("\n" + "=" * 50)
    print(f"Finished: {success}/{len(FILES)} files downloaded.")
    print(f"Location: {BASE_DIR}")
    print("\nKey connectome files:")
    for p in sorted(BASE_DIR.rglob("*.csv")):
        print(f"  {p.relative_to(BASE_DIR)}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
