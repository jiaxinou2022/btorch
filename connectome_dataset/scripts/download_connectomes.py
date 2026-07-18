#!/usr/bin/env python3
"""
Download connectome graphs from NeuroData (https://neurodata.io/project/connectomes)
into species-organized subfolders.

Source: S3 bucket https://s3.amazonaws.com/connectome-graphs/
Format: GraphML
"""

import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Base directory for downloads — graphml/ inside the repo's data/ folder
BASE_DIR = _REPO / "data" / "graphml"

# All datasets from https://neurodata.io/project/connectomes
# URL pattern: https://s3.amazonaws.com/connectome-graphs/{species}/{dataset}.graphml
DATASETS = {
    "cat": [
        "mixed.species_brain_1",
    ],
    "fly": [
        "drosophila_medulla_1",
    ],
    "macaque": [
        "rhesus_brain_1",
        "rhesus_brain_2",
        "rhesus_cerebral.cortex_1",
        "rhesus_interareal.cortical.network_2",
    ],
    "mouse": [
        "mouse_brain_1",
        "mouse_retina_1",
        "kasthuri_graph_v4",
        "mouse_visual.cortex_1",
        "mouse_visual.cortex_2",
    ],
    "rat": [
        "rattus.norvegicus_brain_1",
        "rattus.norvegicus_brain_2",
        "rattus.norvegicus_brain_3",
    ],
    "worm": [
        "c.elegans_neural.male_1",
        "c.elegans.herm_pharynx_1",
        "p.pacificus_neural.synaptic_1",
        "p.pacificus_neural.synaptic_2",
    ],
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file with simple progress."""
    if dest.exists():
        print(f"  [SKIP] Already exists: {dest}")
        return True

    try:
        print(f"  [DOWNLOAD] {url}")
        print(f"       -> {dest}")
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        print(f"       Done ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(len(v) for v in DATASETS.values())
    success = 0

    print(f"Downloading {total} connectome graphs to: {BASE_DIR}\n")

    for species, names in DATASETS.items():
        species_dir = BASE_DIR / species  # e.g. data/graphml/mouse/
        species_dir.mkdir(exist_ok=True)
        print(f"[{species.upper()}] -> {species_dir}/")

        for name in names:
            url = f"https://s3.amazonaws.com/connectome-graphs/{species}/{name}.graphml"
            dest = species_dir / f"{name}.graphml"
            if download_file(url, dest):
                success += 1
            print()

    print("=" * 50)
    print(f"Finished: {success}/{total} files downloaded.")
    print(f"Location: {BASE_DIR}")
    if success < total:
        print("Some downloads failed. Re-run the script to retry.")


if __name__ == "__main__":
    main()
