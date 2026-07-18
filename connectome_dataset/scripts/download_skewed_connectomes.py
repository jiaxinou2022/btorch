#!/usr/bin/env python3
"""
Download connectome graphs from Netzschleuder (networks.skewed.de).
Tagged: Connectome

Source: https://networks.skewed.de/?tags=Connectome
Base URL: https://networks.skewed.de/net/{dataset}/files/{filename}
"""

import urllib.request
from pathlib import Path
import re

_REPO = Path(__file__).resolve().parent.parent

BASE_DIR = _REPO / "data" / "skewed"
BASE_URL = "https://networks.skewed.de"

# Datasets tagged "Connectome" on networks.skewed.de
DATASETS = {
    "budapest_connectome": [
        # All resolutions and sexes
        "all_20k.gt.zst", "all_20k.csv.zip", "all_20k.gml.zst", "all_20k.xml.zst",
        "all_200k.gt.zst", "all_200k.csv.zip", "all_200k.gml.zst", "all_200k.xml.zst",
        "all_1m.gt.zst", "all_1m.csv.zip", "all_1m.gml.zst", "all_1m.xml.zst",
        "female_20k.gt.zst", "female_20k.csv.zip", "female_20k.gml.zst", "female_20k.xml.zst",
        "female_200k.gt.zst", "female_200k.csv.zip", "female_200k.gml.zst", "female_200k.xml.zst",
        "female_1m.gt.zst", "female_1m.csv.zip", "female_1m.gml.zst", "female_1m.xml.zst",
        "male_20k.gt.zst", "male_20k.csv.zip", "male_20k.gml.zst", "male_20k.xml.zst",
        "male_200k.gt.zst", "male_200k.csv.zip", "male_200k.gml.zst", "male_200k.xml.zst",
        "male_1m.gt.zst", "male_1m.csv.zip", "male_1m.gml.zst", "male_1m.xml.zst",
    ],
    "celegans_2019": [
        # Base variants (not corrected/synapse)
        "hermaphrodite_chemical.gt.zst", "hermaphrodite_chemical.csv.zip", "hermaphrodite_chemical.gml.zst", "hermaphrodite_chemical.xml.zst",
        "hermaphrodite_gap_junction.gt.zst", "hermaphrodite_gap_junction.csv.zip", "hermaphrodite_gap_junction.gml.zst", "hermaphrodite_gap_junction.xml.zst",
        "male_chemical.gt.zst", "male_chemical.csv.zip", "male_chemical.gml.zst", "male_chemical.xml.zst",
        "male_gap_junction.gt.zst", "male_gap_junction.csv.zip", "male_gap_junction.gml.zst", "male_gap_junction.xml.zst",
    ],
    "celegansneural": [
        "celegansneural.gt.zst", "celegansneural.csv.zip", "celegansneural.gml.zst", "celegansneural.xml.zst",
    ],
    "cintestinalis": [
        "cintestinalis.gt.zst", "cintestinalis.csv.zip", "cintestinalis.gml.zst", "cintestinalis.xml.zst",
    ],
    "fly_hemibrain": [
        "fly_hemibrain.gt.zst", "fly_hemibrain.csv.zip", "fly_hemibrain.gml.zst", "fly_hemibrain.xml.zst",
    ],
    "fly_larva": [
        "fly_larva.gt.zst", "fly_larva.csv.zip", "fly_larva.gml.zst", "fly_larva.xml.zst",
    ],
    "macaque_neural": [
        "macaque_neural.gt.zst", "macaque_neural.csv.zip", "macaque_neural.gml.zst", "macaque_neural.xml.zst",
    ],
    # human_brains: 66,255 individual brain networks. We download 2 representatives only.
    "human_brains": [
        "BNU1_0025864_1_DTI_AAL.gt.zst", "BNU1_0025864_1_DTI_AAL.csv.zip",
        "BNU1_0025864_1_DTI_CPAC200.gt.zst", "BNU1_0025864_1_DTI_CPAC200.csv.zip",
    ],
}


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [SKIP] {dest.name}")
        return True
    try:
        print(f"  [DOWNLOAD] {url.split('/')[-1]}")
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        print(f"       Done ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        if dest.exists():
            dest.unlink()
        return False


def scrape_files(dataset: str):
    """Scrape the dataset page for available file links."""
    page_url = f"{BASE_URL}/net/{dataset}"
    try:
        html = urllib.request.urlopen(page_url, timeout=30).read().decode("utf-8")
        pattern = re.compile(rf'href="({dataset}/files/[^"]+)"')
        matches = pattern.findall(html)
        # Deduplicate and return full filenames
        seen = set()
        files = []
        for m in matches:
            fname = m.split("/")[-1]
            if fname not in seen:
                seen.add(fname)
                files.append(fname)
        return files
    except Exception as e:
        print(f"  [WARN] Could not scrape {dataset}: {e}")
        return []


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    success = 0

    for dataset, files in DATASETS.items():
        ds_dir = BASE_DIR / dataset
        ds_dir.mkdir(exist_ok=True)
        print(f"[{dataset}] -> {ds_dir}/")

        # For human_brains, note the limitation
        if dataset == "human_brains":
            print("  (Note: human_brains has ~66k files; downloading 2 representatives only)")

        for fname in files:
            url = f"{BASE_URL}/net/{dataset}/files/{fname}"
            dest = ds_dir / fname
            total += 1
            if download_file(url, dest):
                success += 1

        print()

    print("=" * 50)
    print(f"Finished: {success}/{total} files downloaded.")
    print(f"Location: {BASE_DIR}")
    if success < total:
        print("Some downloads failed. Re-run to retry.")


if __name__ == "__main__":
    main()
