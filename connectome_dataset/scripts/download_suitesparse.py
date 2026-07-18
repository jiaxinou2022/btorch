#!/usr/bin/env python3
"""Download SuiteSparse Matrix Collection groups as Matrix Market archives.

Examples:
    python scripts/download_suitesparse.py --groups Arenas Newman
    python scripts/download_suitesparse.py --groups DIMACS10 --max-nnz 1000000
    python scripts/download_suitesparse.py --groups DIMACS10 --max-nnz 0 --dvc
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent

BASE_URL = "https://sparse.tamu.edu"
DEFAULT_GROUPS = ("Arenas", "Newman", "DIMACS10")
OUT_DIR = _REPO / "data" / "external" / "suitesparse"


@dataclass(frozen=True)
class MatrixInfo:
    group: str
    name: str
    rows: int
    cols: int
    nnz: int
    kind: str
    date: str
    url: str
    path: str


def _clean_cell(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _int_cell(text: str) -> int:
    return int(_clean_cell(text).replace(",", ""))


def scrape_group(group: str, timeout: int = 60) -> list[MatrixInfo]:
    """Scrape one SuiteSparse group page for Matrix Market archive links."""
    url = f"{BASE_URL}/{group}?per_page=All"
    print(f"[SCRAPE] {url}")
    raw = urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8")
    rows = re.findall(r"<tr>\s*(.*?)\s*</tr>", raw, flags=re.S)
    matrices: list[MatrixInfo] = []

    for row in rows:
        if f"/MM/{group}/" not in row:
            continue
        name_match = re.search(r"<td class='column-name'>.*?>([^<>]+)</a>", row, re.S)
        group_match = re.search(r"<td class='column-group'>.*?>([^<>]+)</a>", row, re.S)
        rows_match = re.search(r"<td class='column-num_rows'>(.*?)</td>", row, re.S)
        cols_match = re.search(r"<td class='column-num_cols'>(.*?)</td>", row, re.S)
        nnz_match = re.search(r"<td class='column-nonzeros'>(.*?)</td>", row, re.S)
        kind_match = re.search(r"<td class='column-kind[^']*'>(.*?)</td>", row, re.S)
        date_match = re.search(r"<td class='column-date[^']*'>(.*?)</td>", row, re.S)
        url_match = re.search(r'href="([^"]+/MM/' + re.escape(group) + r'/[^"]+\.tar\.gz)"', row)
        if not all((name_match, group_match, rows_match, cols_match, nnz_match, url_match)):
            print(f"  [WARN] Could not parse a row in {group}; skipping")
            continue

        name = _clean_cell(name_match.group(1))
        matrix_group = _clean_cell(group_match.group(1))
        archive_path = f"{matrix_group}/{name}.tar.gz"
        matrices.append(
            MatrixInfo(
                group=matrix_group,
                name=name,
                rows=_int_cell(rows_match.group(1)),
                cols=_int_cell(cols_match.group(1)),
                nnz=_int_cell(nnz_match.group(1)),
                kind=_clean_cell(kind_match.group(1)) if kind_match else "",
                date=_clean_cell(date_match.group(1)) if date_match else "",
                url=html.unescape(url_match.group(1)),
                path=archive_path,
            )
        )

    return matrices


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    if dest.exists():
        print(f"  [SKIP] {dest}")
        return True

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  [DOWNLOAD] {url}")
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(dest)
        print(f"       -> {dest} ({dest.stat().st_size:,} bytes)")
        return True
    except Exception as exc:
        print(f"  [ERROR] {url}: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


def write_manifest(out_dir: Path, matrices: list[MatrixInfo]) -> None:
    """Merge the newly-selected matrices into any existing manifest (union by group/name),
    so downloading a new set adds to the collection instead of replacing it."""
    path = out_dir / "manifest.json"
    by_key: dict[tuple[str, str], dict] = {}
    if path.exists():
        for m in json.loads(path.read_text()).get("matrices", []):
            by_key[(m["group"], m["name"])] = m
    for m in matrices:
        by_key[(m.group, m.name)] = asdict(m)
    manifest = {
        "source": "SuiteSparse Matrix Collection",
        "source_url": BASE_URL,
        "format": "Matrix Market tar.gz",
        "matrices": sorted(by_key.values(), key=lambda m: (m["group"], m["name"])),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[MANIFEST] {path} ({len(manifest['matrices'])} matrices total)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SuiteSparse Matrix Market archives")
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument(
        "--matrices", nargs="+", default=None,
        help="Only download these matrix names (across the scraped groups); curated-set mode.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--max-nnz",
        type=int,
        default=1_000_000,
        help="Skip matrices above this nnz; use 0 to disable the guard.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit downloads per group; 0 means no limit.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dvc", action="store_true", help="Run `dvc add` on the output directory.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected: list[MatrixInfo] = []

    wanted = set(args.matrices) if args.matrices else None
    for group in args.groups:
        matrices = scrape_group(group)
        if wanted is not None:
            matrices = [m for m in matrices if m.name in wanted]
        if args.max_nnz > 0:
            matrices = [m for m in matrices if m.nnz <= args.max_nnz]
        matrices = sorted(matrices, key=lambda m: (m.group, m.name))
        if args.limit > 0:
            matrices = matrices[: args.limit]
        print(f"[{group}] selected {len(matrices)} matrices")

        for matrix in matrices:
            dest = args.out_dir / matrix.path
            selected.append(matrix)
            if args.dry_run:
                print(f"  [DRY] {matrix.group}/{matrix.name} nnz={matrix.nnz:,} url={matrix.url}")
                continue
            download_file(matrix.url, dest)

    write_manifest(args.out_dir, selected)
    if args.dvc and not args.dry_run:
        subprocess.run(["dvc", "add", str(args.out_dir)], check=True)


if __name__ == "__main__":
    main()
