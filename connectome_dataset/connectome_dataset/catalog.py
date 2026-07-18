"""Graph catalog management (SuiteSparse-style).

Stores metadata about each graph in .cache/graph_catalog.json at the repo root.
That file is machine-local (absolute paths, computed stats) and gitignored.
Can be regenerated:

    micromamba run -n ml-py312 python -m connectome_dataset.catalog
    # or
    connectome-catalog   (after pip install -e .)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ── paths ─────────────────────────────────────────────────────────────────────
# connectome_dataset/catalog.py -> repo root
_REPO = Path(__file__).parent.parent
DATA_ROOT = Path(os.environ.get("CONNECTOME_DATA_ROOT", _REPO / "data"))
CONN2RES_DATA = DATA_ROOT / "conn2res"

CATALOG_DIR = _REPO / ".cache"
CATALOG_FILE = CATALOG_DIR / "graph_catalog.json"

MICE_COLUMN_V1_ROOT = DATA_ROOT / "external" / "mice_column_v1"
SUITESPARSE_ROOT = DATA_ROOT / "external" / "suitesparse"
MICRONS_MM3_PATH = DATA_ROOT / "external" / "microns" / "microns_mm3_connectome.h5"


def load_catalog() -> list[dict[str, Any]]:
    if not CATALOG_FILE.exists():
        return []
    with open(CATALOG_FILE) as f:
        return json.load(f)["graphs"]


def save_catalog(graphs: list[dict[str, Any]]) -> None:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_FILE, "w") as f:
        json.dump({"graphs": graphs}, f, indent=2)
    print(f"Saved {len(graphs)} graphs to {CATALOG_FILE}")


def find_graph(name: str) -> dict[str, Any] | None:
    for g in load_catalog():
        if g["name"] == name:
            return g
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Catalog generation
# ──────────────────────────────────────────────────────────────────────────────

def _make_entry(
    name: str,
    mat,
    source_path: str,
    source_format: str,
    species: str,
    region: str,
    tags: list[str],
    description: str = "",
    paper: str = "",
    doi: str = "",
    source_url: str = "",
    notes: str = "",
) -> dict[str, Any]:
    from connectome_dataset.graph_loader import graph_properties
    props = graph_properties(mat)
    entry = {
        "name": name,
        "description": description,
        "paper": paper,
        "doi": doi,
        "source_url": source_url,
        "source_path": source_path,
        "source_format": source_format,
        "species": species,
        "region": region,
        "tags": tags,
        "notes": notes,
    }
    entry.update(props)
    return entry


def _make_file_entry(
    name: str,
    source_path: str,
    source_format: str,
    species: str,
    region: str,
    tags: list[str],
    n_rows: int | None = None,
    n_cols: int | None = None,
    nnz: int | None = None,
    description: str = "",
    paper: str = "",
    doi: str = "",
    source_url: str = "",
    notes: str = "",
) -> dict[str, Any]:
    path = Path(source_path)
    entry = {
        "name": name,
        "description": description,
        "paper": paper,
        "doi": doi,
        "source_url": source_url,
        "source_path": source_path,
        "source_format": source_format,
        "species": species,
        "region": region,
        "tags": tags,
        "notes": notes,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": nnz,
        "density": None,
        "is_square": None,
        "is_symmetric": None,
        "val_min": None,
        "val_max": None,
        "val_mean": None,
        "dtype": "hdf5",
        "file_size_bytes": path.stat().st_size if path.exists() else None,
    }
    return entry


def build_catalog() -> None:
    from connectome_dataset.graph_loader import (
        load_graphml, load_csv_zip, load_mice_column_v1,
        load_conn2res_csv, load_conn2res_npy, load_suitesparse_mm,
    )

    graphs: list[dict[str, Any]] = []

    # ── graphml files (from NeuroData / networks.skewed.de) ──────────────────
    graphml_specs = [
        # (path_rel, name, species, region, tags, description, paper, doi, source_url)
        (
            "graphml/mouse/mouse_visual.cortex_1.graphml",
            "mouse_vis_ctx_1", "mouse", "visual_cortex",
            ["connectome", "graphml", "mouse", "cortex", "EM"],
            "Mouse visual cortex micro-connectome (sample 1), reconstructed by EM. "
            "Nodes are neurons; edges are synaptic contacts.",
            "Bock et al. (2011) Network anatomy and in vivo physiology of visual cortical neurons. "
            "Nature 471:177–182.",
            "10.1038/nature09802",
            "https://doi.org/10.1038/nature09802",
        ),
        (
            "graphml/mouse/mouse_visual.cortex_2.graphml",
            "mouse_vis_ctx_2", "mouse", "visual_cortex",
            ["connectome", "graphml", "mouse", "cortex", "EM"],
            "Mouse visual cortex micro-connectome (sample 2), reconstructed by EM.",
            "Bock et al. (2011) Network anatomy and in vivo physiology of visual cortical neurons. "
            "Nature 471:177–182.",
            "10.1038/nature09802",
            "https://doi.org/10.1038/nature09802",
        ),
        (
            "graphml/mouse/mouse_brain_1.graphml",
            "mouse_brain_1", "mouse", "whole_brain",
            ["connectome", "graphml", "mouse", "macro", "tract_tracing"],
            "Mouse whole-brain mesoscale connectome from the Allen Mouse Brain Connectivity Atlas. "
            "Nodes are brain regions; edges are axonal projections measured by EGFP tracer.",
            "Oh et al. (2014) A mesoscale connectome of the mouse brain. Nature 508:207–214.",
            "10.1038/nature13186",
            "https://doi.org/10.1038/nature13186",
        ),
        (
            "graphml/mouse/mouse_retina_1.graphml",
            "mouse_retina_1", "mouse", "retina",
            ["connectome", "graphml", "mouse", "retina", "EM"],
            "Mouse inner plexiform layer retinal connectome from EM reconstruction. "
            "Nodes are neurons; edges are synaptic contacts.",
            "Helmstaedter et al. (2013) Connectomic reconstruction of the inner plexiform layer "
            "in the mouse retina. Nature 500:168–174.",
            "10.1038/nature12346",
            "https://github.com/ericmjonas/circuitdata/tree/master/mouseretina",
        ),
        (
            "graphml/mouse/kasthuri_graph_v4.graphml",
            "kasthuri_v4", "mouse", "neocortex",
            ["connectome", "graphml", "mouse", "EM", "micro"],
            "Mouse somatosensory neocortex dense EM reconstruction (cylinder volume). "
            "Nodes are neurites; edges are synapses.",
            "Kasthuri et al. (2015) Saturated Reconstruction of a Volume of Neocortex. "
            "Cell 162:648–661.",
            "10.1016/j.cell.2015.06.054",
            "https://doi.org/10.1016/j.cell.2015.06.054",
        ),
        (
            "graphml/fly/drosophila_medulla_1.graphml",
            "drosophila_medulla_1", "fly", "medulla",
            ["connectome", "graphml", "fly", "optic_lobe", "EM"],
            "Drosophila optic lobe medulla connectome reconstructed by EM. "
            "Nodes are neurons; edges are synaptic contacts (presynaptic→postsynaptic).",
            "Takemura et al. (2013) A visual motion detection circuit suggested by Drosophila "
            "connectomics. Nature 500:175–181.",
            "10.1038/nature12450",
            "https://doi.org/10.1038/nature12450",
        ),
        (
            "graphml/worm/c.elegans.herm_pharynx_1.graphml",
            "celegans_herm_pharynx", "worm", "pharynx",
            ["connectome", "graphml", "c.elegans", "EM"],
            "C. elegans hermaphrodite pharyngeal nervous system, reconstructed by EM. "
            "Edge weights are synapse counts.",
            "White et al. (1986) The structure of the nervous system of the nematode "
            "Caenorhabditis elegans. Phil Trans R Soc B 314:1–340.",
            "10.1098/rstb.1986.0056",
            "https://github.com/ericmjonas/circuitdata/tree/master/celegans_herm",
        ),
        (
            "graphml/worm/c.elegans_neural.male_1.graphml",
            "celegans_male_neural", "worm", "whole_brain",
            ["connectome", "graphml", "c.elegans", "EM"],
            "C. elegans male nervous system complete connectome reconstructed by EM. "
            "Nodes are neurons; edges are chemical synapses.",
            "Jarrell et al. (2012) The Connectome of a Decision-Making Neural Network. "
            "Science 337:437–444.",
            "10.1126/science.1221762",
            "https://doi.org/10.1126/science.1221762",
        ),
        (
            "graphml/worm/p.pacificus_neural.synaptic_1.graphml",
            "ppacificus_synaptic_1", "worm", "whole_brain",
            ["connectome", "graphml", "p.pacificus", "EM"],
            "Pristionchus pacificus complete synaptic connectome (sample 1). "
            "Enables cross-species comparison with C. elegans.",
            "Bumbarger et al. (2013) System-wide Rewiring Underlies Behavioral Differences in "
            "Predatory and Bacterial-Feeding Nematodes. Cell 152:109–119.",
            "10.1016/j.cell.2012.12.013",
            "https://doi.org/10.1016/j.cell.2012.12.013",
        ),
        (
            "graphml/worm/p.pacificus_neural.synaptic_2.graphml",
            "ppacificus_synaptic_2", "worm", "whole_brain",
            ["connectome", "graphml", "p.pacificus", "EM"],
            "Pristionchus pacificus complete synaptic connectome (sample 2).",
            "Bumbarger et al. (2013) System-wide Rewiring Underlies Behavioral Differences in "
            "Predatory and Bacterial-Feeding Nematodes. Cell 152:109–119.",
            "10.1016/j.cell.2012.12.013",
            "https://doi.org/10.1016/j.cell.2012.12.013",
        ),
        (
            "graphml/cat/mixed.species_brain_1.graphml",
            "cat_brain_1", "cat", "whole_brain",
            ["connectome", "graphml", "cat", "macro", "tract_tracing"],
            "Cat whole-brain cortical connectivity from tract-tracing studies. "
            "Nodes are cortical areas; edges are axonal projections.",
            "Harriger et al. (2012) Rich Club Organization of Macaque Cerebral Cortex and Its "
            "Role in Network Communication. J Neurosci 32:12,929–12,935.",
            "10.1523/JNEUROSCI.1448-13.2013",
            "https://doi.org/10.1523/JNEUROSCI.1448-13.2013",
        ),
        (
            "graphml/macaque/rhesus_brain_1.graphml",
            "macaque_rhesus_brain_1", "macaque", "cerebral_cortex",
            ["connectome", "graphml", "macaque", "macro", "tract_tracing"],
            "Rhesus macaque cerebral cortex connectivity from tract-tracing (PLoS ONE dataset). "
            "Nodes are cortical areas; edges are axonal projections.",
            "Kötter & Stephan (2003) Network participation indices; compiled in: "
            "Harriger et al. (2012) PLoS ONE 7:e46497.",
            "10.1371/journal.pone.0046497",
            "https://doi.org/10.1371/journal.pone.0046497",
        ),
        (
            "graphml/macaque/rhesus_brain_2.graphml",
            "macaque_rhesus_brain_2", "macaque", "whole_brain",
            ["connectome", "graphml", "macaque", "macro", "retrograde_tracer"],
            "Rhesus macaque whole-brain cortical connectivity from retrograde tracer studies "
            "(CoRe-Nets database). Nodes are brain areas.",
            "Beul & Hilgetag (2014) Towards a 'canonical' agranular cortical microcircuit. "
            "J Comp Neurol 522:1480–1500.",
            "10.1002/cne.23458",
            "https://doi.org/10.1002/cne.23458",
        ),
        (
            "graphml/macaque/rhesus_cerebral.cortex_1.graphml",
            "macaque_rhesus_cortex_1", "macaque", "cerebral_cortex",
            ["connectome", "graphml", "macaque", "macro", "retrograde_tracer"],
            "Rhesus macaque cerebral cortex interareal connectivity from retrograde tracer "
            "(CoRe-Nets). Nodes are cortical areas.",
            "Markov et al. (2013) The role of long-range connections on the specificity of the "
            "macaque interareal cortical network. Cereb Cortex 23:254–272.",
            "10.1093/cercor/bhs270",
            "https://doi.org/10.1093/cercor/bhs270",
        ),
        (
            "graphml/macaque/rhesus_interareal.cortical.network_2.graphml",
            "macaque_interareal_2", "macaque", "cerebral_cortex",
            ["connectome", "graphml", "macaque", "macro", "retrograde_tracer"],
            "Rhesus macaque interareal cortical network (CoRe-Nets, 2013). "
            "High-resolution retrograde tracer dataset of cortical area connectivity.",
            "Markov et al. (2014) A Weighted and Directed Interareal Connectivity Matrix for "
            "Macaque Cerebral Cortex. Cereb Cortex 24:17–36.",
            "10.1073/pnas.1218972110",
            "https://doi.org/10.1073/pnas.1218972110",
        ),
        (
            "graphml/rat/rattus.norvegicus_brain_1.graphml",
            "rat_brain_1", "rat", "whole_brain",
            ["connectome", "graphml", "rat", "macro", "tract_tracing"],
            "Rattus norvegicus (rat) whole-brain connectivity from neuroanatomical experiments "
            "(USC Brancusi connectome, 2011). Nodes are brain regions.",
            "Bota et al. (2015) Architecture of the cerebral cortical association connectome "
            "underlying cognition. PNAS 112:E2093–E2101.",
            "",
            "http://brancusi1.usc.edu/connectome/2011/",
        ),
        (
            "graphml/rat/rattus.norvegicus_brain_2.graphml",
            "rat_brain_2", "rat", "whole_brain",
            ["connectome", "graphml", "rat", "macro", "tract_tracing"],
            "Rattus norvegicus whole-brain connectivity, second dataset.",
            "", "", "",
        ),
        (
            "graphml/rat/rattus.norvegicus_brain_3.graphml",
            "rat_brain_3", "rat", "whole_brain",
            ["connectome", "graphml", "rat", "macro", "tract_tracing"],
            "Rattus norvegicus whole-brain connectivity (USC Brancusi connectome, 2013 update).",
            "Bota et al. (2015) Architecture of the cerebral cortical association connectome "
            "underlying cognition. PNAS 112:E2093–E2101.",
            "",
            "http://brancusi1.usc.edu/connectome/2013/1/",
        ),
    ]

    for rel, name, species, region, tags, desc, paper, doi, src_url in graphml_specs:
        path = DATA_ROOT / rel
        if not path.exists():
            print(f"  [SKIP missing] {path}")
            continue
        print(f"  Loading {name} ...")
        try:
            mat = load_graphml(path)
            graphs.append(_make_entry(
                name, mat, str(path), "graphml", species, region, tags,
                description=desc, paper=paper, doi=doi, source_url=src_url,
            ))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    # ── skewed.de csv.zip files ───────────────────────────────────────────────
    csvzip_specs = [
        (
            "skewed/fly_hemibrain/fly_hemibrain.csv.zip",
            "fly_hemibrain", "fly", "whole_brain",
            ["connectome", "csv.zip", "fly", "hemibrain", "large", "EM"],
            "Drosophila melanogaster hemibrain connectome (~21k neurons, ~50M synapses) "
            "reconstructed from a focused ion beam scanning EM volume.",
            "Scheffer et al. (2020) A connectome and analysis of the adult Drosophila central "
            "brain. eLife 9:e57443.",
            "10.7554/eLife.57443",
            "https://doi.org/10.7554/eLife.57443",
        ),
        (
            "skewed/fly_larva/fly_larva.csv.zip",
            "fly_larva", "fly", "whole_brain",
            ["connectome", "csv.zip", "fly", "larva", "EM"],
            "Drosophila melanogaster larval central nervous system complete connectome. "
            "First complete connectome of a brain capable of complex behaviour.",
            "Winding et al. (2023) The connectome of an insect brain. Science 379:eadd9330.",
            "10.1126/science.add9330",
            "https://doi.org/10.1126/science.add9330",
        ),
        (
            "skewed/celegans_2019/hermaphrodite_chemical.csv.zip",
            "celegans_2019_herm_chem", "worm", "whole_brain",
            ["connectome", "csv.zip", "c.elegans", "chemical", "EM"],
            "C. elegans hermaphrodite complete chemical synapse connectome (2019 update). "
            "Eight developmental time points; this is the adult hermaphrodite chemical network.",
            "Witvliet et al. (2021) Connectomes across development reveal principles of brain "
            "maturation. Nature 596:257–261.",
            "10.1038/s41586-021-03778-8",
            "https://doi.org/10.1038/s41586-021-03778-8",
        ),
        (
            "skewed/celegansneural/celegansneural.csv.zip",
            "celegansneural", "worm", "whole_brain",
            ["connectome", "csv.zip", "c.elegans", "classic", "EM"],
            "C. elegans hermaphrodite neural network (classic White et al. 1986 dataset). "
            "302 neurons, complete chemical and gap-junction synaptic wiring.",
            "White et al. (1986) The structure of the nervous system of the nematode "
            "Caenorhabditis elegans. Phil Trans R Soc B 314:1–340.",
            "10.1098/rstb.1986.0056",
            "https://networks.skewed.de/net/celegansneural",
        ),
        (
            "skewed/cintestinalis/cintestinalis.csv.zip",
            "cintestinalis", "tunicate", "whole_brain",
            ["connectome", "csv.zip", "cintestinalis", "EM"],
            "Ciona intestinalis (tunicate) larval nervous system connectome reconstructed by EM. "
            "Ancestral chordate nervous system; 177 neurons.",
            "Ryan et al. (2016) The CNS connectome of a tadpole larva of Ciona intestinalis "
            "(L.) highlights sidedness in the brain of a chordate sibling. eLife 5:e16962.",
            "10.7554/eLife.16962",
            "https://doi.org/10.7554/eLife.16962",
        ),
        (
            "skewed/budapest_connectome/all_20k.csv.zip",
            "budapest_all_20k", "human", "whole_brain",
            ["connectome", "csv.zip", "human", "DTI", "macro"],
            "Budapest Reference Connectome (20k resolution) — group-averaged human structural "
            "connectome from DTI tractography of 96 subjects.",
            "Szalkai et al. (2017) The Budapest Reference Connectome Server v3.0. "
            "Neuroscience Letters 657:15–19.",
            "10.1016/j.neulet.2017.07.019",
            "https://pitgroup.org/connectome/",
        ),
        (
            "skewed/macaque_neural/macaque_neural.csv.zip",
            "macaque_neural", "macaque", "whole_brain",
            ["connectome", "csv.zip", "macaque", "macro"],
            "Macaque neural connectivity dataset from networks.skewed.de. "
            "Coarse-resolution interareal cortical network.",
            "", "", "https://networks.skewed.de/net/macaque_neural",
        ),
        (
            "skewed/human_brains/BNU1_0025864_1_DTI_AAL.csv.zip",
            "human_bnu1_aal", "human", "whole_brain",
            ["connectome", "csv.zip", "human", "DTI", "individual", "AAL"],
            "Individual human structural connectome (subject BNU1_0025864_1) from DTI "
            "tractography parcellated with the AAL atlas (116 regions).",
            "Zuo et al. (2014) An Open Science Resource for Establishing Reliability and "
            "Reproducibility in Functional Connectomics. Scientific Data 1:140049.",
            "10.1038/sdata.2014.49",
            "https://fcon_1000.projects.nitrc.org/indi/CoRR/html/bnu_1.html",
        ),
        (
            "skewed/human_brains/BNU1_0025864_1_DTI_CPAC200.csv.zip",
            "human_bnu1_cpac200", "human", "whole_brain",
            ["connectome", "csv.zip", "human", "DTI", "individual", "CPAC200"],
            "Individual human structural connectome (subject BNU1_0025864_1) from DTI "
            "tractography parcellated with CPAC200 atlas (200 regions).",
            "Zuo et al. (2014) An Open Science Resource for Establishing Reliability and "
            "Reproducibility in Functional Connectomics. Scientific Data 1:140049.",
            "10.1038/sdata.2014.49",
            "https://fcon_1000.projects.nitrc.org/indi/CoRR/html/bnu_1.html",
        ),
    ]

    for rel, name, species, region, tags, desc, paper, doi, src_url in csvzip_specs:
        path = DATA_ROOT / rel
        if not path.exists():
            print(f"  [SKIP missing] {path}")
            continue
        print(f"  Loading {name} ...")
        try:
            mat = load_csv_zip(path)
            graphs.append(_make_entry(
                name, mat, str(path), "csv.zip", species, region, tags,
                description=desc, paper=paper, doi=doi, source_url=src_url,
            ))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    # ── conn2res datasets (Zenodo 10205004) ───────────────────────────────────
    conn2res_specs = [
        (
            "drosophila", "conn2res_drosophila", "conn2res_csv",
            "fly", "whole_brain",
            ["connectome", "conn2res", "fly", "region_level"],
            "Drosophila melanogaster brain-wide connectivity at region level (49 regions), "
            "used in conn2res cross-species comparison example.",
            "Chiang et al. (2011) Three-dimensional reconstruction of brain-wide wiring networks "
            "in Drosophila at single-cell resolution. Curr Biol 21:1–11.",
            "10.1016/j.cub.2010.11.056",
            "https://zenodo.org/records/10205004",
        ),
        (
            "macaque_modha", "conn2res_macaque_modha", "conn2res_csv",
            "macaque", "whole_brain",
            ["connectome", "conn2res", "macaque", "macro"],
            "Macaque long-distance pathways network (Modha & Singh 2010), 242 brain regions.",
            "Modha & Singh (2010) Network architecture of the long-distance pathways in the "
            "macaque brain. PNAS 107:13485–13490.",
            "10.1073/pnas.1008054107",
            "https://zenodo.org/records/10205004",
        ),
    ]

    for subdir, name, fmt, species, region, tags, desc, paper, doi, src_url in conn2res_specs:
        conn_dir = CONN2RES_DATA / subdir
        if not (conn_dir / "conn.csv").exists():
            print(f"  [SKIP missing] {conn_dir}/conn.csv")
            continue
        print(f"  Loading {name} ...")
        try:
            mat = load_conn2res_csv(conn_dir)
            graphs.append(_make_entry(
                name, mat, str(conn_dir), fmt, species, region, tags,
                description=desc, paper=paper, doi=doi, source_url=src_url,
            ))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    # ── conn2res human consensus connectomes (.npy) ───────────────────────────
    human_dir = CONN2RES_DATA / "human"
    for i in range(6):
        npy_path = human_dir / f"consensus_{i}.npy"
        if not npy_path.exists():
            print(f"  [SKIP missing] {npy_path}")
            continue
        name = f"conn2res_human_consensus_{i}"
        print(f"  Loading {name} ...")
        try:
            mat = load_conn2res_npy(npy_path)
            graphs.append(_make_entry(
                name, mat, str(npy_path), "conn2res_npy",
                "human", "whole_brain",
                ["connectome", "conn2res", "human", "DTI", "consensus", "macro"],
                description=(
                    f"Group-level consensus human structural connectome #{i} "
                    f"(1015 parcels) from bootstrapped samples of 70 subjects. "
                    f"Used in conn2res global network organization example."
                ),
                paper=(
                    "Suárez et al. (2024) Connectome-based reservoir computing with the "
                    "conn2res toolbox. PLOS Computational Biology."
                ),
                doi="10.1371/journal.pcbi.1011370",
                source_url="https://zenodo.org/records/10205004",
            ))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    # ── mice_column_v1 network + synthetic block-diagonal tilings ────────────
    if (MICE_COLUMN_V1_ROOT / "mice_connections_processed.parquet").exists():
        print("  Loading mice_column_v1 ...")
        base = load_mice_column_v1(MICE_COLUMN_V1_ROOT, use_weights=False)
        graphs.append(_make_entry(
            "mice_column_v1", base, str(MICE_COLUMN_V1_ROOT), "mice_column_v1",
            "mouse", "V1",
            ["connectome", "parquet", "mouse", "V1", "GLIF", "guozhang"],
            description=(
                "Mouse V1 cortical-column patch (4,166 neurons) from Guozhang's "
                "network-generator re-fit with MICrONS distance and degree rules. "
                "Binary adjacency is loaded from repo-local processed parquet data. "
                "Covers a 200um x 200um column at x=[2000,2200], z=[2200,2400]."
            ),
            paper="Internal dataset - Guozhang V1 generator fit (unpublished).",
            source_url="datasets/mice_column_v1/README.md",
            notes=(
                "In-repo metadata lives under datasets/mice_column_v1. Large data "
                "lives under data/external/mice_column_v1 and should be symlinked "
                "or DVC-managed, not committed."
            ),
        ))

    else:
        print(f"  [SKIP missing] {MICE_COLUMN_V1_ROOT}")

    # ── MICrONS mm3 HDF5 payload (cataloged without heavy conntility load) ───
    if MICRONS_MM3_PATH.exists():
        print("  Registering microns_mm3 ...")
        graphs.append(_make_file_entry(
            "microns_mm3", str(MICRONS_MM3_PATH), "microns_h5",
            "mouse", "visual_cortex",
            ["connectome", "microns", "hdf5", "mouse", "cortex", "EM"],
            n_rows=None,
            n_cols=None,
            nnz=None,
            description=(
                "MICrONS mm3 connectome HDF5 payload. This catalog entry records "
                "the DVC-managed source file; loading requires optional conntility."
            ),
            paper="MICrONS Consortium dataset.",
            source_url="docs/microns.md",
            notes=(
                "DVC-managed under data/external/microns. Not loaded during catalog "
                "generation because conntility is optional and the payload is large."
            ),
        ))
    else:
        print(f"  [SKIP missing] {MICRONS_MM3_PATH}")

    # ── SuiteSparse Matrix Collection archives ───────────────────────────────
    suitesparse_manifest = SUITESPARSE_ROOT / "manifest.json"
    if suitesparse_manifest.exists():
        with open(suitesparse_manifest) as f:
            manifest = json.load(f)
        for item in manifest.get("matrices", []):
            group = item["group"]
            matrix_name = item["name"]
            rel = Path(item["path"])
            path = SUITESPARSE_ROOT / rel
            if not path.exists():
                print(f"  [SKIP missing] {path}")
                continue
            name = f"suitesparse_{group.lower()}_{matrix_name.lower().replace('-', '_')}"
            print(f"  Loading {name} ...")
            try:
                mat = load_suitesparse_mm(path)
                graphs.append(_make_entry(
                    name, mat, str(path), "suitesparse_mm",
                    "unknown", "graph",
                    ["suitesparse", group, "matrix_market", "graph"],
                    description=(
                        f"SuiteSparse Matrix Collection graph {group}/{matrix_name} "
                        f"({item.get('kind', '').strip()})."
                    ),
                    paper="SuiteSparse Matrix Collection.",
                    source_url=f"https://sparse.tamu.edu/{group}/{matrix_name}",
                    notes=(
                        f"Downloaded from {item.get('url', '')}. "
                        "Payload is DVC-managed under data/external/suitesparse."
                    ),
                ))
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
    else:
        print(f"  [SKIP missing] {suitesparse_manifest}")

    save_catalog(graphs)


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print catalog
# ──────────────────────────────────────────────────────────────────────────────

def print_catalog() -> None:
    graphs = load_catalog()
    if not graphs:
        print("Catalog is empty. Run `connectome-catalog` to build it.")
        return

    header = (
        f"{'Name':<38} {'n':>7} {'nnz':>10} {'density':>9}  "
        f"{'species':<10} {'format':<15} {'doi'}"
    )
    print(header)
    print("-" * len(header))
    for g in graphs:
        n = g.get("n_rows", g.get("n", "?"))
        n_text = f"{n:,}" if isinstance(n, int) else "?"
        nnz = g.get("nnz")
        nnz_text = f"{nnz:,}" if isinstance(nnz, int) else "?"
        density = g.get("density")
        density_text = f"{density:>9.2e}" if isinstance(density, (int, float)) else f"{'?':>9}"
        doi = g.get("doi", "")[:40]
        print(
            f"{g['name']:<38} {n_text:>7} {nnz_text:>10} "
            f"{density_text}  {g['species']:<10} "
            f"{g['source_format']:<15} {doi}"
        )


def main() -> None:
    print("Building graph catalog...")
    build_catalog()
    print()
    print_catalog()


if __name__ == "__main__":
    main()
