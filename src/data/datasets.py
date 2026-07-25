from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlretrieve

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.utils.graph import average_degree, clean_edge_index, graph_density
from src.utils.io import ensure_dir, file_checksum, save_json

# Official SNAP mirrors (PyG graphmining.ai URLs are currently 404).
# Facebook large zip is marked unavailable on SNAP; use MUSAE GitHub raw files.
SOURCES = {
    "facebook": {
        "kind": "musae_files",
        "edges": "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/edges/facebook_edges.csv",
        "features": "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/features/facebook.json",
        "target": "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/target/facebook_target.csv",
        "label_col": "page_type",
        "label_mode": "category_index",
        "citation": "Rozemberczki et al., MUSAE / Facebook Page-Page (SNAP musae-facebook)",
    },
    "lastfm": {
        "kind": "snap_zip",
        "url": "https://snap.stanford.edu/data/lastfm_asia.zip",
        "edges_rel": "lasftm_asia/lastfm_asia_edges.csv",  # SNAP folder spelling
        "features_rel": "lasftm_asia/lastfm_asia_features.json",
        "target_rel": "lasftm_asia/lastfm_asia_target.csv",
        "label_col": 1,
        "label_mode": "column_index",
        "citation": "Rozemberczki & Sarkar, LastFM Asia (SNAP)",
    },
    "github": {
        "kind": "snap_zip",
        "url": "https://snap.stanford.edu/data/git_web_ml.zip",
        "edges_rel": "git_web_ml/musae_git_edges.csv",
        "features_rel": "git_web_ml/musae_git_features.json",
        "target_rel": "git_web_ml/musae_git_target.csv",
        "label_col": "ml_target",
        "label_mode": "named_column",
        "citation": "Rozemberczki et al., GitHub Social (SNAP musae-github)",
    },
}


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)
    return dest


def _features_to_matrix(features: dict, ids: np.ndarray, n_feats: Optional[int] = None) -> torch.Tensor:
    """Convert MUSAE-style bag-of-word index lists to a multi-hot feature matrix.

    The historical PyG ``.npz`` hosts exposed a pre-baked 128-d matrix; those URLs are
    currently 404. We reconstruct binary multi-hot features from the official MUSAE/SNAP
    index lists (scientifically preferable to treating indices as dense values).
    """
    all_idx = []
    for key, vals in features.items():
        if vals:
            all_idx.extend(int(v) for v in vals)
    vocab = (max(all_idx) + 1) if all_idx else 1
    if n_feats is not None:
        vocab = n_feats
    x = np.zeros((len(ids), vocab), dtype=np.float32)
    for row, node_id in enumerate(ids):
        vals = features.get(str(int(node_id)), features.get(str(node_id), []))
        for v in vals:
            vi = int(v)
            if 0 <= vi < vocab:
                x[row, vi] = 1.0
    return torch.from_numpy(x)


def _load_from_files(
    edges_path: Path,
    features_path: Path,
    target_path: Path,
    label_col: Any,
    label_mode: str,
) -> Data:
    edges = pd.read_csv(edges_path)
    # handle header/no-header
    if edges.shape[1] < 2:
        edges = pd.read_csv(edges_path, header=None)
    edges = edges.iloc[:, :2].astype(int)
    features = json.load(open(features_path, "r", encoding="utf-8"))
    target = pd.read_csv(target_path)
    if "id" not in target.columns:
        # assume first column is id
        target = target.rename(columns={target.columns[0]: "id"})

    if label_mode == "category_index":
        cats = sorted(target[label_col].value_counts().index.tolist())
        y = target[label_col].apply(lambda x: cats.index(x)).values
    elif label_mode == "named_column":
        y = target[label_col].values
    elif label_mode == "column_index":
        y = target.values[:, int(label_col)]
    else:
        raise ValueError(label_mode)

    x = _features_to_matrix(features, target["id"].values)
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)
    y_t = torch.tensor(y, dtype=torch.long)
    return Data(x=x, y=y_t, edge_index=edge_index)


def download_dataset(name: str, raw_dir: str | Path = "data/raw") -> Data:
    name = name.lower()
    if name not in SOURCES:
        raise ValueError(f"Unknown dataset {name}. Choose from {list(SOURCES)}")
    meta = SOURCES[name]
    root = ensure_dir(Path(raw_dir) / name)

    if meta["kind"] == "musae_files":
        edges = _download(meta["edges"], root / "edges.csv")
        feats = _download(meta["features"], root / "features.json")
        target = _download(meta["target"], root / "target.csv")
        data = _load_from_files(edges, feats, target, meta["label_col"], meta["label_mode"])
    else:
        zip_path = _download(meta["url"], root / Path(meta["url"]).name)
        extract_dir = root / "extracted"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        # SNAP zip may extract with a top-level folder already included in rel paths
        edges_p = extract_dir / meta["edges_rel"]
        feats_p = extract_dir / meta["features_rel"]
        target_p = extract_dir / meta["target_rel"]
        if not edges_p.exists():
            # try without nested prefix
            candidates = list(extract_dir.rglob(Path(meta["edges_rel"]).name))
            if not candidates:
                raise FileNotFoundError(f"Could not find edges for {name} under {extract_dir}")
            edges_p = candidates[0]
            feats_p = next(extract_dir.rglob(Path(meta["features_rel"]).name))
            target_p = next(extract_dir.rglob(Path(meta["target_rel"]).name))
        data = _load_from_files(edges_p, feats_p, target_p, meta["label_col"], meta["label_mode"])

    data.name = name  # type: ignore[attr-defined]
    data.citation = meta["citation"]  # type: ignore[attr-defined]
    return data


def preprocess_graph(data: Data) -> tuple[Data, Dict[str, Any]]:
    ei, audit = clean_edge_index(data.edge_index, num_nodes=data.num_nodes, make_undirected=True)
    out = Data(x=data.x, edge_index=ei, y=data.y)
    if hasattr(data, "name"):
        out.name = data.name  # type: ignore[attr-defined]
    if hasattr(data, "citation"):
        out.citation = data.citation  # type: ignore[attr-defined]
    return out, audit


def compute_homophily(data: Data) -> float:
    if data.y is None:
        return float("nan")
    src, dst = data.edge_index
    return float((data.y[src] == data.y[dst]).float().mean().item())


def feature_sparsity(x: torch.Tensor) -> float:
    if x is None or x.numel() == 0:
        return float("nan")
    return float((x == 0).float().mean().item())


def audit_dataset(
    data: Data,
    edge_audit: Optional[Dict[str, Any]] = None,
    library_versions: Optional[Dict[str, str]] = None,
    download_date: Optional[str] = None,
    raw_checksums: Optional[Dict[str, str]] = None,
    preprocessing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import torch_geometric

    if edge_audit is None:
        data, edge_audit = preprocess_graph(data)

    num_nodes = int(data.num_nodes)
    n_dir = int(data.edge_index.size(1))
    n_und = edge_audit.get("unique_undirected_edges", n_dir // 2)
    y = data.y.cpu().numpy() if data.y is not None else None
    classes = int(np.unique(y).size) if y is not None else 0
    class_dist = {}
    if y is not None:
        vals, counts = np.unique(y, return_counts=True)
        class_dist = {int(v): int(c) for v, c in zip(vals, counts)}

    G = to_networkx(data, to_undirected=True)
    n_cc = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)
    giant_ratio = len(largest_cc) / num_nodes if num_nodes else 0.0
    assort = float("nan")
    clustering = float("nan")
    try:
        assort = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        pass
    try:
        clustering = float(nx.average_clustering(G))
    except Exception:
        pass

    return {
        "dataset": getattr(data, "name", "unknown"),
        "citation": getattr(data, "citation", ""),
        "num_nodes": num_nodes,
        "directed_edge_entries": n_dir,
        "unique_undirected_edges": int(n_und),
        "self_loops": edge_audit.get("self_loops"),
        "duplicate_edges_approx": edge_audit.get("duplicate_directed_edges_approx"),
        "feature_dimension": int(data.x.size(1)) if data.x is not None else 0,
        "num_classes": classes,
        "class_distribution": class_dist,
        "density": graph_density(num_nodes, int(n_und)),
        "average_degree": average_degree(num_nodes, n_dir),
        "num_connected_components": int(n_cc),
        "giant_component_ratio": float(giant_ratio),
        "homophily": compute_homophily(data),
        "feature_sparsity": feature_sparsity(data.x) if data.x is not None else float("nan"),
        "assortativity": assort,
        "clustering_coefficient": clustering,
        "library_versions": library_versions
        or {
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        "dataset_download_date": download_date or dt.date.today().isoformat(),
        "raw_file_checksums": raw_checksums or {},
        "preprocessing": preprocessing
        or {
            "undirected": True,
            "remove_self_loops": True,
            "coalesce_duplicates": True,
            "feature_normalization": "none_at_audit",
            "source": "SNAP/MUSAE mirrors (PyG graphmining.ai unavailable)",
        },
    }


def checksum_raw_files(raw_root: str | Path) -> Dict[str, str]:
    root = Path(raw_root)
    out: Dict[str, str] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.stat().st_size < 200_000_000 and p.suffix != ".zip":
            rel = str(p.relative_to(root))
            try:
                out[rel] = file_checksum(p)
            except Exception:
                continue
    return out


def save_processed(data: Data, name: str, processed_dir: str | Path = "data/processed") -> Path:
    path = ensure_dir(processed_dir) / f"{name}.pt"
    torch.save(data, path)
    return path
