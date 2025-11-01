from __future__ import annotations
import os
import json
import time
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

PRED_DIR = os.path.join(".", "pred_results")
DATA_PATH = os.path.join(".", "benchmark", "datasets", "mouse_atlas", "MouseAtlas.subset.h5ad")

OUT_H5AD = os.path.join(PRED_DIR, "mouse_atlas_harmony_gold.h5ad")
TIMINGS_JSON = os.path.join(PRED_DIR, "mouse_atlas_harmony_timings.json")
STATS_JSON = os.path.join(PRED_DIR, "mouse_atlas_harmony_stats.json")

ANTI_REMOVED_TXT = os.path.join(PRED_DIR, "anti_removed_cells.txt")
ANTI_META_JSON = os.path.join(PRED_DIR, "anti_meta.json")
PROCESSED_CELLS_TXT = os.path.join(PRED_DIR, "processed_cells.txt")

def _ensure_dirs():
    os.makedirs(PRED_DIR, exist_ok=True)

def _save_matrix_preview(path: str, X: np.ndarray, n_rows: int = 6, n_cols: int = 6):
    import csv
    X = np.asarray(X)
    r = min(n_rows, X.shape[0])
    c = min(n_cols, X.shape[1])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"col_{j}" for j in range(c)])
        for i in range(r):
            w.writerow([f"{float(X[i, j]):.6f}" for j in range(c)])

def _neighbors_stats(adata) -> dict:
    out = {}
    con = adata.obsp.get("connectivities", None)
    if con is not None:
        con = con.tocsr()
        out["connectivities_shape"] = [int(con.shape[0]), int(con.shape[1])]
        out["nnz_connectivities"] = int(con.nnz)
        out["avg_degree"] = float(con.nnz / con.shape[0]) if con.shape and con.shape[0] else 0.0
    else:
        out["connectivities_shape"] = None
        out["nnz_connectivities"] = 0
        out["avg_degree"] = 0.0
    return out

def _build_knn_graph_sklearn(Z: np.ndarray, n_neighbors: int = 15):
    Z = np.asarray(Z)
    n = Z.shape[0]
    k = min(n_neighbors, max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=1)
    nn.fit(Z)
    dist, ind = nn.kneighbors(Z, n_neighbors=k + 1, return_distance=True)
    rows, cols, dvals = [], [], []
    for i in range(n):
        neigh_idx = ind[i]
        neigh_dst = dist[i]
        if neigh_idx[0] == i:
            neigh_idx = neigh_idx[1:]
            neigh_dst = neigh_dst[1:]
        else:
            neigh_idx = neigh_idx[:k]
            neigh_dst = neigh_dst[:k]
        for j, dj in zip(neigh_idx, neigh_dst):
            rows.append(i); cols.append(int(j)); dvals.append(float(dj))
    D = sp.coo_matrix((np.array(dvals), (np.array(rows), np.array(cols))), shape=(n, n)).tocsr()
    D = D.minimum(D.T)
    C = D.copy()
    C.data = 1.0 / (1.0 + C.data)
    params = {
        "connectivity_method": "sklearn_knn_safe",
        "n_neighbors": int(k),
        "n_cells": int(n),
        "n_features": int(Z.shape[1]),
    }
    return C.tocsr(), D.tocsr(), params

def main():
    _ensure_dirs()
    sc.settings.verbosity = 2
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please place MouseAtlas.subset.h5ad under ./benchmark/datasets/mouse_atlas/ ."
        )
    timings = {}
    t0 = time.perf_counter()
    adata = sc.read(DATA_PATH)
    timings["load_seconds"] = time.perf_counter() - t0
    rng = np.random.default_rng(20240901)
    n_remove = min(5, adata.n_obs)
    removed = list(rng.choice(adata.obs_names.to_numpy(), size=n_remove, replace=False))
    with open(ANTI_REMOVED_TXT, "w") as f:
        f.write("\n".join(removed) + ("\n" if removed else ""))
    adata = adata[~adata.obs_names.isin(removed)].copy()
    with open(PROCESSED_CELLS_TXT, "w") as f:
        for cid in adata.obs_names.tolist():
            f.write(f"{cid}\n")
    anti_meta = {
        "removed_count": int(n_remove),
        "removed_ids_head": removed[:5],
        "total_after_removal": int(adata.n_obs),
        "data_path": DATA_PATH,
    }
    with open(ANTI_META_JSON, "w") as f:
        json.dump(anti_meta, f, indent=2)
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50, svd_solver="arpack")
    X_pca = adata.obsm["X_pca"]
    _save_matrix_preview(os.path.join(PRED_DIR, "preview_pca_before_harmony_6x6.csv"), X_pca)
    if "sample" not in adata.obs.columns:
        raise KeyError("Expected 'sample' in adata.obs for batch labels.")
    try:
        import pandas as pd
        from harmonypy import run_harmony
        meta = pd.DataFrame({"sample": adata.obs["sample"].astype(str).values})
        t1 = time.perf_counter()
        ho = run_harmony(X_pca, meta, vars_use=["sample"], max_iter_harmony=10)
        timings["harmony_seconds"] = time.perf_counter() - t1
        Z = getattr(ho, "Z_corr", None)
        if Z is None:
            raise RuntimeError("harmonypy returned no Z_corr")
        Z = np.asarray(Z)
        if Z.shape != X_pca.shape:
            if Z.T.shape == X_pca.shape:
                Z = Z.T
            else:
                raise RuntimeError(f"Unexpected Z_corr shape: {Z.shape} vs X_pca {X_pca.shape}")
        adata.obsm["X_pca"] = Z
        _save_matrix_preview(os.path.join(PRED_DIR, "preview_pca_after_harmony_6x6.csv"), Z)
    except Exception as e:
        timings["harmony_seconds"] = None
        timings["harmony_error"] = str(e)
        _save_matrix_preview(os.path.join(PRED_DIR, "preview_pca_after_harmony_6x6.csv"), X_pca)
    t2 = time.perf_counter()
    Z_use = adata.obsm["X_pca"]
    C, D, params = _build_knn_graph_sklearn(Z_use, n_neighbors=15)
    timings["neighbors_seconds"] = time.perf_counter() - t2
    adata.obsp["connectivities"] = C
    adata.obsp["distances"] = D
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": params,
    }
    X_dense = adata.X.A if hasattr(adata.X, "A") else np.asarray(adata.X)
    _save_matrix_preview(os.path.join(PRED_DIR, "preview_rawX_6x6.csv"), X_dense)
    stats = _neighbors_stats(adata)
    with open(STATS_JSON, "w") as f:
        json.dump(
            {
                "graph": stats,
                "anti": anti_meta,
                "notes": "Harmony on PCA with batch=sample; neighbors via sklearn (segfault-safe).",
            },
            f,
            indent=2,
        )
    with open(TIMINGS_JSON, "w") as f:
        json.dump(timings, f, indent=2)
    adata.write(OUT_H5AD)
    print("Done. Outputs written to ./pred_results/")

if __name__ == "__main__":
    main()
