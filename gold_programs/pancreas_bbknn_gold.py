from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

try:
    import bbknn
except Exception:
    bbknn = None

PRED_DIR = os.path.join(".", "pred_results")
DATA_PATH = os.path.join(".", "benchmark", "datasets", "pancreas", "pancreas.h5ad")
ANTI_REMOVED_TXT = os.path.join(PRED_DIR, "anti_removed_cells.txt")
ANTI_META_JSON = os.path.join(PRED_DIR, "anti_meta.json")
PROCESSED_CELLS_TXT = os.path.join(PRED_DIR, "processed_cells.txt")

def ensure_dirs():
    os.makedirs(PRED_DIR, exist_ok=True)

def save_matrix_preview(path: str, X: np.ndarray, n_rows: int = 6, n_cols: int = 6):
    import csv
    X = np.asarray(X)
    n_rows = min(n_rows, X.shape[0])
    n_cols = min(n_cols, X.shape[1])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"col_{j}" for j in range(n_cols)])
        for i in range(n_rows):
            w.writerow([f"{float(X[i, j]):.5f}" for j in range(n_cols)])

def build_batch_balanced_graph(Z: np.ndarray, batch: np.ndarray, k_other: int = 3, k_same: int = 3):
    Z = np.asarray(Z)
    n = Z.shape[0]
    batch = np.asarray(batch)
    ub = np.unique(batch)
    per_batch = {}
    for b in ub:
        idx = np.flatnonzero(batch == b)
        nbrs = NearestNeighbors(algorithm='kd_tree', n_jobs=1)
        nbrs.fit(Z[idx])
        per_batch[b] = (idx, nbrs)
    rows, cols, dists = [], [], []
    def add_edges(src_i, global_idx, drow):
        for g, d in zip(global_idx, drow):
            rows.append(src_i)
            cols.append(int(g))
            dists.append(float(d))
    for i in range(n):
        zi = Z[i:i+1]
        bi = batch[i]
        for b in ub:
            if b == bi:
                continue
            idx_b, nbrs_b = per_batch[b]
            nn = min(k_other, idx_b.size)
            dist, ind = nbrs_b.kneighbors(zi, n_neighbors=nn)
            add_edges(i, idx_b[ind[0]], dist[0])
        idx_bi, nbrs_bi = per_batch[bi]
        nn_same = min(k_same + 1, idx_bi.size)
        dist, ind = nbrs_bi.kneighbors(zi, n_neighbors=nn_same)
        gidx = idx_bi[ind[0]].tolist()
        dd = dist[0].tolist()
        if i in gidx:
            j = gidx.index(i)
            gidx.pop(j); dd.pop(j)
        add_edges(i, gidx[:k_same], dd[:k_same])
    D = sp.coo_matrix((np.array(dists), (np.array(rows), np.array(cols))), shape=(n, n)).tocsr()
    D = D.minimum(D.T)
    C = D.copy()
    C.data = 1.0 / (1.0 + C.data)
    params = {'method': 'bbknn_like_safe', 'k_other': int(k_other), 'k_same': int(k_same), 'n_batches': int(len(ub)), 'n_cells': int(n), 'n_pcs': int(Z.shape[1])}
    return C.tocsr(), D.tocsr(), params

def main():
    ensure_dirs()
    sc.settings.verbosity = 2
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Local dataset not found at {DATA_PATH}. Place pancreas.h5ad locally under ./benchmark/datasets/pancreas/.")
    adata = sc.read(DATA_PATH)
    rng = np.random.default_rng(20240801)
    n_remove = min(5, adata.n_obs)
    removed = list(rng.choice(adata.obs_names.to_numpy(), size=n_remove, replace=False))
    with open(ANTI_REMOVED_TXT, "w") as f:
        f.write("\n".join(removed) + ("\n" if removed else ""))
    adata = adata[~adata.obs_names.isin(removed)].copy()
    assert not np.isin(removed, adata.obs_names.to_numpy()).any()
    with open(PROCESSED_CELLS_TXT, "w") as f:
        for cid in adata.obs_names.tolist():
            f.write(f"{cid}\n")
    anti_meta = {"removed_count": int(n_remove), "removed_ids_head": removed[:5], "total_after_removal": int(adata.n_obs), "data_path": DATA_PATH}
    with open(ANTI_META_JSON, "w") as f:
        json.dump(anti_meta, f, indent=2)
    sc.pp.pca(adata, n_comps=50, svd_solver='arpack')
    Z = adata.obsm['X_pca']
    batch = adata.obs['batch'].to_numpy()
    C, D, params = build_batch_balanced_graph(Z, batch, k_other=3, k_same=3)
    adata.obsp['connectivities'] = C
    adata.obsp['distances'] = D
    adata.uns['neighbors'] = {'connectivities_key': 'connectivities', 'distances_key': 'distances', 'params': params}
    X_dense = adata.X.A if hasattr(adata.X, 'A') else np.asarray(adata.X)
    save_matrix_preview(os.path.join(PRED_DIR, 'preview_rawX_6x6.csv'), X_dense)
    save_matrix_preview(os.path.join(PRED_DIR, 'preview_pca_6x6.csv'), Z)
    if bbknn is not None:
        try:
            bbknn.ridge_regression(adata, batch_key=['batch'], confounder_key=['celltype'])
            X_resid = adata.X.A if hasattr(adata.X, 'A') else np.asarray(adata.X)
            save_matrix_preview(os.path.join(PRED_DIR, 'preview_residualX_celltype_6x6.csv'), X_resid)
        except Exception:
            pass
    adata.write(os.path.join(PRED_DIR, 'pancreas_bbknn_minimal.h5ad'))
    stats = {'nnz_connectivities': int(C.nnz), 'avg_degree': float(C.nnz / C.shape[0]) if C.shape[0] else 0.0, 'params': params, 'anti': anti_meta}
    with open(os.path.join(PRED_DIR, 'bbknn_like_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print('Done. Outputs written to ./pred_results/')

if __name__ == "__main__":
    main()
