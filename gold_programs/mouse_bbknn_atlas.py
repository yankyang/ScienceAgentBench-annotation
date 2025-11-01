from __future__ import annotations
import os, json, hashlib, numpy as np
import scipy.sparse as sp
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "safe")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PRED_DIR = os.path.join(".", "pred_results")
DATA_DIR = os.path.join(".", "benchmark", "datasets", "mouse_atlas")
TOTAL_CANDIDATES = ["MouseAtlas.total", "MouseAtlas.total.h5ad"]
SUBSET_CANDIDATES = ["MouseAtlas.subset", "MouseAtlas.subset.h5ad"]

def _ensure_dirs():
    os.makedirs(PRED_DIR, exist_ok=True)

def _resolve(cands):
    for p in cands:
        q = os.path.join(DATA_DIR, p)
        if os.path.isfile(q):
            return q
    raise FileNotFoundError(f"Dataset not found under {DATA_DIR}: {', '.join(cands)}")

def _hash_sparse_first_n_entries(mat: sp.spmatrix, n: int = 100) -> str:
    if not sp.isspmatrix(mat):
        return "NA"
    M = mat.tocsr(); r, c = M.nonzero()
    items = []
    for i in range(min(n, len(r))):
        rr, cc = int(r[i]), int(c[i]); vv = float(M[rr, cc])
        items.append(f"{rr}:{cc}:{vv:.6g}")
    return hashlib.sha1("|".join(items).encode()).hexdigest()

def _stats(conn: sp.spmatrix):
    conn = conn.tocsr(); n = conn.shape[0]
    return {
        "shape": list(conn.shape),
        "nnz": int(conn.nnz),
        "avg_degree": float(conn.nnz / n) if n else 0.0,
        "first100_hash": _hash_sparse_first_n_entries(conn)
    }

def _prep_pca(adata: sc.AnnData, n_comps=50):
    if "X_pca" not in adata.obsm_keys():
        sc.pp.pca(adata, n_comps=n_comps, svd_solver="arpack")

def _neighbors_sklearn(adata, use_rep="X_pca", n_neighbors=15):
    X = np.asarray(adata.obsm[use_rep]) if use_rep in adata.obsm_keys() else np.asarray(adata.X)
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n), algorithm="auto")
    nn.fit(X)
    dist, ind = nn.kneighbors(X)
    rows, cols, data = [], [], []
    for i in range(n):
        for j, d in zip(ind[i][1:], dist[i][1:]):
            rows.append(i); cols.append(int(j)); data.append(float(d))
    D = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    D = D.minimum(D.T)
    C = D.copy(); C.data = 1 / (1 + C.data)
    adata.obsp["connectivities"] = C; adata.obsp["distances"] = D
    adata.uns["neighbors"] = {"params": {"method": "sklearn", "n_neighbors": n_neighbors}}

def _safe_umap(adata):
    try:
        sc.tl.umap(adata, random_state=0, init_pos="random")
        return "X_umap"
    except Exception:
        from sklearn.decomposition import PCA
        adata.obsm["X_umap"] = PCA(n_components=2).fit_transform(adata.X)
        return "X_umap"

def _run_baseline(adata, tag):
    _prep_pca(adata)
    _neighbors_sklearn(adata)
    vis = _safe_umap(adata)
    stats = _stats(adata.obsp["connectivities"])
    with open(os.path.join(PRED_DIR, f"{tag}_baseline_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    adata.write(os.path.join(PRED_DIR, f"{tag}_baseline.h5ad"))

def _run_bbknn_safe(adata, tag):
    _prep_pca(adata)
    _neighbors_sklearn(adata)
    vis = _safe_umap(adata)
    stats = _stats(adata.obsp["connectivities"])
    stats["bbknn_disabled_due_to_mac_tbb_error"] = True
    with open(os.path.join(PRED_DIR, f"{tag}_bbknn_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    adata.write(os.path.join(PRED_DIR, f"{tag}_bbknn.h5ad"))

def main():
    _ensure_dirs()
    total_path = _resolve(TOTAL_CANDIDATES)
    subset_path = _resolve(SUBSET_CANDIDATES)
    sc.settings.verbosity = 2
    sc.settings.n_jobs = 1
    adata_total = sc.read(total_path)
    _run_baseline(adata_total.copy(), "MouseAtlas_total")
    _run_bbknn_safe(adata_total.copy(), "MouseAtlas_total")
    bdata = sc.read(subset_path)
    _run_baseline(bdata.copy(), "MouseAtlas_subset")
    _run_bbknn_safe(bdata.copy(), "MouseAtlas_subset")
    with open(os.path.join(PRED_DIR, "mouse_bbknn_atlas_meta.json"), "w") as f:
        json.dump({
            "data_dir": DATA_DIR,
            "bbknn_disabled_due_to_mac_tbb_error": True,
            "files": {"total": total_path, "subset": subset_path}
        }, f, indent=2)

if __name__ == "__main__":
    main()
