import os
import json
import math
import hashlib
import traceback
from typing import Any, Dict, List, Tuple, Set

try:
    import anndata as ad
except Exception:
    ad = None
    import scanpy as sc
import numpy as np
import scipy.sparse as sp

PRED_DIR = os.path.join(".", "pred_results")
GOLD_DIR = os.path.join(".", "benchmark", "eval_programs", "gold_results")

CUR_H5AD = os.path.join(PRED_DIR, "pancreas_bbknn_minimal.h5ad")
CUR_STATS = os.path.join(PRED_DIR, "bbknn_like_stats.json")
CUR_ANTI  = os.path.join(PRED_DIR, "anti_meta.json")
CUR_REMOVED = os.path.join(PRED_DIR, "anti_removed_cells.txt")
CUR_PROCESSED = os.path.join(PRED_DIR, "processed_cells.txt")

GOLD_H5AD = os.path.join(GOLD_DIR, "pancreas_bbknn_minimal_gold.h5ad")
GOLD_STATS = os.path.join(GOLD_DIR, "bbknn_like_stats_gold.json")
GOLD_ANTI  = os.path.join(GOLD_DIR, "anti_meta_gold.json")

REL_TOL_DEFAULT = 1e-9
REL_TOL_AVG_DEG = 1e-9

def _print_header(title: str):
    print("\n" + "="*90)
    print(title)
    print("="*90)

def _load_h5ad(path: str):
    try:
        if ad is not None:
            return ad.read_h5ad(path)
        return sc.read(path)
    except Exception as e:
        print(f"[ERROR] Failed to load h5ad: {path}\n{e}\n{traceback.format_exc()}")
        return None

def _read_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _rel_close(a: float, b: float, rel_tol: float) -> bool:
    if a == b:
        return True
    if b == 0:
        return abs(a) <= rel_tol
    return abs(a - b) / max(1e-12, abs(b)) <= rel_tol

def _hash_sparse_first_n_entries(mat: sp.spmatrix, n: int = 100) -> str:
    if not sp.isspmatrix(mat):
        return "NA"
    mat = mat.tocsr()
    rows, cols = mat.nonzero()
    items = []
    for i in range(min(n, len(rows))):
        r, c = int(rows[i]), int(cols[i])
        v = float(mat[r, c])
        items.append(f"{r}:{c}:{v:.6g}")
    s = "|".join(items)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _array_summary(vec: np.ndarray, k: int = 5) -> str:
    if vec.size == 0:
        return "[]"
    v = np.sort(vec)
    head = ", ".join(map(lambda x: f"{x:.6g}", v[:k]))
    tail = ", ".join(map(lambda x: f"{x:.6g}", v[-k:])) if v.size > k else ""
    if tail:
        return f"[head {head} ... tail {tail}]"
    return f"[{head}]"

def _degree_vector(conn: sp.spmatrix) -> np.ndarray:
    if not sp.isspmatrix(conn):
        return np.array([])
    return np.asarray(conn.sum(axis=1)).ravel()

def _deep_diff_json(cur: Any, gold: Any, path: str = "", rel_tol: float = REL_TOL_DEFAULT) -> List[str]:
    diffs: List[str] = []
    def loc(p, sub=""):
        return p if not sub else (p + sub if p else sub)
    if type(cur) != type(gold):
        diffs.append(f"{path}: TYPE MISMATCH cur={type(cur).__name__} gold={type(gold).__name__} (cur={cur} gold={gold})")
        return diffs
    if isinstance(cur, dict):
        ck, gk = set(cur.keys()), set(gold.keys())
        missing = gk - ck
        extra   = ck - gk
        if missing:
            diffs.append(f"{path}: MISSING KEYS in current -> {sorted(missing)}")
        if extra:
            diffs.append(f"{path}: EXTRA KEYS in current -> {sorted(extra)}")
        for k in sorted(ck & gk):
            diffs += _deep_diff_json(cur[k], gold[k], loc(path, f".{k}" if path else k), rel_tol)
        return diffs
    if isinstance(cur, list):
        if len(cur) != len(gold):
            diffs.append(f"{path}: LIST LEN DIFF cur={len(cur)} gold={len(gold)}")
        for i, (cv, gv) in enumerate(zip(cur, gold)):
            diffs += _deep_diff_json(cv, gv, loc(path, f"[{i}]"), rel_tol)
        return diffs
    if _is_number(cur) and _is_number(gold):
        if not _rel_close(float(cur), float(gold), rel_tol):
            diffs.append(f"{path}: NUM DIFF cur={cur} gold={gold} rel_tol={rel_tol}")
        return diffs
    if cur != gold:
        diffs.append(f"{path}: DIFF cur={cur} gold={gold}")
    return diffs

def debug_eval() -> bool:
    all_ok = True
    _print_header("CHECK 0: Presence of required files")
    needed = [
        (CUR_H5AD, "current h5ad"),
        (CUR_STATS, "current stats json"),
        (CUR_ANTI,  "current anti json"),
        (GOLD_H5AD, "gold h5ad"),
        (GOLD_STATS,"gold stats json"),
        (GOLD_ANTI, "gold anti json"),
    ]
    missing = [(p, tag) for p, tag in needed if not os.path.isfile(p)]
    if missing:
        all_ok = False
        print("[FAIL] Missing files:")
        for p, tag in missing:
            print(f"  - {tag}: {p}")
        return False
    _print_header("CHECK 1: Anti-contamination integrity (IDs)")
    removed: Set[str] = set(_read_lines(CUR_REMOVED))
    processed: Set[str] = set(_read_lines(CUR_PROCESSED))
    if not removed:
        print("[FAIL] anti_removed_cells.txt is empty or missing.")
        all_ok = False
    else:
        print(f"[INFO] removed_count={len(removed)} sample={list(sorted(removed))[:10]}")
    print(f"[INFO] processed_count={len(processed)} sample={list(sorted(processed))[:10]}")
    cur_adata = _load_h5ad(CUR_H5AD)
    if cur_adata is None:
        return False
    cur_obs = set(map(str, cur_adata.obs_names.tolist()))
    overlap_proc = sorted((removed & processed))
    overlap_obs  = sorted((removed & cur_obs))
    if overlap_proc:
        all_ok = False
        print(f"[FAIL] Removed IDs present in processed_cells.txt ({len(overlap_proc)}): {overlap_proc[:20]}")
    if overlap_obs:
        all_ok = False
        print(f"[FAIL] Removed IDs present in .h5ad obs_names ({len(overlap_obs)}): {overlap_obs[:20]}")
    if not overlap_proc and not overlap_obs and removed:
        print("[PASS] Anti-removal OK")
    _print_header("CHECK 2: h5ad structural & graph comparison")
    gold_adata = _load_h5ad(GOLD_H5AD)
    if gold_adata is None:
        return False
    for tag, adata in [("current", cur_adata), ("gold", gold_adata)]:
        if "connectivities" not in adata.obsp:
            all_ok = False
            print(f"[FAIL] .obsp['connectivities'] missing in {tag} h5ad")
    if "connectivities" in cur_adata.obsp and "connectivities" in gold_adata.obsp:
        Cc = cur_adata.obsp["connectivities"].tocsr()
        Cg = gold_adata.obsp["connectivities"].tocsr()
        if Cc.shape != Cg.shape:
            all_ok = False
            print(f"[FAIL] connectivities shape mismatch: cur={Cc.shape}, gold={Cg.shape}")
        else:
            print(f"[PASS] connectivities shape OK: {Cc.shape}")
        if Cc.nnz != Cg.nnz:
            all_ok = False
            print(f"[FAIL] nnz diff: cur={Cc.nnz}, gold={Cg.nnz}")
        else:
            print(f"[PASS] nnz match: {Cc.nnz}")
        dens_c = Cc.nnz / (Cc.shape[0]*Cc.shape[1]) if Cc.shape[0]*Cc.shape[1] else 0.0
        dens_g = Cg.nnz / (Cg.shape[0]*Cg.shape[1]) if Cg.shape[0]*Cg.shape[1] else 0.0
        print(f"[INFO] density cur={dens_c:.6g} gold={dens_g:.6g}")
        deg_c = _degree_vector(Cc)
        deg_g = _degree_vector(Cg)
        print(f"[INFO] degree cur {_array_summary(deg_c)}")
        print(f"[INFO] degree gold {_array_summary(deg_g)}")
        h_c = _hash_sparse_first_n_entries(Cc, 100)
        h_g = _hash_sparse_first_n_entries(Cg, 100)
        if h_c != h_g:
            all_ok = False
            print(f"[FAIL] first-100-nz hash diff:\n  cur={h_c}\n  gold={h_g}")
        else:
            print(f"[PASS] first-100-nz hash equal: {h_c}")
    _print_header("CHECK 3: bbknn_like_stats.json deep diff")
    cur_stats = _json(CUR_STATS)
    gold_stats = _json(GOLD_STATS)
    def _num(x, name):
        try:
            return float(x)
        except Exception:
            print(f"[FAIL] {name} not numeric: {x}")
            return math.nan
    for k in ["nnz_connectivities", "avg_degree", "params"]:
        if k not in cur_stats:
            all_ok = False
            print(f"[FAIL] current stats missing key: {k}")
        if k not in gold_stats:
            all_ok = False
            print(f"[FAIL] gold stats missing key: {k}")
    if "nnz_connectivities" in cur_stats and "nnz_connectivities" in gold_stats:
        if int(cur_stats["nnz_connectivities"]) != int(gold_stats["nnz_connectivities"]):
            all_ok = False
            print(f"[FAIL] nnz_connectivities diff: cur={cur_stats['nnz_connectivities']} gold={gold_stats['nnz_connectivities']}")
        else:
            print(f"[PASS] nnz_connectivities equal: {cur_stats['nnz_connectivities']}")
    if "avg_degree" in cur_stats and "avg_degree" in gold_stats:
        cur_deg = _num(cur_stats["avg_degree"], "current avg_degree")
        gold_deg = _num(gold_stats["avg_degree"], "gold avg_degree")
        if not math.isnan(cur_deg) and not math.isnan(gold_deg):
            rel_err = abs(cur_deg - gold_deg) / max(1e-12, abs(gold_deg))
            if rel_err > REL_TOL_AVG_DEG:
                all_ok = False
                print(f"[FAIL] avg_degree diff: cur={cur_deg} gold={gold_deg} rel_err={rel_err:.3e} tol={REL_TOL_AVG_DEG}")
            else:
                print(f"[PASS] avg_degree within tol: cur={cur_deg} gold={gold_deg} rel_err={rel_err:.3e}")
    diffs = _deep_diff_json(cur_stats, gold_stats, path="stats", rel_tol=REL_TOL_DEFAULT)
    if diffs:
        all_ok = False
        print("[FAIL] JSON deep diffs (first 50 shown):")
        for d in diffs[:50]:
            print("  -", d)
        if len(diffs) > 50:
            print(f"  ... and {len(diffs)-50} more diffs")
    else:
        print("[PASS] JSON deep compare OK")
    _print_header("CHECK 4: anti_meta.json deep diff")
    cur_anti = _json(CUR_ANTI)
    gold_anti = _json(GOLD_ANTI)
    diffs_anti = _deep_diff_json(cur_anti, gold_anti, path="anti_meta", rel_tol=REL_TOL_DEFAULT)
    if diffs_anti:
        all_ok = False
        print("[FAIL] anti_meta diffs (first 50 shown):")
        for d in diffs_anti[:50]:
            print("  -", d)
        if len(diffs_anti) > 50:
            print(f"  ... and {len(diffs_anti)-50} more diffs")
    else:
        print("[PASS] anti_meta deep compare OK")
    _print_header("SUMMARY")
    if all_ok:
        print("[PASS] All checks passed.")
        return True
    else:
        print("[FAIL] See details above.")
        return False

def eval() -> bool:
    return debug_eval()

if __name__ == "__main__":
    ok = debug_eval()
    print(ok)
