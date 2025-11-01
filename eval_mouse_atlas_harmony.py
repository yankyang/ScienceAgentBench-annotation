import os
import json
import math
import hashlib
import traceback
from typing import Any, Dict, List, Set

import numpy as np
import scipy.sparse as sp

try:
    import anndata as ad
except Exception:
    ad = None
    import scanpy as sc

PRED_DIR = os.path.join(".", "pred_results")
CUR_H5AD = os.path.join(PRED_DIR, "mouse_atlas_harmony_gold.h5ad")
CUR_STATS = os.path.join(PRED_DIR, "mouse_atlas_harmony_stats.json")
CUR_ANTI = os.path.join(PRED_DIR, "anti_meta.json")

GOLD_DIR = os.path.join(".", "benchmark", "eval_programs", "gold_results")
GOLD_H5AD = os.path.join(GOLD_DIR, "mouse_atlas_harmony_gold.h5ad")
GOLD_STATS = os.path.join(GOLD_DIR, "mouse_atlas_harmony_stats_gold.json")
GOLD_ANTI = os.path.join(GOLD_DIR, "mouse_atlas_anti_meta_gold.json")

REL_TOL_DEFAULT = 1e-9
REL_TOL_AVG_DEG = 1e-9

def _print_header(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def _load_h5ad(path: str):
    try:
        if ad is not None:
            return ad.read_h5ad(path)
        return sc.read(path)
    except Exception as e:
        print(f"[ERROR] Failed to load h5ad: {path}\n{e}\n{traceback.format_exc()}")
        return None

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
    return f"[head {head}" + (f" ... tail {tail}]" if tail else "]")

def _degree_vector(conn: sp.spmatrix) -> np.ndarray:
    if not sp.isspmatrix(conn):
        return np.array([])
    return np.asarray(conn.sum(axis=1)).ravel()

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _rel_close(a: float, b: float, rel_tol: float) -> bool:
    if a == b:
        return True
    if b == 0:
        return abs(a) <= rel_tol
    return abs(a - b) / max(1e-12, abs(b)) <= rel_tol

def _deep_diff_json(cur: Any, gold: Any, path: str = "", rel_tol: float = REL_TOL_DEFAULT) -> List[str]:
    diffs: List[str] = []
    def loc(p, sub=""):
        return p if not sub else (p + sub if p else sub)
    if type(cur) != type(gold):
        diffs.append(f"{path}: TYPE cur={type(cur).__name__} gold={type(gold).__name__} (cur={cur} gold={gold})")
        return diffs
    if isinstance(cur, dict):
        ck, gk = set(cur.keys()), set(gold.keys())
        missing = gk - ck
        extra = ck - gk
        if missing:
            diffs.append(f"{path}: MISSING KEYS -> {sorted(missing)}")
        if extra:
            diffs.append(f"{path}: EXTRA KEYS -> {sorted(extra)}")
        for k in sorted(ck & gk):
            diffs += _deep_diff_json(cur[k], gold[k], loc(path, f".{k}" if path else k), rel_tol)
        return diffs
    if isinstance(cur, list):
        if len(cur) != len(gold):
            diffs.append(f"{path}: LIST LEN cur={len(cur)} gold={len(gold)}")
        for i, (cv, gv) in enumerate(zip(cur, gold)):
            diffs += _deep_diff_json(cv, gv, loc(path, f"[{i}]"), rel_tol)
        return diffs
    if _is_number(cur) and _is_number(gold):
        if not _rel_close(float(cur), float(gold), rel_tol):
            diffs.append(f"{path}: NUM cur={cur} gold={gold} tol={rel_tol}")
        return diffs
    if cur != gold:
        diffs.append(f"{path}: DIFF cur={cur} gold={gold}")
    return diffs

def _check_presence() -> bool:
    _print_header("CHECK 0: presence of required files")
    ok = True
    need = [
        (CUR_H5AD, "current h5ad"),
        (CUR_STATS, "current stats"),
        (CUR_ANTI, "current anti"),
        (GOLD_H5AD, "gold h5ad"),
        (GOLD_STATS, "gold stats"),
        (GOLD_ANTI, "gold anti"),
    ]
    miss = [(p, tag) for p, tag in need if not os.path.isfile(p)]
    if miss:
        ok = False
        print("[FAIL] Missing files:")
        for p, tag in miss:
            print(f"  - {tag}: {p}")
    else:
        print("[PASS] all required files present")
    return ok

def _compare_h5ad() -> bool:
    _print_header("CHECK 1: h5ad structural & graph comparison")
    ok = True
    cur = _load_h5ad(CUR_H5AD)
    gold = _load_h5ad(GOLD_H5AD)
    if cur is None or gold is None:
        return False
    for tag, ad in [("current", cur), ("gold", gold)]:
        if "connectivities" not in ad.obsp:
            ok = False
            print(f"[FAIL] {tag} missing obsp['connectivities']")
    if "connectivities" in cur.obsp and "connectivities" in gold.obsp:
        Cc = cur.obsp["connectivities"].tocsr()
        Cg = gold.obsp["connectivities"].tocsr()
        if Cc.shape != Cg.shape:
            ok = False
            print(f"[FAIL] shape diff  cur={Cc.shape}  gold={Cg.shape}")
        else:
            print(f"[PASS] shape {Cc.shape}")
        if Cc.nnz != Cg.nnz:
            ok = False
            print(f"[FAIL] nnz diff  cur={Cc.nnz}  gold={Cg.nnz}")
        else:
            print(f"[PASS] nnz {Cc.nnz}")
        dens_c = Cc.nnz / (Cc.shape[0] * Cc.shape[1]) if Cc.shape[0] * Cc.shape[1] else 0.0
        dens_g = Cg.nnz / (Cg.shape[0] * Cg.shape[1]) if Cg.shape[0] * Cg.shape[1] else 0.0
        print(f"[INFO] density  cur={dens_c:.6g}  gold={dens_g:.6g}")
        deg_c = _degree_vector(Cc)
        deg_g = _degree_vector(Cg)
        print(f"[INFO] degree cur {_array_summary(deg_c)}")
        print(f"[INFO] degree gold {_array_summary(deg_g)}")
        h_c = _hash_sparse_first_n_entries(Cc, 100)
        h_g = _hash_sparse_first_n_entries(Cg, 100)
        if h_c != h_g:
            ok = False
            print(f"[FAIL] first-100-nz hash diff:\n  cur={h_c}\n  gold={h_g}")
        else:
            print(f"[PASS] first-100-nz hash {h_c}")
    return ok

def _compare_json(cur_path: str, gold_path: str, title: str, extra_numeric_checks: Dict[str, float] = None) -> bool:
    _print_header(title)
    ok = True
    try:
        with open(cur_path, "r") as f:
            cur = json.load(f)
        with open(gold_path, "r") as f:
            gold = json.load(f)
    except Exception as e:
        print(f"[ERROR] read json failed: {e}")
        return False
    if extra_numeric_checks:
        for key, rel_tol in extra_numeric_checks.items():
            cur_v = cur
            gold_v = gold
            for k in key.split("."):
                cur_v = cur_v.get(k) if isinstance(cur_v, dict) else None
                gold_v = gold_v.get(k) if isinstance(gold_v, dict) else None
            if cur_v is None or gold_v is None:
                ok = False
                print(f"[FAIL] missing numeric key: {key} (cur={cur_v}, gold={gold_v})")
            else:
                try:
                    cur_f = float(cur_v)
                    gold_f = float(gold_v)
                    if not _rel_close(cur_f, gold_f, rel_tol):
                        ok = False
                        print(f"[FAIL] {key} diff cur={cur_f} gold={gold_f} tol={rel_tol}")
                    else:
                        print(f"[PASS] {key} within tol (cur={cur_f}, gold={gold_f})")
                except Exception:
                    ok = False
                    print(f"[FAIL] {key} non-numeric (cur={cur_v}, gold={gold_v})")
    diffs = _deep_diff_json(cur, gold, path=os.path.basename(cur_path).replace(".json", ""), rel_tol=REL_TOL_DEFAULT)
    if diffs:
        ok = False
        print("[FAIL] JSON deep diffs (first 50):")
        for d in diffs[:50]:
            print("  -", d)
        if len(diffs) > 50:
            print(f"  ... and {len(diffs)-50} more")
    else:
        print("[PASS] JSON deep compare OK")
    return ok

def debug_eval() -> bool:
    all_ok = True
    if not _check_presence():
        return False
    if not _compare_h5ad():
        all_ok = False
    stats_ok = _compare_json(
        CUR_STATS, GOLD_STATS,
        "CHECK 2: stats json deep diff",
        extra_numeric_checks={
            "graph.nnz_connectivities": 0.0,
            "graph.avg_degree": REL_TOL_AVG_DEG,
        },
    )
    if not stats_ok:
        all_ok = False
    anti_ok = _compare_json(
        CUR_ANTI, GOLD_ANTI,
        "CHECK 3: anti_meta json deep diff",
        extra_numeric_checks=None,
    )
    if not anti_ok:
        all_ok = False
    _print_header("SUMMARY")
    print("[PASS]" if all_ok else "[FAIL]")
    return all_ok

def eval() -> bool:
    return debug_eval()

if __name__ == "__main__":
    ok = debug_eval()
    print(ok)
