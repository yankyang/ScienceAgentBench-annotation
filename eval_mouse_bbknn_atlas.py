import os
import json
import math
import hashlib
import traceback
from typing import Any, Dict, List, Tuple

PRED_DIR = os.path.join(".", "pred_results")
GOLD_DIR = os.path.join(".", "benchmark", "eval_programs", "gold_results")

FILES = [
    ("MouseAtlas_total_baseline_stats.json",  "MouseAtlas_total_baseline_stats_gold.json",  "total_baseline"),
    ("MouseAtlas_total_bbknn_stats.json",     "MouseAtlas_total_bbknn_stats_gold.json",     "total_bbknn"),
    ("MouseAtlas_subset_baseline_stats.json", "MouseAtlas_subset_baseline_stats_gold.json", "subset_baseline"),
    ("MouseAtlas_subset_bbknn_stats.json",    "MouseAtlas_subset_bbknn_stats_gold.json",    "subset_bbknn"),
    ("mouse_bbknn_atlas_meta.json",           "mouse_bbknn_atlas_meta_gold.json",           "meta"),
]

REL_TOL_DEFAULT = 1e-9
KEYS_STATS_MUST = ["shape", "nnz", "avg_degree", "first100_hash"]
OPTIONAL_FLAGS = {"bbknn_disabled_due_to_mac_tbb_error"}

def _json_load(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _rel_close(a: float, b: float, tol: float) -> bool:
    if a == b:
        return True
    if b == 0:
        return abs(a) <= tol
    return abs(a - b) / max(1e-12, abs(b)) <= tol

def _print_header(title: str):
    print("\n" + "="*90)
    print(title)
    print("="*90)

def _diff_json(cur: Any, gold: Any, path: str = "", rel_tol: float = REL_TOL_DEFAULT) -> List[str]:
    diffs: List[str] = []
    def loc(p, sub=""):
        return p if not sub else (p + sub if p else sub)
    if type(cur) != type(gold):
        diffs.append(f"{path}: TYPE {type(cur).__name__} vs {type(gold).__name__} (cur={cur} gold={gold})")
        return diffs
    if isinstance(cur, dict):
        ck, gk = set(cur.keys()), set(gold.keys())
        ck_eff = ck - OPTIONAL_FLAGS
        gk_eff = gk - OPTIONAL_FLAGS
        missing = gk_eff - ck_eff
        extra   = ck_eff - gk_eff
        if missing:
            diffs.append(f"{path}: MISSING KEYS -> {sorted(missing)}")
        if extra:
            diffs.append(f"{path}: EXTRA KEYS -> {sorted(extra)}")
        for k in sorted(ck_eff & gk_eff):
            diffs += _diff_json(cur[k], gold[k], loc(path, f".{k}" if path else k), rel_tol)
        return diffs
    if isinstance(cur, list):
        if len(cur) != len(gold):
            diffs.append(f"{path}: LIST LEN cur={len(cur)} gold={len(gold)}")
        for i, (cv, gv) in enumerate(zip(cur, gold)):
            diffs += _diff_json(cv, gv, loc(path, f"[{i}]"), rel_tol)
        return diffs
    if _is_num(cur) and _is_num(gold):
        if not _rel_close(float(cur), float(gold), rel_tol):
            diffs.append(f"{path}: NUM cur={cur} gold={gold} tol={rel_tol}")
        return diffs
    if cur != gold:
        diffs.append(f"{path}: DIFF cur={cur} gold={gold}")
    return diffs

def _check_stats_shape(cur: Dict[str, Any], gold: Dict[str, Any], tag: str) -> bool:
    ok = True
    for k in KEYS_STATS_MUST:
        if k not in cur:
            print(f"[FAIL] {tag}: current missing key '{k}'")
            ok = False
        if k not in gold:
            print(f"[FAIL] {tag}: gold missing key '{k}'")
            ok = False
    if not ok:
        return False
    if cur["shape"] != gold["shape"]:
        print(f"[FAIL] {tag}: shape cur={cur['shape']} gold={gold['shape']}")
        ok = False
    else:
        print(f"[PASS] {tag}: shape match {cur['shape']}")
    if int(cur["nnz"]) != int(gold["nnz"]):
        print(f"[FAIL] {tag}: nnz cur={cur['nnz']} gold={gold['nnz']}")
        ok = False
    else:
        print(f"[PASS] {tag}: nnz match {cur['nnz']}")
    cd, gd = float(cur["avg_degree"]), float(gold["avg_degree"])
    if _rel_close(cd, gd, REL_TOL_DEFAULT):
        print(f"[PASS] {tag}: avg_degree close cur={cd} gold={gd}")
    else:
        print(f"[FAIL] {tag}: avg_degree diff cur={cd} gold={gd} tol={REL_TOL_DEFAULT}")
        ok = False
    if cur["first100_hash"] != gold["first100_hash"]:
        print(f"[FAIL] {tag}: first100_hash cur={cur['first100_hash']} gold={gold['first100_hash']}")
        ok = False
    else:
        print(f"[PASS] {tag}: first100_hash equal {cur['first100_hash']}")
    return ok

def eval_one(cur_path: str, gold_path: str, tag: str) -> bool:
    _print_header(f"CHECK: {tag}")
    if not os.path.isfile(cur_path):
        print(f"[FAIL] current missing: {cur_path}")
        return False
    if not os.path.isfile(gold_path):
        print(f"[FAIL] gold missing: {gold_path}")
        return False
    try:
        cur = _json_load(cur_path)
        gold = _json_load(gold_path)
    except Exception as e:
        print(f"[FAIL] JSON load error\n{e}\n{traceback.format_exc()}")
        return False
    if tag != "meta":
        ok_core = _check_stats_shape(cur, gold, tag)
        diffs = _diff_json(cur, gold, path=tag, rel_tol=REL_TOL_DEFAULT)
        if diffs:
            print("[FAIL] deep diff (first 50):")
            for d in diffs[:50]:
                print("  -", d)
            if len(diffs) > 50:
                print(f"  ... and {len(diffs)-50} more diffs")
            return False and ok_core
        if ok_core:
            print("[PASS] deep compare OK")
        return ok_core
    diffs = _diff_json(cur, gold, path=tag, rel_tol=REL_TOL_DEFAULT)
    if diffs:
        print("[FAIL] meta deep diff (first 50):")
        for d in diffs[:50]:
            print("  -", d)
        if len(diffs) > 50:
            print(f"  ... and {len(diffs)-50} more diffs")
        return False
    print("[PASS] meta deep compare OK")
    return True

def debug_eval() -> bool:
    all_ok = True
    pairs = []
    for cur_name, gold_name, tag in FILES:
        pairs.append((
            os.path.join(PRED_DIR, cur_name),
            os.path.join(GOLD_DIR, gold_name),
            tag
        ))
    _print_header("CHECK 0: Presence")
    missing = [(c, g, t) for c, g, t in pairs if (not os.path.isfile(c) or not os.path.isfile(g))]
    if missing:
        all_ok = False
        print("[FAIL] Missing files:")
        for c, g, t in missing:
            if not os.path.isfile(c):
                print(f"  - current missing ({t}): {c}")
            if not os.path.isfile(g):
                print(f"  - gold missing    ({t}): {g}")
    else:
        print("[PASS] All target files present")
    for cur_path, gold_path, tag in pairs:
        if os.path.isfile(cur_path) and os.path.isfile(gold_path):
            ok = eval_one(cur_path, gold_path, tag)
            all_ok = all_ok and ok
    _print_header("SUMMARY")
    if all_ok:
        print("[PASS] All comparisons passed.")
    else:
        print("[FAIL] See details above.")
    return all_ok

def eval() -> bool:
    return debug_eval()

if __name__ == "__main__":
    ok = debug_eval()
    print(ok)
