import os, json, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from gpt4_visual_judge import encode_image, score_figure

PRED_DIR = Path("./pred_results")
GOLD_DIR = Path("./benchmark/eval_programs/gold_results")
OUT_PATH = Path("./benchmark/eval_programs/eval_logs/deephit_competing_risks_eval.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

METRICS_PRED = PRED_DIR / "deephit_competing_risks_metrics.json"
METRICS_GOLD = GOLD_DIR / "deephit_competing_risks_metrics_gold.json"
CIF_PRED = PRED_DIR / "deephit_competing_risks_cif.png"
CIF_GOLD = GOLD_DIR / "deephit_competing_risks_cif_gold.png"
GOLD_CSV = GOLD_DIR / "deephit_competing_risks_gold.csv"

def compare_metrics(pred, gold, tol=1e-3):
    ok = True
    diffs = {}
    for key in gold:
        if key not in pred:
            ok = False
            diffs[key] = "missing_in_pred"
        else:
            diff = abs(pred[key] - gold[key])
            diffs[key] = diff
            if diff > tol:
                ok = False
    return ok, diffs

def eval_visual(pred_path, gold_path):
    pred_fig = encode_image(pred_path)
    gold_fig = encode_image(gold_path)
    full_response, score = score_figure(pred_fig, gold_fig)
    passed = int(score >= 60)
    return passed, float(score), full_response

def main():
    result = {"status": "ok"}
    try:
        print("Loading JSON metrics...")
        pred_metrics = json.load(open(METRICS_PRED))
        gold_metrics = json.load(open(METRICS_GOLD))

        metrics_ok, metric_diffs = compare_metrics(pred_metrics, gold_metrics)
        print(f"Metrics comparison: {'PASS' if metrics_ok else 'FAIL'}")
        print("   Diffs:", metric_diffs)
        result["metrics_match"] = metrics_ok
        result["metric_diffs"] = metric_diffs

        print("\n  Evaluating CIF image similarity (via GPT-4V judge)...")
        cif_pass, cif_score, cif_response = eval_visual(CIF_PRED, CIF_GOLD)
        print(f" CIF visual score: {cif_score:.2f} ({'PASS' if cif_pass else 'FAIL'})")
        result["cif_pass"] = bool(cif_pass)
        result["cif_score"] = cif_score
        result["cif_response"] = cif_response

        csv_ok = GOLD_CSV.exists()
        print(f"\n📄 Gold CSV exists: {csv_ok}")
        result["gold_csv_exists"] = csv_ok

        result["all_passed"] = metrics_ok and cif_pass and csv_ok
        print(f"\n Final Evaluation: {'ALL PASSED ' if result['all_passed'] else 'FAILED '}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print("\n💥 Error during evaluation:")
        print(traceback.format_exc())

    json.dump(result, open(OUT_PATH, "w"), indent=2)
    print(f"\n Evaluation results saved to: {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
