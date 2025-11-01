from pathlib import Path
import pandas as pd
import numpy as np
from gpt4_visual_judge import encode_image, score_figure

RTOL = 1e-3
ATOL = 1e-3

def _compare_csv(pred_csv: Path, gold_csv: Path):
    if not pred_csv.exists() and not gold_csv.exists():
        return False, "Missing both CSV files"
    if not pred_csv.exists():
        return False, f"Missing prediction CSV: {pred_csv}"
    if not gold_csv.exists():
        return False, f"Missing gold CSV: {gold_csv}"

    try:
        df_pred = pd.read_csv(pred_csv)
        df_gold = pd.read_csv(gold_csv)
    except Exception as e:
        return False, f"Error reading CSV files: {e}"

    # 列检查
    if set(df_pred.columns) != set(df_gold.columns):
        return False, f"Column mismatch: pred={list(df_pred.columns)}, gold={list(df_gold.columns)}"

    # 对齐列顺序
    df_pred = df_pred[df_gold.columns]

    if len(df_pred) != len(df_gold):
        return False, f"Row count mismatch: pred={len(df_pred)}, gold={len(df_gold)}"

    # 数值列比较
    numeric_cols = [c for c in df_gold.columns if df_gold[c].dtype != object]
    for col in numeric_cols:
        pred_vals = pd.to_numeric(df_pred[col], errors="coerce").values
        gold_vals = pd.to_numeric(df_gold[col], errors="coerce").values
        if not np.allclose(pred_vals, gold_vals, rtol=RTOL, atol=ATOL, equal_nan=True):
            diffs = np.abs(pred_vals - gold_vals)
            max_abs = np.nanmax(diffs)
            return False, f"Column {col} mismatch, max_abs_diff={max_abs:.4g}"

    return True, f"CSV matched within tolerance (rows={len(df_pred)}, cols={len(df_pred.columns)})"


def _compare_png(pred_png: Path, gold_png: Path):
    if not pred_png.exists() and not gold_png.exists():
        return False, "Missing both PNG files"
    if not pred_png.exists():
        return False, f"Missing prediction plot: {pred_png}"
    if not gold_png.exists():
        return False, f"Missing gold plot: {gold_png}"
    if pred_png.stat().st_size == 0:
        return False, "Prediction plot file empty"
    if gold_png.stat().st_size == 0:
        return False, "Gold plot file empty"

    pred_fig = encode_image(str(pred_png))
    gold_fig = encode_image(str(gold_png))
    full_response, score = score_figure(pred_fig, gold_fig)

    if score < 60:
        return False, f"Plot similarity too low (score={score}); response={full_response}"
    return True, f"Plot OK (score={score}); response={full_response}"


def eval():
    pred_csv = Path("pred_results/pycox_introduction2_pred.csv")
    gold_csv = Path("benchmark/eval_programs/gold_results/pycox_introduction2_pred_gold.csv")
    pred_png = Path("pred_results/pycox_introduction2_plot.png")
    gold_png = Path("benchmark/eval_programs/gold_results/pycox_introduction2_plot_gold.png")

    csv_ok, csv_msg = _compare_csv(pred_csv, gold_csv)
    png_ok, png_msg = _compare_png(pred_png, gold_png)

    ok = bool(csv_ok and png_ok)
    return ok, f"[CSV] {csv_msg}\n[PNG] {png_msg}"


if __name__ == "__main__":
    success, message = eval()
    print(success, message)
