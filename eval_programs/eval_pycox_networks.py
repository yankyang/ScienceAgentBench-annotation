from pathlib import Path
import pandas as pd
import numpy as np
from gpt4_visual_judge import encode_image, score_figure

RTOL = 1e-12
ATOL = 1e-12
MAX_EXAMPLES = 5

def _compare_csv(pred_csv: Path, gold_csv: Path):
    if not pred_csv.exists() or not gold_csv.exists():
        return False, "Missing CSV file(s)"

    try:
        df_pred = pd.read_csv(pred_csv)
        df_gold = pd.read_csv(gold_csv)
    except Exception as e:
        return False, f"Error reading CSV: {e}"

    if list(df_pred.columns) != list(df_gold.columns):
        return False, f"Column mismatch: pred={list(df_pred.columns)}, gold={list(df_gold.columns)}"

    if len(df_pred) != len(df_gold):
        return False, f"Row count mismatch: pred={len(df_pred)}, gold={len(df_gold)}"

    diffs = []
    for col in df_pred.columns:
        pred_vals = df_pred[col].values
        gold_vals = df_gold[col].values

        # 尝试转成数值比较，否则用字符串比较
        try:
            pred_num = pd.to_numeric(pred_vals, errors="coerce")
            gold_num = pd.to_numeric(gold_vals, errors="coerce")
            if pred_num.notna().all() and gold_num.notna().all():
                mask = ~np.isclose(pred_num, gold_num, rtol=RTOL, atol=ATOL, equal_nan=True)
                idx = np.where(mask)[0]
                if len(idx) > 0:
                    sample = [f"row {i}: pred={pred_num[i]}, gold={gold_num[i]}" for i in idx[:MAX_EXAMPLES]]
                    diffs.append(f"Column {col} differs in {len(idx)} rows\n" + "\n".join(sample))
            else:
                if not (pred_vals == gold_vals).all():
                    idx = np.where(pred_vals != gold_vals)[0]
                    sample = [f"row {i}: pred={pred_vals[i]}, gold={gold_vals[i]}" for i in idx[:MAX_EXAMPLES]]
                    diffs.append(f"Column {col} differs in {len(idx)} rows\n" + "\n".join(sample))
        except Exception:
            if not (pred_vals == gold_vals).all():
                idx = np.where(pred_vals != gold_vals)[0]
                sample = [f"row {i}: pred={pred_vals[i]}, gold={gold_vals[i]}" for i in idx[:MAX_EXAMPLES]]
                diffs.append(f"Column {col} differs in {len(idx)} rows\n" + "\n".join(sample))

    if diffs:
        return False, "CSV mismatch:\n" + "\n\n".join(diffs)

    return True, f"CSV exact match (rows={len(df_pred)}, cols={len(df_pred.columns)})"


def _compare_png(pred_png: Path, gold_png: Path):
    if not pred_png.exists() or not gold_png.exists():
        return False, "Missing PNG file(s)"
    if pred_png.stat().st_size == 0 or gold_png.stat().st_size == 0:
        return False, "One of the PNG files is empty"

    pred_fig = encode_image(str(pred_png))
    gold_fig = encode_image(str(gold_png))
    full_response, score = score_figure(pred_fig, gold_fig)

    if score < 60:
        return False, f"Plot similarity too low (score={score}); response={full_response}"
    return True, f"Plot OK (score={score}); response={full_response}"


def eval():
    pred_csv = Path("pred_results/pycox_networks_pred.csv")
    gold_csv = Path("benchmark/eval_programs/gold_results/pycox_networks_pred_gold.csv")
    pred_png = Path("pred_results/pycox_networks_plot.png")
    gold_png = Path("benchmark/eval_programs/gold_results/pycox_networks_plot_gold.png")

    csv_ok, csv_msg = _compare_csv(pred_csv, gold_csv)
    png_ok, png_msg = _compare_png(pred_png, gold_png)

    ok = bool(csv_ok and png_ok)
    return ok, f"[CSV] {csv_msg}\n[PNG] {png_msg}"


if __name__ == "__main__":
    success, message = eval()
    print(success, message)
