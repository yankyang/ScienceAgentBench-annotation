from pathlib import Path
import pandas as pd
import numpy as np
from gpt4_visual_judge import encode_image, score_figure

NUM_RTOL = 1e-12
NUM_ATOL = 1e-12
MAX_ROWS_TO_SHOW = 10
MAX_COLS_TO_SHOW = 8

def _format_row_diffs(df_pred, df_gold, col, idx):
    lines = []
    for i in idx[:MAX_ROWS_TO_SHOW]:
        pv = df_pred.iloc[i][col]
        gv = df_gold.iloc[i][col]
        try:
            pvf = float(pv); gvf = float(gv)
            absd = abs(pvf - gvf)
            reld = absd / (abs(gvf) + 1e-12)
            lines.append(f"row {i}: pred={pvf:.6g}, gold={gvf:.6g}, abs_diff={absd:.3g}, rel_diff={reld:.3g}")
        except Exception:
            lines.append(f"row {i}: pred={pv!r}, gold={gv!r}")
    if len(idx) > MAX_ROWS_TO_SHOW:
        lines.append(f"... (+{len(idx) - MAX_ROWS_TO_SHOW} more)")
    return "\n".join(lines)

def _compare_csv_strict(pred_csv: Path, gold_csv: Path):
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

    cols_pred = list(df_pred.columns)
    cols_gold = list(df_gold.columns)
    if set(cols_pred) != set(cols_gold):
        missing_in_pred = sorted(list(set(cols_gold) - set(cols_pred)))
        extra_in_pred   = sorted(list(set(cols_pred) - set(cols_gold)))
        return False, (
            "Column set mismatch:\n"
            f"- Missing in pred: {missing_in_pred}\n"
            f"- Extra in pred: {extra_in_pred}"
        )

    order_note = ""
    if cols_pred != cols_gold:
        order_note = f" (NOTE: column order differs; pred order={cols_pred})"

    df_pred = df_pred[cols_gold]

    if len(df_pred) != len(df_gold):
        return False, f"Row count mismatch: pred={len(df_pred)}, gold={len(df_gold)}{order_note}"
    diff_report = []
    diff_cols = []
    for col in cols_gold:
        s_pred = df_pred[col]
        s_gold = df_gold[col]

        pred_num = pd.to_numeric(s_pred, errors="coerce")
        gold_num = pd.to_numeric(s_gold, errors="coerce")
        pred_num_isnum = pred_num.notna().mean() > 0.9 and gold_num.notna().mean() > 0.9

        if pred_num_isnum:
            close_mask = np.isclose(pred_num.values, gold_num.values, rtol=NUM_RTOL, atol=NUM_ATOL, equal_nan=True)
            bad_idx = np.where(~close_mask)[0]
            if len(bad_idx) > 0:
                diff_cols.append(col)
                max_abs = float(np.nanmax(np.abs(pred_num.values - gold_num.values)))
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = np.abs(pred_num.values - gold_num.values) / (np.abs(gold_num.values) + 1e-12)
                max_rel = float(np.nanmax(rel))
                examples = _format_row_diffs(df_pred, df_gold, col, bad_idx)
                diff_report.append(
                    f"[{col}] {len(bad_idx)} differing rows; max_abs={max_abs:.6g}, max_rel={max_rel:.6g}\n{examples}"
                )
        else:
            neq_mask = (s_pred.astype(str).values != s_gold.astype(str).values)
            bad_idx = np.where(neq_mask)[0]
            if len(bad_idx) > 0:
                diff_cols.append(col)
                examples = _format_row_diffs(df_pred, df_gold, col, bad_idx)
                diff_report.append(f"[{col}] {len(bad_idx)} differing rows (string compare)\n{examples}")

        if len(diff_report) >= MAX_COLS_TO_SHOW:
            diff_report.append("... too many differing columns; truncating report")
            break

    if diff_report:
        return False, "CSV differs" + order_note + "\n" + "\n\n".join(diff_report)

    return True, f"CSV exact match within tolerance (rows={len(df_pred)}, cols={len(cols_gold)})" + order_note

def _compare_png_gpt4(pred_png: Path, gold_png: Path):
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
    pred_csv = Path("pred_results/pycox_introduction_pred.csv")
    gold_csv = Path("benchmark/eval_programs/gold_results/pycox_introduction_pred_gold.csv")
    pred_png = Path("pred_results/pycox_introduction_plot.png")
    gold_png = Path("benchmark/eval_programs/gold_results/pycox_introduction_plot_gold.png")

    csv_ok, csv_msg = _compare_csv_strict(pred_csv, gold_csv)
    png_ok, png_msg = _compare_png_gpt4(pred_png, gold_png)

    ok = bool(csv_ok and png_ok)
    return ok, f"[CSV] {csv_msg}\n[PNG] {png_msg}"

if __name__ == "__main__":
    ok, msg = eval()
    print(ok, msg)
