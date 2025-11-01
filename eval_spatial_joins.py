import pandas as pd
from pathlib import Path

def eval():
    pred_file = Path("./pred_results/spatial_joins_result.csv")
    gold_file = Path("./benchmark/eval_programs/gold_results/spatial_joins_result_gold.csv")
    removed_file = Path("./benchmark/eval_programs/gold_results/spatial_joins_removed.csv")

    if not pred_file.exists():
        return False, "Prediction file not found"
    if not gold_file.exists():
        return False, "Gold file not found"

    pred = pd.read_csv(pred_file)
    gold = pd.read_csv(gold_file)

    pred = pred[sorted(pred.columns)]
    gold = gold[sorted(gold.columns)]

    for col in pred.select_dtypes(include=["object"]).columns:
        pred[col] = pred[col].astype(str).str.strip()
        gold[col] = gold[col].astype(str).str.strip()

    for col in pred.select_dtypes(include=["float", "int"]).columns:
        pred[col] = pred[col].round(6)
        gold[col] = gold[col].round(6)

    pred = pred.sort_values(by=pred.columns.tolist()).reset_index(drop=True)
    gold = gold.sort_values(by=gold.columns.tolist()).reset_index(drop=True)

    if removed_file.exists():
        removed = pd.read_csv(removed_file)
        for _, row in removed.iterrows():
            if ((pred == row).all(axis=1)).any():
                return False, f"Contamination detected: removed row found in prediction {row.to_dict()}"

    if pred.equals(gold):
        return True, "Files match exactly after normalization"

    diff_cells = (pred != gold)
    diff_report = []
    for col in pred.columns:
        mismatches = diff_cells[col].sum()
        if mismatches > 0:
            sample_rows = pred.loc[diff_cells[col], col].head(3).tolist()
            gold_vals = gold.loc[diff_cells[col], col].head(3).tolist()
            diff_report.append(f"Column '{col}' has {mismatches} mismatches. "
                               f"Examples: pred={sample_rows}, gold={gold_vals}")
    report = " | ".join(diff_report)
    return False, f"Files differ. {report}"

if __name__ == "__main__":
    success, info = eval()
    print(success, info)
