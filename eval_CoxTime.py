import json
import os

def main():
    pred_path = "./pred_results/CoxTime.json"
    gold_path = "./benchmark/eval_programs/gold_results/CoxTime_gold.json"

    if not os.path.exists(pred_path):
        print(f" Predicted result not found: {pred_path}")
        return
    if not os.path.exists(gold_path):
        print(f" Gold result not found: {gold_path}")
        return

    with open(pred_path, "r") as f_pred, open(gold_path, "r") as f_gold:
        pred = json.load(f_pred)
        gold = json.load(f_gold)

    if pred == gold:
        print(" Evaluation passed: pred_results and gold_results are identical.")
    else:
        print(" Evaluation failed: differences detected between pred and gold results.")
        print("\n--- Pred Results ---")
        print(json.dumps(pred, indent=4))
        print("\n--- Gold Results ---")
        print(json.dumps(gold, indent=4))

if __name__ == "__main__":
    main()
