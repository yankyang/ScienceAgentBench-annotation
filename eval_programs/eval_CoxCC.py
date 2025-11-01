import json
import os

def main():
    pred_path = './pred_results/logistic_hazard_results.json'
    gold_path = './benchmark/eval_programs/gold_results/logistic_hazard_gold_results.json'

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    with open(pred_path, 'r') as f:
        pred = json.load(f)
    with open(gold_path, 'r') as f:
        gold = json.load(f)

    if pred == gold:
        print("Files are identical.")
    else:
        print("Files are different.")
        pred_keys = set(pred.keys())
        gold_keys = set(gold.keys())

        only_in_pred = pred_keys - gold_keys
        only_in_gold = gold_keys - pred_keys
        common_keys = pred_keys & gold_keys

        if only_in_pred:
            print("Keys only in pred:", list(only_in_pred))
        if only_in_gold:
            print("Keys only in gold:", list(only_in_gold))

        for k in common_keys:
            if pred[k] != gold[k]:
                print(f"Difference in '{k}': pred={pred[k]}, gold={gold[k]}")

if __name__ == "__main__":
    main()
