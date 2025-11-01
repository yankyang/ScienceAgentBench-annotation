import os
import json

def main():
    pred_path = "./pred_results/deephit_results.json"
    gold_path = "./benchmark/eval_programs/gold_results/deephit_gold_results.json"

    if not os.path.exists(pred_path):
        print(f" Prediction file not found at {pred_path}")
        return
    if not os.path.exists(gold_path):
        print(f" Gold file not found at {gold_path}")
        return

    with open(pred_path, "r") as f:
        pred_data = json.load(f)
    with open(gold_path, "r") as f:
        gold_data = json.load(f)

    same = (pred_data == gold_data)
    if same:
        print(" DeepHit prediction matches gold results perfectly.")
    else:
        print(" Differences detected between prediction and gold results:")
        for key in pred_data.keys():
            if key in gold_data:
                if pred_data[key] != gold_data[key]:
                    print(f"  - {key}: pred={pred_data[key]} | gold={gold_data[key]}")
            else:
                print(f"  - {key}: only in pred file")

        for key in gold_data.keys():
            if key not in pred_data:
                print(f"  - {key}: only in gold file")

if __name__ == "__main__":
    main()
