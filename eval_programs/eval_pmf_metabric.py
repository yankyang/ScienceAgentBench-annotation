import json
from gpt4_visual_judge import encode_image, score_figure

def eval():
    pred_metrics = "pred_results/pmf_metabric_metrics.json"
    gold_metrics = "benchmark/eval_programs/gold_results/pmf_metabric_metrics_gold.json"
    pred_fig = "pred_results/pmf_metabric_surv.png"
    gold_fig = "benchmark/eval_programs/gold_results/pmf_metabric_surv_gold.png"

    pred_encoded = encode_image(pred_fig)
    gold_encoded = encode_image(gold_fig)
    full_response, fig_score = score_figure(pred_encoded, gold_encoded)

    with open(pred_metrics) as f:
        pred = json.load(f)
    with open(gold_metrics) as f:
        gold = json.load(f)

    pred_val = float(pred.get("concordance_td", 0))
    gold_val = float(gold.get("concordance_td", 0))
    metric_diff = abs(pred_val - gold_val)
    metric_pass = metric_diff < 1e-3 

    passed = (fig_score >= 60) and metric_pass
    result = {
        "figure_score": fig_score,
        "metric_diff": metric_diff,
        "metric_pass": metric_pass,
        "passed": int(passed)
    }

    print(json.dumps(result, indent=2))
    return int(passed), full_response

if __name__ == "__main__":
    eval()
