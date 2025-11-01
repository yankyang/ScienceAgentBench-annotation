import json
from gpt4_visual_judge import encode_image, score_figure

def eval():
    pred_path = "pred_results/mtlr_metabric_surv.png"
    gold_path = "benchmark/eval_programs/gold_results/mtlr_metabric_surv_gold.png"

    pred_fig = encode_image(pred_path)
    gold_fig = encode_image(gold_path)
    full_response, score = score_figure(pred_fig, gold_fig)

    with open("pred_results/mtlr_metabric_metrics.json") as f:
        pred_metrics = json.load(f)
    with open("benchmark/eval_programs/gold_results/mtlr_metabric_metrics_gold.json") as f:
        gold_metrics = json.load(f)
    concordance_close = abs(pred_metrics["concordance_td"] - gold_metrics["concordance_td"]) < 0.05

    passed = int(score >= 60 and concordance_close)
    result = {
        "passed": passed,
        "visual_score": score,
        "concordance_td_pred": pred_metrics["concordance_td"],
        "concordance_td_gold": gold_metrics["concordance_td"],
        "judge_feedback": full_response
    }

    print(json.dumps(result, indent=2))
    return passed, result

if __name__ == "__main__":
    eval()
