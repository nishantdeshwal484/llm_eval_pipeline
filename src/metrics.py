from typing import List, Dict, Any
from src.schemas import TestCase

def compute_cohens_kappa(gold_labels: List[str], predicted_labels: List[str]) -> float:
    if not gold_labels: return 0.0
    categories = list(set(gold_labels + predicted_labels))
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    y1 = [cat_to_int[lbl] for lbl in gold_labels]
    y2 = [cat_to_int[lbl] for lbl in predicted_labels]
    n = len(y1)
    
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    freq1 = {i: y1.count(i) / n for i in range(len(categories))}
    freq2 = {i: y2.count(i) / n for i in range(len(categories))}
    pe = sum(freq1.get(i, 0) * freq2.get(i, 0) for i in range(len(categories)))

    if pe == 1.0: return 1.0
    return round((po - pe) / (1.0 - pe), 4)


def aggregate_pipeline_results(eval_results: List[Dict[str, Any]], test_cases: List[TestCase]) -> Dict[str, Any]:
    total_cases = len(eval_results)
    if total_cases == 0: return {}

    flips = sum(1 for r in eval_results if r["position_flip_detected"])
    winners = [r["consensus_winner"] for r in eval_results]
    
    case_map = {tc.test_id: tc for tc in test_cases}
    
    # Calculate Kappa lists
    gold_list = []
    pred_list = []
    
    # Calculate Adversarial metrics
    adv_fooled_count = 0
    total_adv_cases = 0

    for r in eval_results:
        tc = case_map.get(r["test_id"])
        if tc and tc.gold_winner:
            gold_list.append(tc.gold_winner.value)
            pred_list.append(r["consensus_winner"])

        # Check for adversarial resistance
        if tc and tc.is_adversarial_probe:
            total_adv_cases += 1
            # If the verbose/wrong model (Model A) won, the judge was fooled
            if r["consensus_winner"] == "Model A":
                adv_fooled_count += 1

    kappa_score = compute_cohens_kappa(gold_list, pred_list) if gold_list else 0.0
    adv_resistance = ((total_adv_cases - adv_fooled_count) / max(total_adv_cases, 1)) * 100.0 if total_adv_cases > 0 else 100.0

    return {
        "total_evaluated": total_cases,
        "position_flip_rate_percent": round((flips / total_cases) * 100.0, 2),
        "cohens_kappa_gold_agreement": kappa_score,
        "adversarial_probe_resistance_percent": round(adv_resistance, 2),
        "declared_winner": max(set(winners), key=winners.count) if winners else "Tie"
    }