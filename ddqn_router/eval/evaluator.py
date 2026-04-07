"""Metrics computation and bucketed evaluation for routing."""

from __future__ import annotations

from ddqn_router.rl.reward import jaccard_similarity


def _precision_recall_f1(
    predicted: set[int], target: set[int]
) -> tuple[float, float, float]:
    if not predicted and not target:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not target:
        return 0.0, 0.0, 0.0
    tp = len(predicted & target)
    precision = tp / len(predicted)
    recall = tp / len(target)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _bucket_name(set_size: int) -> str:
    if set_size <= 3:
        return "small_2_3"
    elif set_size <= 5:
        return "medium_4_5"
    else:
        return "large_6plus"


def evaluate_routing(
    predictions: list[set[int]],
    targets: list[set[int]],
) -> dict:
    """Compute full metrics suite over a list of predictions vs targets."""
    n = len(predictions)
    if n == 0:
        return {}

    jaccards: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    exact_matches = 0
    successes = 0

    bucket_data: dict[str, dict[str, list[float]]] = {}

    for pred, tgt in zip(predictions, targets):
        jacc = jaccard_similarity(pred, tgt)
        p, r, f1 = _precision_recall_f1(pred, tgt)

        jaccards.append(jacc)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        if pred == tgt:
            exact_matches += 1
        if tgt.issubset(pred):
            successes += 1

        bucket = _bucket_name(len(tgt))
        if bucket not in bucket_data:
            bucket_data[bucket] = {
                "jaccards": [],
                "precisions": [],
                "recalls": [],
                "f1s": [],
            }
        bucket_data[bucket]["jaccards"].append(jacc)
        bucket_data[bucket]["precisions"].append(p)
        bucket_data[bucket]["recalls"].append(r)
        bucket_data[bucket]["f1s"].append(f1)

    result: dict = {
        "mean_jaccard": sum(jaccards) / n,
        "success_rate": successes / n,
        "exact_match_rate": exact_matches / n,
        "mean_precision": sum(precisions) / n,
        "mean_recall": sum(recalls) / n,
        "mean_f1": sum(f1s) / n,
        "n_samples": n,
    }

    for bucket, data in sorted(bucket_data.items()):
        bn = len(data["jaccards"])
        result[f"bucket_{bucket}"] = {
            "n": bn,
            "mean_jaccard": sum(data["jaccards"]) / bn,
            "mean_precision": sum(data["precisions"]) / bn,
            "mean_recall": sum(data["recalls"]) / bn,
            "mean_f1": sum(data["f1s"]) / bn,
        }

    return result


def print_metrics(metrics: dict, label: str = "Evaluation") -> None:
    print(f"\n  {label}")
    print(f"  {'─' * 50}")
    print(f"  Samples:      {metrics.get('n_samples', 'N/A')}")
    print(f"  Jaccard:      {metrics.get('mean_jaccard', 0):.4f}")
    print(f"  Success rate: {metrics.get('success_rate', 0):.4f}")
    print(f"  Exact match:  {metrics.get('exact_match_rate', 0):.4f}")
    print(f"  Precision:    {metrics.get('mean_precision', 0):.4f}")
    print(f"  Recall:       {metrics.get('mean_recall', 0):.4f}")
    print(f"  F1:           {metrics.get('mean_f1', 0):.4f}")

    for key in sorted(metrics.keys()):
        if key.startswith("bucket_"):
            bucket = metrics[key]
            name = key.replace("bucket_", "").replace("_", " ")
            print(f"\n  [{name}] (n={bucket['n']})")
            print(f"    Jaccard: {bucket['mean_jaccard']:.4f}  "
                  f"P: {bucket['mean_precision']:.4f}  "
                  f"R: {bucket['mean_recall']:.4f}  "
                  f"F1: {bucket['mean_f1']:.4f}")
    print()
