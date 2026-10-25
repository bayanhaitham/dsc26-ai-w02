
import math
import numpy as np


def check_inputs(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("Inputs must be 1-D")

    if len(y_true) == 0 or len(y_true) != len(scores):
        raise ValueError("Inputs must have the same non-zero length")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")

    try:
        scores = scores.astype(float)
    except (TypeError, ValueError):
        raise ValueError("Scores must be numeric")

    if np.isnan(scores).any():
        raise ValueError("Scores cannot contain NaN")

    return y_true.astype(int), scores


def confusion_at(y_true, scores, threshold: float) -> dict:
    y_true, scores = check_inputs(y_true, scores)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_true)):
        if scores[i] >= threshold:
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y_true[i] == 1:
                fn += 1
            else:
                tn += 1

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    result = confusion_at(y_true, scores, threshold)

    tp = result["tp"]
    fp = result["fp"]
    fn = result["fn"]

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    y_true, scores = check_inputs(y_true, scores)

    if k <= 0:
        raise ValueError("k must be positive")

    if k > len(y_true):
        k = len(y_true)

    order = np.argsort(-scores, kind="stable")
    correct = 0

    for i in range(k):
        if y_true[order[i]] == 1:
            correct += 1

    return float(correct / k)


def expected_cost(
    y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float
) -> float:
    result = confusion_at(y_true, scores, threshold)

    cost = result["fp"] * cost_fp + result["fn"] * cost_fn
    return float(cost)


def best_threshold(
    y_true, scores, *, cost_fp: float, cost_fn: float
) -> float:
    y_true, scores = check_inputs(y_true, scores)

    thresholds = list(np.unique(scores))
    thresholds.append(math.inf)

    best = math.inf
    lowest_cost = math.inf

    for threshold in thresholds:
        cost = expected_cost(
            y_true,
            scores,
            threshold,
            cost_fp=cost_fp,
            cost_fn=cost_fn
        )

        if cost < lowest_cost or (
            cost == lowest_cost and threshold > best
        ):
            lowest_cost = cost
            best = threshold

    return float(best)


def threshold_for_precision(
    y_true, scores, min_precision: float
) -> float | None:
    y_true, scores = check_inputs(y_true, scores)

    thresholds = np.unique(scores)
    valid_thresholds = []

    for threshold in thresholds:
        precision, recall = precision_recall_at(
            y_true, scores, threshold
        )

        if precision >= min_precision:
            valid_thresholds.append(threshold)

    if len(valid_thresholds) == 0:
        return None

    return float(min(valid_thresholds))
