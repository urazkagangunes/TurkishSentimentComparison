"""Three-output sentiment metrics, cluster bootstrap and paired comparisons."""
import numpy as np
from scipy.stats import binomtest

from src.evaluation.checkpoint_store import LABELS

PRIMARY_PAIRS = (("hisnet", "hisnet_dp"), ("hisnet", "dp_hisnet"),
                 ("dp_hisnet", "hisnet_dp"), ("hisnet_dp", "lw_hisnet_dp"),
                 ("llm", "dp_llm"))


def encoded(labels):
    return np.array([LABELS.index(x) if x in LABELS else -1 for x in labels], dtype=np.int8)


def prediction_summary(distribution):
    """Distribution rates use valid outputs, including inference-only datasets."""
    total = sum(distribution.values())
    neutral = distribution.get("neutral", 0)
    return dict(prediction_count=total, neutral_prediction_count=neutral,
                neutral_prediction_rate=neutral / total if total else None)


def confusion(gold, predictions):
    gold, predictions = np.asarray(gold), np.asarray(predictions)
    valid = (gold >= 0) & (predictions >= 0)
    return np.bincount(gold[valid].astype(int) * 3 + predictions[valid].astype(int), minlength=9).reshape(3, 3)


def _stats(matrices, active):
    """Vectorized statistics; absent classes remain zero only inside macro arithmetic."""
    m = np.asarray(matrices, dtype=float).reshape(-1, 3, 3)
    tp = np.diagonal(m, axis1=1, axis2=2)
    support, predicted = m.sum(axis=2), m.sum(axis=1)
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    p, r = divide(tp, predicted), divide(tp, support)
    f = divide(2 * tp, support + predicted)
    n = support.sum(axis=1)
    acc = divide(tp.sum(axis=1), n)
    denominator = np.sqrt((n*n - (support*support).sum(axis=1)) *
                          (n*n - (predicted*predicted).sum(axis=1)))
    mcc = divide(tp.sum(axis=1)*n - (support*predicted).sum(axis=1), denominator)
    return dict(accuracy=acc, error_rate=1-acc, macro_precision=p[:, active].mean(axis=1),
                macro_recall=r[:, active].mean(axis=1), macro_f1=f[:, active].mean(axis=1),
                weighted_f1=divide((f*support).sum(axis=1), n), mcc=mcc,
                precision=p, recall=r, f1=f)


def metrics(gold, predictions, active=None):
    gold, predictions = np.asarray(gold), np.asarray(predictions)
    active = list(active) if active is not None else sorted(set(gold[gold >= 0].tolist()))
    valid = (gold >= 0) & (predictions >= 0)
    n = int(valid.sum())
    matrix = confusion(gold, predictions)
    result = dict(n=n, macro_labels=[LABELS[i] for i in active], label_order=list(LABELS),
                  confusion_matrix=matrix.tolist(), classes={},
                  prediction_distribution={label: int((predictions == i).sum()) for i, label in enumerate(LABELS)})
    result.update(prediction_summary(result["prediction_distribution"]))
    stats = _stats(matrix, active) if n and active else None
    for key in ("accuracy", "error_rate", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "mcc"):
        result[key] = float(stats[key][0]) if stats else None
    support, pred = matrix.sum(axis=1), matrix.sum(axis=0)
    for i, label in enumerate(LABELS):
        tp = int(matrix[i, i])
        fp, fn = int(pred[i])-tp, int(support[i])-tp
        result["classes"][label] = dict(support=int(support[i]), predicted=int(pred[i]), correct=tp,
            tp=tp, fp=fp, fn=fn, tn=n-tp-fp-fn,
            precision=float(stats["precision"][0, i]) if support[i] and stats else None,
            recall=float(stats["recall"][0, i]) if support[i] and stats else None,
            f1=float(stats["f1"][0, i]) if support[i] and stats else None)
    result["confusion_row_percent"] = [
        [100 * int(v) / int(support[i]) for v in row] if support[i] else [None]*3
        for i, row in enumerate(matrix)]
    neutral_support = int(support[1])
    polar_support = int(support[0]+support[2])
    result["neutral_to_polar_rate"] = int(matrix[1, 0]+matrix[1, 2])/neutral_support if neutral_support else None
    result["polar_to_neutral_rate"] = int(matrix[0, 1]+matrix[2, 1])/polar_support if polar_support else None
    return result


def _cluster_draws(categories, groups, width, repetitions, seed):
    """Bootstrap groups, compressing identical contingency vectors for large corpora."""
    categories = np.asarray(categories, dtype=int)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    ngroups = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    if ngroups == len(categories):
        counts = np.bincount(categories, minlength=width)
        return rng.multinomial(len(categories), counts/counts.sum(), size=repetitions)
    vectors = np.zeros((ngroups, width), dtype=np.int64)
    np.add.at(vectors, (inverse, categories), 1)
    patterns, counts = np.unique(vectors, axis=0, return_counts=True)
    # Bound temporary allocation even when custom product/user groups are diverse.
    output = np.empty((repetitions, width), dtype=np.int64)
    batch = max(1, min(64, 1_000_000 // len(patterns)))
    for start in range(0, repetitions, batch):
        weights = rng.multinomial(ngroups, counts/counts.sum(), size=min(batch, repetitions-start))
        output[start:start+len(weights)] = weights @ patterns
    return output


def _intervals(stats, active):
    result = {}
    for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1", "mcc"):
        result[key] = np.quantile(stats[key], [.025, .975]).tolist()
    for key in ("precision", "recall", "f1"):
        for i in active:
            result[f"{LABELS[i]}_{key}"] = np.quantile(stats[key][:, i], [.025, .975]).tolist()
    return result


def bootstrap(gold, first, groups, active, repetitions=2000, seed=42, second=None):
    """Percentile 95% CI. Paired differences use the SAME sampled clusters."""
    gold, first = np.asarray(gold), np.asarray(first)
    valid = (gold >= 0) & (first >= 0)
    if second is not None:
        second = np.asarray(second)
        valid &= second >= 0
    if not valid.any() or not active or repetitions == 0:
        return None
    y, a, group = gold[valid].astype(int), first[valid].astype(int), np.asarray(groups)[valid]
    if second is None:
        draws = _cluster_draws(y*3+a, group, 9, repetitions, seed)
        stats = _stats(draws.reshape(-1, 3, 3), active)
    else:
        b = second[valid].astype(int)
        draws = _cluster_draws(y*9+a*3+b, group, 27, repetitions, seed).reshape(-1, 3, 3, 3)
        sa, sb = _stats(draws.sum(axis=3), active), _stats(draws.sum(axis=2), active)
        stats = {key: sb[key]-sa[key] for key in sa}
    return dict(method="paired_cluster_percentile" if second is not None else "cluster_percentile",
                confidence=.95, repetitions=repetitions, seed=seed,
                clusters=len(set(group.tolist())), intervals=_intervals(stats, active))


def paired(gold, first, second, active=None):
    gold, first, second = map(np.asarray, (gold, first, second))
    valid = (gold >= 0) & (first >= 0) & (second >= 0)
    y, a, b = gold[valid], first[valid], second[valid]
    good_a, good_b = a == y, b == y
    improved, harmed = int((~good_a & good_b).sum()), int((good_a & ~good_b).sum())
    wrong = int((~good_a).sum())
    correct = int(good_a.sum())
    n = len(y)
    ma, mb = metrics(y, a, active), metrics(y, b, active)
    delta = {k: mb[k]-ma[k] if n else None for k in ("accuracy", "error_rate", "macro_f1", "macro_precision", "macro_recall", "weighted_f1", "mcc")}
    class_delta = {l: {k: mb["classes"][l][k]-ma["classes"][l][k]
                      if ma["classes"][l][k] is not None else None for k in ("precision", "recall", "f1")}
                   for l in LABELS}
    return dict(n=n, improved=improved, harmed=harmed, net_correct=improved-harmed,
        both_correct=int((good_a & good_b).sum()),
        both_wrong_unchanged=int((~good_a & ~good_b & (a == b)).sum()),
        both_wrong_changed=int((~good_a & ~good_b & (a != b)).sum()),
        prediction_changed=int((a != b).sum()), repair_rate=improved/wrong if wrong else None,
        harm_rate=harmed/correct if correct else None, net_accuracy_gain=(improved-harmed)/n if n else None,
        mcnemar_exact_p=float(binomtest(improved, improved+harmed, p=.5).pvalue) if improved+harmed else (1.0 if n else None),
        deltas=delta, class_deltas=class_delta,
        transitions=[dict(gold=LABELS[g], before=LABELS[x], after=LABELS[z], count=int(((y == g)&(a == x)&(b == z)).sum()))
                     for g in range(3) for x in range(3) for z in range(3)],
        gains_by_gold={l: dict(improved=int(((y == i)&~good_a&good_b).sum()),
                              harmed=int(((y == i)&good_a&~good_b).sum())) for i, l in enumerate(LABELS)})


def holm(pairs):
    valid = [p for p in pairs if p["mcnemar_exact_p"] is not None]
    previous = 0.0
    for rank, pair in enumerate(sorted(valid, key=lambda x: x["mcnemar_exact_p"])):
        previous = min(1.0, max(previous, (len(valid)-rank)*pair["mcnemar_exact_p"]))
        pair["holm_adjusted_p"] = previous
    return pairs
