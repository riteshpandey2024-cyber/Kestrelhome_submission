import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .features import GOODWILL_COST


def topk_report(y, score, amount, k):
    """Review the k highest-scored claims. Fraud stopped = claim amounts of frauds in the queue;
    goodwill = Rs 380 per genuine claim held (ops-policy s4)."""
    idx = np.argsort(-score)[:k]
    y, amount = np.asarray(y), np.asarray(amount)
    caught = y[idx] == 1
    stopped = float(amount[idx][caught].sum())
    goodwill = float((~caught).sum() * GOODWILL_COST)
    return dict(k=int(k), frauds_caught=int(caught.sum()), frauds_total=int(y.sum()),
                precision=float(caught.mean()), recall=float(caught.sum() / max(y.sum(), 1)),
                fraud_inr_stopped=stopped, goodwill_inr=goodwill,
                net_inr=stopped - goodwill, stopped_per_check=stopped / k,
                net_per_check=(stopped - goodwill) / k)


def summary(y, score, amount, k):
    y = np.asarray(y)
    return dict(n=int(len(y)), frauds=int(y.sum()), base_rate=float(y.mean()),
                auc=float(roc_auc_score(y, score)), pr_auc=float(average_precision_score(y, score)),
                accuracy_always_legit=float(1 - y.mean()), **topk_report(y, score, amount, k))
