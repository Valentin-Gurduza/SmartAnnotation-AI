"""
SmartAnnotate-AI — Quality Metrics Module
==========================================
Inter-annotator agreement (Cohen's & Fleiss' Kappa),
human-vs-AI alignment scoring, and triage statistics.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from schemas import AnnotationRecord, AnnotationTask, HumanReview, RoutingDecision


# ──────────────────────────────────────────────
# Cohen's Kappa (2 annotators)
# ──────────────────────────────────────────────

def cohens_kappa(
    annotations_a: list[str],
    annotations_b: list[str],
) -> float:
    """
    Calculate Cohen's Kappa for two annotators.

    Args:
        annotations_a: Labels from annotator A.
        annotations_b: Labels from annotator B (same order).

    Returns:
        Kappa score between -1 and 1.
    """
    if len(annotations_a) != len(annotations_b):
        raise ValueError("Both annotation lists must have the same length.")
    if len(annotations_a) == 0:
        return 0.0

    return float(cohen_kappa_score(annotations_a, annotations_b))


# ──────────────────────────────────────────────
# Fleiss' Kappa (multi-annotator)
# ──────────────────────────────────────────────

def fleiss_kappa(
    ratings_matrix: list[list[int]],
) -> float:
    """
    Calculate Fleiss' Kappa for multiple annotators.

    Args:
        ratings_matrix: N x k matrix where N = number of items,
                        k = number of categories.
                        Each cell = number of annotators who assigned
                        that category to that item.
                        Row sums must all be equal (= number of annotators).

    Returns:
        Fleiss' Kappa score.
    """
    mat = np.array(ratings_matrix, dtype=float)
    N, k = mat.shape  # noqa: N806
    n = mat[0].sum()  # number of annotators per item

    if n <= 1:
        return 0.0

    # Proportion of all assignments to each category
    p_j = mat.sum(axis=0) / (N * n)

    # Per-item agreement
    P_i = (mat ** 2).sum(axis=1) - n  # noqa: N806
    P_i /= n * (n - 1)  # noqa: N806

    P_bar = P_i.mean()  # noqa: N806
    P_e = (p_j ** 2).sum()  # noqa: N806

    if P_e == 1.0:
        return 1.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return float(round(kappa, 4))


# ──────────────────────────────────────────────
# Human-vs-AI Alignment
# ──────────────────────────────────────────────

def human_ai_alignment(
    ai_labels: list[str],
    human_labels: list[str],
    label_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute alignment metrics between AI predictions and human annotations.

    Returns dict with accuracy, per-label precision/recall/f1, and confusion matrix.
    """
    if len(ai_labels) != len(human_labels):
        raise ValueError("Label lists must have the same length.")

    if not ai_labels:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "per_label": {},
            "confusion_matrix": [],
        }

    accuracy = accuracy_score(human_labels, ai_labels)

    if label_names is None:
        label_names = sorted(set(ai_labels + human_labels))

    precision, recall, f1, support = precision_recall_fscore_support(
        human_labels, ai_labels, labels=label_names, average=None, zero_division=0
    )

    per_label = {}
    for i, label in enumerate(label_names):
        per_label[label] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    # Macro averages
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        human_labels, ai_labels, average="macro", zero_division=0
    )

    cm = confusion_matrix(human_labels, ai_labels, labels=label_names)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "per_label": per_label,
        "confusion_matrix": cm.tolist(),
        "label_names": label_names,
    }


# ──────────────────────────────────────────────
# Triage & Pipeline Statistics
# ──────────────────────────────────────────────

def triage_statistics(tasks: list[AnnotationTask]) -> dict:
    """
    Compute triage and routing statistics from processed tasks.

    Returns dashboard-ready metrics.
    """
    if not tasks:
        return {
            "total": 0,
            "auto_approved": 0,
            "human_review": 0,
            "auto_approved_pct": 0.0,
            "human_review_pct": 0.0,
            "avg_confidence": 0.0,
            "confidence_distribution": {},
            "estimated_time_saved_pct": 0.0,
        }

    total = len(tasks)
    auto = sum(1 for t in tasks if t.routing == RoutingDecision.AUTO_APPROVED)
    human = total - auto

    confidences = [t.overall_confidence for t in tasks]
    avg_conf = sum(confidences) / total

    # Confidence distribution buckets
    dist = Counter()
    for c in confidences:
        if c >= 0.9:
            dist["0.9-1.0"] += 1
        elif c >= 0.8:
            dist["0.8-0.9"] += 1
        elif c >= 0.7:
            dist["0.7-0.8"] += 1
        elif c >= 0.6:
            dist["0.6-0.7"] += 1
        else:
            dist["<0.6"] += 1

    # Estimated time savings:
    # Auto-approved items need ~10s spot-check vs ~120s full review
    # Human review items still need full time
    full_review_time = total * 120  # seconds without AI
    ai_assisted_time = auto * 10 + human * 120
    time_saved_pct = (
        (1 - ai_assisted_time / full_review_time) * 100
        if full_review_time > 0
        else 0.0
    )

    return {
        "total": total,
        "auto_approved": auto,
        "human_review": human,
        "auto_approved_pct": round(auto / total * 100, 1),
        "human_review_pct": round(human / total * 100, 1),
        "avg_confidence": round(avg_conf, 4),
        "confidence_distribution": dict(sorted(dist.items())),
        "estimated_time_saved_pct": round(time_saved_pct, 1),
    }


# ──────────────────────────────────────────────
# Review Statistics
# ──────────────────────────────────────────────

def review_statistics(reviews: list[HumanReview]) -> dict:
    """Compute statistics from human reviews."""
    if not reviews:
        return {
            "total_reviews": 0,
            "accepted": 0,
            "edited": 0,
            "rejected": 0,
            "accepted_pct": 0.0,
            "edited_pct": 0.0,
            "rejected_pct": 0.0,
            "avg_review_time": 0.0,
        }

    total = len(reviews)
    actions = Counter(r.action for r in reviews)
    accepted = actions.get("accept", 0)
    edited = actions.get("edit", 0)
    rejected = actions.get("reject", 0)

    review_times = [
        r.time_spent_seconds
        for r in reviews
        if r.time_spent_seconds is not None
    ]
    avg_time = sum(review_times) / len(review_times) if review_times else 0.0

    return {
        "total_reviews": total,
        "accepted": accepted,
        "edited": edited,
        "rejected": rejected,
        "accepted_pct": round(accepted / total * 100, 1),
        "edited_pct": round(edited / total * 100, 1),
        "rejected_pct": round(rejected / total * 100, 1),
        "avg_review_time": round(avg_time, 1),
    }


# ──────────────────────────────────────────────
# Aggregate Quality Report
# ──────────────────────────────────────────────

def generate_quality_report(
    tasks: list[AnnotationTask],
    reviews: list[HumanReview],
    ai_labels: Optional[list[str]] = None,
    human_labels: Optional[list[str]] = None,
) -> dict:
    """
    Generate a comprehensive quality report combining all metrics.

    Returns a dict suitable for dashboard rendering.
    """
    report = {
        "triage": triage_statistics(tasks),
        "reviews": review_statistics(reviews),
    }

    if ai_labels and human_labels and len(ai_labels) == len(human_labels):
        report["alignment"] = human_ai_alignment(ai_labels, human_labels)

    return report
