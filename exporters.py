"""
SmartAnnotate-AI — Multi-Format Exporters
==========================================
Export annotated data in Hugging Face JSONL, CoNLL/BIO,
and DPO/RLHF preference pair formats.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from schemas import (
    AnnotationRecord,
    AnnotationTask,
    CoNLLRecord,
    CoNLLToken,
    DPOPair,
    HuggingFaceRecord,
    HumanReview,
    NEREntity,
    TaskType,
)

EXPORT_DIR = Path("data/exports")


def _ensure_export_dir() -> Path:
    """Create export directory if it doesn't exist."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORT_DIR


def _timestamp_filename(prefix: str, ext: str) -> str:
    """Generate a timestamped filename."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


# ──────────────────────────────────────────────
# Hugging Face JSONL Export
# ──────────────────────────────────────────────

def export_huggingface_jsonl(
    tasks: list[AnnotationTask],
    reviews: Optional[dict[str, HumanReview]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Export classification annotations in Hugging Face Datasets JSONL format.

    Format: {"text": "...", "label": "...", "label_id": N, "metadata": {...}}

    Args:
        tasks: List of annotation tasks with predictions.
        reviews: Optional dict mapping task_id -> HumanReview for final labels.
        output_path: Custom output path; auto-generated if None.

    Returns:
        Path to the exported file.
    """
    _ensure_export_dir()
    if output_path is None:
        output_path = str(EXPORT_DIR / _timestamp_filename("hf_dataset", "jsonl"))

    # Build label-to-id mapping
    label_set: set[str] = set()
    for task in tasks:
        label = _get_final_label(task, reviews)
        if label:
            label_set.add(label)
    label_to_id = {label: idx for idx, label in enumerate(sorted(label_set))}

    records = []
    for task in tasks:
        label = _get_final_label(task, reviews)
        if not label:
            continue

        record = HuggingFaceRecord(
            text=task.record.text,
            label=label,
            label_id=label_to_id.get(label),
            metadata={
                "record_id": task.record.id,
                "source": task.record.source or "",
                "language": task.record.language or "en",
                "ai_confidence": task.overall_confidence,
                "routing": task.routing,
                "model": task.model_used,
            },
        )
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")

    return output_path


def _get_final_label(
    task: AnnotationTask,
    reviews: Optional[dict[str, HumanReview]] = None,
) -> Optional[str]:
    """Get the final label for a task, considering human review edits."""
    review = reviews.get(task.id) if reviews else None

    if task.task_type == TaskType.CLASSIFICATION:
        if review and review.action == "edit" and review.edited_sentiment:
            return f"{review.edited_sentiment}_{review.edited_intent or 'other'}"
        elif task.classification:
            return f"{task.classification.sentiment}_{task.classification.intent}"
    elif task.task_type == TaskType.NER:
        # For NER, use a simplified label
        if review and review.action == "edit":
            return "ner_edited"
        elif task.ner:
            return "ner_auto"
    return None


# ──────────────────────────────────────────────
# CoNLL / BIO Format Export (NER)
# ──────────────────────────────────────────────

def export_conll_bio(
    tasks: list[AnnotationTask],
    reviews: Optional[dict[str, HumanReview]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Export NER annotations in CoNLL / BIO tagging format.

    Format: One token per line with BIO tag, blank line between sentences.
    Example:
        John    B-PERSON
        Smith   I-PERSON
        works   O
        at      O
        Google  B-ORG

    Args:
        tasks: List of NER annotation tasks.
        reviews: Optional human review overrides.
        output_path: Custom output path.

    Returns:
        Path to the exported file.
    """
    _ensure_export_dir()
    if output_path is None:
        output_path = str(EXPORT_DIR / _timestamp_filename("conll_ner", "txt"))

    lines: list[str] = []

    for task in tasks:
        if task.task_type != TaskType.NER:
            continue

        entities = _get_final_entities(task, reviews)
        text = task.record.text
        tokens = _tokenize_with_offsets(text)

        # Assign BIO tags to tokens
        tagged_tokens = _assign_bio_tags(tokens, entities)

        conll_record = CoNLLRecord(
            tokens=[CoNLLToken(token=t, bio_tag=tag) for t, _, _, tag in tagged_tokens],
            raw_text=text,
        )

        # Write in CoNLL format
        lines.append(f"# text: {text}")
        for ct in conll_record.tokens:
            lines.append(f"{ct.token}\t{ct.bio_tag}")
        lines.append("")  # Blank line between records

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def _get_final_entities(
    task: AnnotationTask,
    reviews: Optional[dict[str, HumanReview]] = None,
) -> list[NEREntity]:
    """Get final entities considering human edits."""
    review = reviews.get(task.id) if reviews else None

    if review and review.action == "edit" and review.edited_entities:
        return review.edited_entities
    elif task.ner:
        return task.ner.entities
    return []


def _tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """
    Simple whitespace + punctuation tokenizer that preserves character offsets.

    Returns list of (token, start_offset, end_offset).
    """
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def _assign_bio_tags(
    tokens: list[tuple[str, int, int]],
    entities: list[NEREntity],
) -> list[tuple[str, int, int, str]]:
    """
    Assign BIO tags to tokens based on entity character offsets.

    Returns list of (token, start, end, bio_tag).
    """
    result = []
    for token_text, tok_start, tok_end in tokens:
        tag = "O"
        for entity in entities:
            # Check if token overlaps with entity span
            if tok_start >= entity.start and tok_end <= entity.end:
                # Token is within entity
                if tok_start == entity.start:
                    tag = f"B-{entity.label}"
                else:
                    tag = f"I-{entity.label}"
                break
            elif tok_start < entity.end and tok_end > entity.start:
                # Partial overlap — assign B tag
                if tok_start <= entity.start:
                    tag = f"B-{entity.label}"
                else:
                    tag = f"I-{entity.label}"
                break
        result.append((token_text, tok_start, tok_end, tag))
    return result


# ──────────────────────────────────────────────
# DPO / RLHF Preference Pairs
# ──────────────────────────────────────────────

def export_dpo_pairs(
    tasks: list[AnnotationTask],
    reviews: dict[str, HumanReview],
    output_path: Optional[str] = None,
) -> str:
    """
    Export DPO preference pairs from accept/reject decisions.

    Format: {"prompt": "...", "chosen": "...", "rejected": "..."}

    Pairs are generated from tasks where the human edited or rejected
    the AI prediction, creating a natural preference signal:
    - chosen = human-corrected annotation
    - rejected = original AI annotation

    Args:
        tasks: List of annotation tasks.
        reviews: Dict mapping task_id -> HumanReview.
        output_path: Custom output path.

    Returns:
        Path to the exported file.
    """
    _ensure_export_dir()
    if output_path is None:
        output_path = str(EXPORT_DIR / _timestamp_filename("dpo_pairs", "jsonl"))

    pairs: list[DPOPair] = []

    for task in tasks:
        review = reviews.get(task.id)
        if not review:
            continue

        # Generate pairs from edits (human preferred different label)
        if review.action == "edit":
            prompt = task.record.text
            ai_annotation = _format_ai_annotation(task)
            human_annotation = _format_human_annotation(task, review)

            if ai_annotation and human_annotation and ai_annotation != human_annotation:
                pair = DPOPair(
                    prompt=prompt,
                    chosen=human_annotation,
                    rejected=ai_annotation,
                    metadata={
                        "record_id": task.record.id,
                        "task_type": task.task_type,
                        "ai_confidence": task.overall_confidence,
                        "annotator_id": review.annotator_id,
                    },
                )
                pairs.append(pair)

        # Rejected AI annotations paired with a corrected version
        elif review.action == "reject":
            prompt = task.record.text
            ai_annotation = _format_ai_annotation(task)

            if ai_annotation:
                pair = DPOPair(
                    prompt=prompt,
                    chosen=f"[FLAGGED FOR RE-ANNOTATION] {review.comments or 'No reason given'}",
                    rejected=ai_annotation,
                    metadata={
                        "record_id": task.record.id,
                        "task_type": task.task_type,
                        "ai_confidence": task.overall_confidence,
                        "annotator_id": review.annotator_id,
                        "rejection_reason": review.comments,
                    },
                )
                pairs.append(pair)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")

    return output_path


def _format_ai_annotation(task: AnnotationTask) -> Optional[str]:
    """Format AI annotation as a string for DPO pairs."""
    if task.task_type == TaskType.CLASSIFICATION and task.classification:
        return json.dumps({
            "sentiment": task.classification.sentiment,
            "intent": task.classification.intent,
            "reasoning": task.classification.reasoning,
        })
    elif task.task_type == TaskType.NER and task.ner:
        return json.dumps({
            "entities": [
                {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
                for e in task.ner.entities
            ],
        })
    return None


def _format_human_annotation(
    task: AnnotationTask,
    review: HumanReview,
) -> Optional[str]:
    """Format human-corrected annotation as a string for DPO pairs."""
    if task.task_type == TaskType.CLASSIFICATION:
        sentiment = review.edited_sentiment or (
            task.classification.sentiment if task.classification else "neutral"
        )
        intent = review.edited_intent or (
            task.classification.intent if task.classification else "other"
        )
        return json.dumps({
            "sentiment": sentiment,
            "intent": intent,
            "reasoning": review.comments or "Human-corrected annotation",
        })
    elif task.task_type == TaskType.NER and review.edited_entities:
        return json.dumps({
            "entities": [
                {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
                for e in review.edited_entities
            ],
        })
    return None


# ──────────────────────────────────────────────
# Export Summary
# ──────────────────────────────────────────────

def get_export_summary(filepath: str) -> dict:
    """Get a summary of an exported file."""
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}

    line_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                line_count += 1

    return {
        "filepath": str(path.absolute()),
        "filename": path.name,
        "format": path.suffix.lstrip("."),
        "records": line_count,
        "size_bytes": path.stat().st_size,
        "size_kb": round(path.stat().st_size / 1024, 1),
    }
