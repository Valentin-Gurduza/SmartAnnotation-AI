"""
SmartAnnotate-AI — Pydantic v2 Data Schemas
============================================
Strict data contracts for annotation tasks, AI predictions,
human reviews, confidence routing, and export formats.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class TaskType(str, Enum):
    """Supported annotation task types."""
    CLASSIFICATION = "classification"
    NER = "ner"


class SentimentLabel(str, Enum):
    """Multiclass sentiment / intent labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class IntentLabel(str, Enum):
    """Customer-intent labels for classification tasks."""
    COMPLAINT = "complaint"
    INQUIRY = "inquiry"
    FEEDBACK = "feedback"
    REQUEST = "request"
    ESCALATION = "escalation"
    OTHER = "other"


class NERLabel(str, Enum):
    """Named-entity labels for NER tasks."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    PRODUCT = "PRODUCT"
    MONETARY = "MONETARY"
    EVENT = "EVENT"
    MISC = "MISC"


class RoutingDecision(str, Enum):
    """Confidence-based routing outcome."""
    AUTO_APPROVED = "auto_approved"
    HUMAN_REVIEW = "human_review"


class ReviewAction(str, Enum):
    """Human reviewer action on an annotation."""
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"


# ──────────────────────────────────────────────
# Input Schemas
# ──────────────────────────────────────────────

class RawTextRecord(BaseModel):
    """A single raw text record for annotation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier",
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Raw text to be annotated",
    )
    source: Optional[str] = Field(
        default=None,
        description="Origin of the text (e.g., 'zendesk', 'twitter')",
    )
    language: Optional[str] = Field(
        default="en",
        description="ISO 639-1 language code",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary metadata key-value pairs",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation timestamp",
    )


# ──────────────────────────────────────────────
# AI Prediction Schemas
# ──────────────────────────────────────────────

class ClassificationResult(BaseModel):
    """LLM classification prediction with reasoning."""
    model_config = ConfigDict(str_strip_whitespace=True)

    sentiment: SentimentLabel = Field(
        ..., description="Predicted sentiment label"
    )
    intent: IntentLabel = Field(
        ..., description="Predicted intent category"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0.0–1.0)",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Model's reasoning for the prediction",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Clamp confidence to valid range."""
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        return v


class NEREntity(BaseModel):
    """A single named entity with character offsets."""
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(..., min_length=1, description="Entity surface text")
    label: NERLabel = Field(..., description="Entity type label")
    start: int = Field(..., ge=0, description="Start character offset")
    end: int = Field(..., ge=0, description="End character offset")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Entity-level confidence",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        return v

    @model_validator(mode="after")
    def validate_offsets(self) -> "NEREntity":
        """Ensure start < end."""
        if self.start >= self.end:
            raise ValueError(
                f"start ({self.start}) must be less than end ({self.end})"
            )
        return self


class NERResult(BaseModel):
    """LLM NER prediction for a text record."""
    entities: list[NEREntity] = Field(
        default_factory=list,
        description="List of extracted entities",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall NER confidence score",
    )
    reasoning: str = Field(
        default="",
        description="Model's reasoning for entity extraction",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        return v


# ──────────────────────────────────────────────
# Annotation Task & Routing
# ──────────────────────────────────────────────

class AnnotationTask(BaseModel):
    """
    A complete annotation task: raw input + AI prediction + routing.
    This is the primary unit of work flowing through the pipeline.
    """
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique task identifier",
    )
    record: RawTextRecord = Field(
        ..., description="The raw text record being annotated"
    )
    task_type: TaskType = Field(
        ..., description="Type of annotation task"
    )
    model_used: str = Field(
        default="", description="Model identifier used for pre-labeling"
    )

    # AI predictions (populated after pre-labeling)
    classification: Optional[ClassificationResult] = Field(
        default=None,
        description="Classification result (if task_type == classification)",
    )
    ner: Optional[NERResult] = Field(
        default=None,
        description="NER result (if task_type == ner)",
    )

    # Routing
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregated confidence score for routing",
    )
    routing: RoutingDecision = Field(
        default=RoutingDecision.HUMAN_REVIEW,
        description="Routing decision based on confidence",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    prelabeled_at: Optional[datetime] = Field(default=None)


# ──────────────────────────────────────────────
# Human Review
# ──────────────────────────────────────────────

class HumanReview(BaseModel):
    """A human reviewer's annotation decision."""
    model_config = ConfigDict(str_strip_whitespace=True, use_enum_values=True)

    task_id: str = Field(..., description="ID of the annotation task reviewed")
    annotator_id: str = Field(
        ..., min_length=1, description="Identifier of the human annotator"
    )
    action: ReviewAction = Field(
        ..., description="Review action taken"
    )

    # Edited labels (populated when action == EDIT)
    edited_sentiment: Optional[SentimentLabel] = Field(default=None)
    edited_intent: Optional[IntentLabel] = Field(default=None)
    edited_entities: Optional[list[NEREntity]] = Field(default=None)

    comments: str = Field(
        default="",
        description="Annotator notes or critique",
    )
    reviewed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    time_spent_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time spent reviewing in seconds",
    )


# ──────────────────────────────────────────────
# Final Annotation Record (Merged)
# ──────────────────────────────────────────────

class AnnotationRecord(BaseModel):
    """
    Final annotation record with full provenance.
    Merges AI prediction, human review, and routing metadata.
    """
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    record_id: str = Field(..., description="Original raw record ID")
    text: str = Field(..., description="Original text")

    # Final labels
    task_type: TaskType
    final_sentiment: Optional[SentimentLabel] = None
    final_intent: Optional[IntentLabel] = None
    final_entities: Optional[list[NEREntity]] = None

    # Provenance
    ai_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    routing: RoutingDecision = RoutingDecision.HUMAN_REVIEW
    review_action: Optional[ReviewAction] = None
    annotator_id: Optional[str] = None
    annotator_comments: str = ""

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    finalized_at: Optional[datetime] = None


# ──────────────────────────────────────────────
# Export Format Schemas
# ──────────────────────────────────────────────

class HuggingFaceRecord(BaseModel):
    """Hugging Face Datasets JSONL format for text classification."""
    text: str
    label: str
    label_id: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


class CoNLLToken(BaseModel):
    """A single token in CoNLL / BIO format."""
    token: str
    bio_tag: str  # e.g., "B-PERSON", "I-ORG", "O"


class CoNLLRecord(BaseModel):
    """CoNLL-formatted sentence for NER."""
    tokens: list[CoNLLToken]
    raw_text: str


class DPOPair(BaseModel):
    """
    Direct Preference Optimization pair for RLHF fine-tuning.
    Generated from accept/reject decisions.
    """
    prompt: str = Field(..., description="Input text / instruction")
    chosen: str = Field(
        ..., description="Preferred annotation (human-approved)"
    )
    rejected: str = Field(
        ..., description="Rejected annotation (AI prediction or flagged)"
    )
    metadata: dict = Field(default_factory=dict)


# ──────────────────────────────────────────────
# Batch Processing Schemas
# ──────────────────────────────────────────────

class BatchConfig(BaseModel):
    """Configuration for a batch pre-labeling run."""
    model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct",
        description="OpenRouter model identifier",
    )
    task_type: TaskType = Field(
        default=TaskType.CLASSIFICATION,
        description="Annotation task type",
    )
    confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for auto-approval",
    )
    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum concurrent API requests",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature",
    )


class BatchResult(BaseModel):
    """Summary of a batch pre-labeling run."""
    total_records: int = 0
    successful: int = 0
    failed: int = 0
    auto_approved: int = 0
    human_review: int = 0
    avg_confidence: float = 0.0
    processing_time_seconds: float = 0.0
    tasks: list[AnnotationTask] = Field(default_factory=list)
