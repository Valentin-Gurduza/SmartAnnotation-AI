"""
SmartAnnotate-AI — Streamlit Application
==========================================
Production-grade Human-in-the-Loop annotation interface with:
  - Dashboard with real-time progress metrics
  - Interactive annotation workspace
  - Quality & agreement analytics
  - Data management & multi-format export
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from schemas import (
    AnnotationTask,
    BatchConfig,
    ClassificationResult,
    HumanReview,
    IntentLabel,
    NEREntity,
    NERLabel,
    NERResult,
    RawTextRecord,
    ReviewAction,
    RoutingDecision,
    SentimentLabel,
    TaskType,
)
from pipeline import process_batch, AVAILABLE_MODELS
from metrics import (
    cohens_kappa,
    fleiss_kappa,
    human_ai_alignment,
    triage_statistics,
    review_statistics,
    generate_quality_report,
)
from exporters import (
    export_huggingface_jsonl,
    export_conll_bio,
    export_dpo_pairs,
    get_export_summary,
)


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="SmartAnnotate-AI",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .metric-sublabel {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* Confidence gauge */
    .confidence-high {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.1));
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        color: #4ade80;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .confidence-medium {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.1));
        border: 1px solid rgba(251, 191, 36, 0.4);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        color: #fbbf24;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .confidence-low {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        color: #f87171;
        font-weight: 700;
        font-size: 1.1rem;
    }

    /* Entity highlight styles */
    .entity-tag {
        display: inline;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0 2px;
    }
    .entity-PERSON { background: rgba(96, 165, 250, 0.25); border: 1px solid #60a5fa; color: #93c5fd; }
    .entity-ORG { background: rgba(52, 211, 153, 0.25); border: 1px solid #34d399; color: #6ee7b7; }
    .entity-LOC { background: rgba(251, 191, 36, 0.25); border: 1px solid #fbbf24; color: #fde68a; }
    .entity-DATE { background: rgba(244, 114, 182, 0.25); border: 1px solid #f472b6; color: #f9a8d4; }
    .entity-PRODUCT { background: rgba(168, 85, 247, 0.25); border: 1px solid #a855f7; color: #c4b5fd; }
    .entity-MONETARY { background: rgba(34, 197, 94, 0.25); border: 1px solid #22c55e; color: #86efac; }
    .entity-EVENT { background: rgba(249, 115, 22, 0.25); border: 1px solid #f97316; color: #fdba74; }
    .entity-MISC { background: rgba(148, 163, 184, 0.25); border: 1px solid #94a3b8; color: #cbd5e1; }

    /* Routing badge */
    .routing-auto {
        display: inline-block;
        background: linear-gradient(135deg, #065f46, #047857);
        color: #6ee7b7;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .routing-human {
        display: inline-block;
        background: linear-gradient(135deg, #7c2d12, #9a3412);
        color: #fdba74;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }

    /* Annotation text display */
    .text-display {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #e2e8f0;
    }

    /* Progress steps */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "tasks": [],
        "reviews": {},
        "current_index": 0,
        "annotator_name": "Annotator-1",
        "batch_processed": False,
        "review_start_time": None,
        "uploaded_records": [],
        "task_type": TaskType.CLASSIFICATION,
        "model": "meta-llama/llama-3.3-70b-instruct",
        "confidence_threshold": 0.85,
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def render_metric_card(value: str, label: str, sublabel: str = "") -> str:
    """Generate HTML for a styled metric card."""
    sub = f'<div class="metric-sublabel">{sublabel}</div>' if sublabel else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub}
    </div>
    """


def render_confidence_gauge(confidence: float) -> str:
    """Render a confidence gauge with color coding."""
    if confidence >= 0.85:
        cls = "confidence-high"
        icon = "🟢"
        label = "High Confidence"
    elif confidence >= 0.65:
        cls = "confidence-medium"
        icon = "🟡"
        label = "Review Recommended"
    else:
        cls = "confidence-low"
        icon = "🔴"
        label = "Low Confidence"

    return f"""
    <div class="{cls}">
        {icon} {label}: <strong>{confidence:.1%}</strong>
    </div>
    """


def render_routing_badge(routing: str) -> str:
    """Render a routing decision badge."""
    if routing == RoutingDecision.AUTO_APPROVED or routing == "auto_approved":
        return '<span class="routing-auto">✓ AUTO-APPROVED</span>'
    return '<span class="routing-human">⚠ HUMAN REVIEW</span>'


def render_ner_text(text: str, entities: list[NEREntity]) -> str:
    """Render text with color-coded NER entity highlights."""
    if not entities:
        return f'<div class="text-display">{text}</div>'

    # Sort entities by start position (reverse for safe replacement)
    sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

    # Build highlighted text
    result = text
    for entity in sorted_entities:
        label = entity.label if isinstance(entity.label, str) else entity.label.value
        entity_html = (
            f'<span class="entity-tag entity-{label}">'
            f'{entity.text}<sub style="font-size:0.7em;opacity:0.7;margin-left:3px">{label}</sub>'
            f'</span>'
        )
        result = result[:entity.start] + entity_html + result[entity.end:]

    return f'<div class="text-display">{result}</div>'


def load_jsonl(filepath: str | Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_pending_tasks() -> list[AnnotationTask]:
    """Get tasks that haven't been reviewed yet."""
    reviewed_ids = set(st.session_state.reviews.keys())
    return [t for t in st.session_state.tasks if t.id not in reviewed_ids]


def get_human_review_tasks() -> list[AnnotationTask]:
    """Get tasks routed to human review that haven't been reviewed."""
    reviewed_ids = set(st.session_state.reviews.keys())
    return [
        t for t in st.session_state.tasks
        if (t.routing == RoutingDecision.HUMAN_REVIEW or t.routing == "human_review")
        and t.id not in reviewed_ids
    ]


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    """Render the application sidebar."""
    with st.sidebar:
        st.markdown("## 🏷️ SmartAnnotate-AI")
        st.markdown("---")

        # Annotator settings
        st.markdown("### 👤 Annotator")
        st.session_state.annotator_name = st.text_input(
            "Your Name",
            value=st.session_state.annotator_name,
            key="sidebar_annotator",
        )

        st.markdown("---")

        # Model settings
        st.markdown("### 🤖 AI Model")
        st.session_state.model = st.selectbox(
            "Pre-labeling Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.model)
            if st.session_state.model in AVAILABLE_MODELS
            else 0,
            key="sidebar_model",
        )

        st.markdown("---")

        # Task settings
        st.markdown("### ⚙️ Configuration")
        task_type_str = st.radio(
            "Annotation Task",
            ["Classification", "NER"],
            index=0 if st.session_state.task_type == TaskType.CLASSIFICATION else 1,
            key="sidebar_task_type",
        )
        st.session_state.task_type = (
            TaskType.CLASSIFICATION if task_type_str == "Classification" else TaskType.NER
        )

        st.session_state.confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
            key="sidebar_threshold",
            help="Tasks below this threshold are routed to human review",
        )

        st.markdown("---")

        # Session stats
        st.markdown("### 📊 Session Stats")
        total = len(st.session_state.tasks)
        reviewed = len(st.session_state.reviews)

        if total > 0:
            st.progress(reviewed / total, text=f"{reviewed}/{total} reviewed")
        else:
            st.progress(0.0, text="No tasks loaded")

        auto = sum(
            1 for t in st.session_state.tasks
            if t.routing == RoutingDecision.AUTO_APPROVED or t.routing == "auto_approved"
        )
        st.caption(f"🟢 Auto-approved: {auto}")
        st.caption(f"🟡 Human review: {total - auto}")
        st.caption(f"✅ Reviewed: {reviewed}")


# ──────────────────────────────────────────────
# Tab 1: Dashboard
# ──────────────────────────────────────────────

def render_dashboard():
    """Render the metrics dashboard tab."""
    st.markdown('<div class="section-header">📊 Pipeline Dashboard</div>', unsafe_allow_html=True)

    tasks = st.session_state.tasks
    reviews = st.session_state.reviews

    if not tasks:
        st.info("📂 No data loaded yet. Go to **Data Management** to upload records and run the AI pipeline.")
        return

    stats = triage_statistics(tasks)

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_metric_card(
            str(stats["total"]), "Total Records", "Loaded in pipeline"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card(
            f"{stats['auto_approved_pct']}%", "Auto-Annotated",
            f"{stats['auto_approved']} records"
        ), unsafe_allow_html=True)
    with col3:
        reviewed_pct = round(len(reviews) / stats["total"] * 100, 1) if stats["total"] else 0
        st.markdown(render_metric_card(
            f"{reviewed_pct}%", "Human Reviewed",
            f"{len(reviews)} records"
        ), unsafe_allow_html=True)
    with col4:
        backlog = len(get_pending_tasks())
        st.markdown(render_metric_card(
            str(backlog), "Queue Backlog", "Pending review"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Confidence distribution
        confidences = [t.overall_confidence for t in tasks]
        fig = px.histogram(
            x=confidences,
            nbins=20,
            title="Confidence Distribution",
            labels={"x": "Confidence Score", "y": "Count"},
            color_discrete_sequence=["#818cf8"],
        )
        fig.add_vline(
            x=st.session_state.confidence_threshold,
            line_dash="dash",
            line_color="#f97316",
            annotation_text=f"Threshold ({st.session_state.confidence_threshold})",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            title_font_color="#e2e8f0",
            xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
            yaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        # Routing breakdown pie
        routing_data = pd.DataFrame({
            "Routing": ["Auto-Approved", "Human Review"],
            "Count": [stats["auto_approved"], stats["human_review"]],
        })
        fig2 = px.pie(
            routing_data,
            values="Count",
            names="Routing",
            title="Routing Breakdown",
            color="Routing",
            color_discrete_map={
                "Auto-Approved": "#22c55e",
                "Human Review": "#f97316",
            },
            hole=0.45,
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            title_font_color="#e2e8f0",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Time savings estimate
    st.markdown("<br>", unsafe_allow_html=True)
    col_ts1, col_ts2 = st.columns(2)
    with col_ts1:
        st.markdown(render_metric_card(
            f"{stats['estimated_time_saved_pct']}%", "Est. Time Saved",
            "vs. fully manual annotation"
        ), unsafe_allow_html=True)
    with col_ts2:
        st.markdown(render_metric_card(
            f"{stats['avg_confidence']:.1%}", "Avg Confidence",
            "Across all predictions"
        ), unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Tab 2: Annotation Workspace
# ──────────────────────────────────────────────

def render_annotation_workspace():
    """Render the interactive annotation workspace."""
    st.markdown('<div class="section-header">📝 Annotation Workspace</div>', unsafe_allow_html=True)

    tasks = st.session_state.tasks
    if not tasks:
        st.info("📂 No tasks available. Upload data and run the pipeline first.")
        return

    # Queue selector
    queue_option = st.radio(
        "Queue Filter",
        ["All Tasks", "Human Review Only", "Auto-Approved (Spot Check)", "Unreviewed Only"],
        horizontal=True,
        key="queue_filter",
    )

    if queue_option == "Human Review Only":
        filtered = [t for t in tasks if t.routing == RoutingDecision.HUMAN_REVIEW or t.routing == "human_review"]
    elif queue_option == "Auto-Approved (Spot Check)":
        filtered = [t for t in tasks if t.routing == RoutingDecision.AUTO_APPROVED or t.routing == "auto_approved"]
    elif queue_option == "Unreviewed Only":
        filtered = get_pending_tasks()
    else:
        filtered = tasks

    if not filtered:
        st.success("✅ All tasks in this queue have been reviewed!")
        return

    # Navigation
    total = len(filtered)
    idx = st.session_state.current_index % total if total > 0 else 0

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 3])
    with nav_col1:
        if st.button("⬅ Previous", key="btn_prev", use_container_width=True):
            st.session_state.current_index = max(0, idx - 1)
            st.session_state.review_start_time = time.time()
            st.rerun()
    with nav_col2:
        if st.button("Next ➡", key="btn_next", use_container_width=True):
            st.session_state.current_index = min(total - 1, idx + 1)
            st.session_state.review_start_time = time.time()
            st.rerun()
    with nav_col3:
        # Jump to specific record
        jump = st.number_input(
            "Go to #", min_value=1, max_value=total, value=idx + 1, key="jump_to"
        )
        if jump - 1 != idx:
            st.session_state.current_index = jump - 1
            st.session_state.review_start_time = time.time()
            st.rerun()

    # Track review start time
    if st.session_state.review_start_time is None:
        st.session_state.review_start_time = time.time()

    task = filtered[idx]
    is_reviewed = task.id in st.session_state.reviews

    # Header with record info
    st.markdown(f"**Record {idx + 1} of {total}** — ID: `{task.record.id}`", unsafe_allow_html=True)
    header_col1, header_col2, header_col3 = st.columns([2, 2, 2])
    with header_col1:
        st.markdown(render_confidence_gauge(task.overall_confidence), unsafe_allow_html=True)
    with header_col2:
        st.markdown(render_routing_badge(task.routing), unsafe_allow_html=True)
    with header_col3:
        if is_reviewed:
            review = st.session_state.reviews[task.id]
            st.success(f"✅ Reviewed: {review.action}")
        else:
            st.warning("⏳ Pending review")

    st.markdown("<br>", unsafe_allow_html=True)

    # Text display
    main_col, side_col = st.columns([3, 1])

    with main_col:
        st.markdown("**Original Text:**")
        if task.task_type == TaskType.NER or task.task_type == "ner":
            entities = task.ner.entities if task.ner else []
            st.markdown(render_ner_text(task.record.text, entities), unsafe_allow_html=True)

            # Entity table
            if entities:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Extracted Entities:**")
                ent_data = []
                for e in entities:
                    ent_data.append({
                        "Entity": e.text,
                        "Label": e.label if isinstance(e.label, str) else e.label.value,
                        "Start": e.start,
                        "End": e.end,
                        "Confidence": f"{e.confidence:.1%}",
                    })
                st.dataframe(pd.DataFrame(ent_data), use_container_width=True, hide_index=True)
        else:
            st.markdown(f'<div class="text-display">{task.record.text}</div>', unsafe_allow_html=True)

    with side_col:
        st.markdown("**AI Prediction:**")
        if task.task_type == TaskType.CLASSIFICATION or task.task_type == "classification":
            if task.classification:
                st.markdown(f"🏷️ **Sentiment:** `{task.classification.sentiment}`")
                st.markdown(f"🎯 **Intent:** `{task.classification.intent}`")
                st.markdown(f"📊 **Confidence:** `{task.classification.confidence:.1%}`")
                with st.expander("💭 AI Reasoning"):
                    st.write(task.classification.reasoning)
        elif task.ner and task.ner.entities:
            st.markdown(f"📊 **Overall:** `{task.ner.confidence:.1%}`")
            st.markdown(f"🔍 **Entities found:** `{len(task.ner.entities)}`")

        # Metadata
        st.markdown("---")
        st.markdown("**Metadata:**")
        st.caption(f"Source: {task.record.source or 'N/A'}")
        st.caption(f"Language: {task.record.language or 'N/A'}")
        st.caption(f"Model: {task.model_used}")

    st.markdown("---")

    # Annotation Actions
    st.markdown("**Review Actions:**")

    if task.task_type == TaskType.CLASSIFICATION or task.task_type == "classification":
        _render_classification_actions(task)
    else:
        _render_ner_actions(task)


def _render_classification_actions(task: AnnotationTask):
    """Render classification-specific review actions."""
    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        accept = st.button(
            "✅ Accept AI Labels (A)",
            key=f"accept_{task.id}",
            use_container_width=True,
            type="primary",
        )

    with action_col2:
        edit_mode = st.button(
            "✏️ Edit Labels (E)",
            key=f"edit_{task.id}",
            use_container_width=True,
        )

    with action_col3:
        reject = st.button(
            "❌ Reject / Flag (R)",
            key=f"reject_{task.id}",
            use_container_width=True,
        )

    # Edit form (always visible, used when Edit is clicked)
    with st.expander("✏️ Edit Labels", expanded=edit_mode):
        edit_col1, edit_col2 = st.columns(2)
        with edit_col1:
            sentiment_options = [e.value for e in SentimentLabel]
            current_sent = task.classification.sentiment if task.classification else "neutral"
            if isinstance(current_sent, SentimentLabel):
                current_sent = current_sent.value
            edited_sentiment = st.selectbox(
                "Sentiment",
                sentiment_options,
                index=sentiment_options.index(current_sent) if current_sent in sentiment_options else 0,
                key=f"edit_sent_{task.id}",
            )
        with edit_col2:
            intent_options = [e.value for e in IntentLabel]
            current_intent = task.classification.intent if task.classification else "other"
            if isinstance(current_intent, IntentLabel):
                current_intent = current_intent.value
            edited_intent = st.selectbox(
                "Intent",
                intent_options,
                index=intent_options.index(current_intent) if current_intent in intent_options else 0,
                key=f"edit_int_{task.id}",
            )
        if st.button("💾 Save Edits", key=f"save_edit_{task.id}", type="primary"):
            _submit_review(task, ReviewAction.EDIT, edited_sentiment, edited_intent)

    # Comment field
    comments = st.text_area(
        "Annotator Comments / Critique",
        key=f"comments_{task.id}",
        placeholder="Optional: explain your decision, flag issues, or note edge cases...",
    )

    # Handle button actions
    if accept:
        _submit_review(task, ReviewAction.ACCEPT, comments_text=comments)
    elif reject:
        _submit_review(task, ReviewAction.REJECT, comments_text=comments)


def _render_ner_actions(task: AnnotationTask):
    """Render NER-specific review actions."""
    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        accept = st.button(
            "✅ Accept Entities (A)",
            key=f"accept_{task.id}",
            use_container_width=True,
            type="primary",
        )

    with action_col2:
        edit_mode = st.button(
            "✏️ Edit Entities (E)",
            key=f"edit_{task.id}",
            use_container_width=True,
        )

    with action_col3:
        reject = st.button(
            "❌ Reject / Flag (R)",
            key=f"reject_{task.id}",
            use_container_width=True,
        )

    # NER editing interface
    with st.expander("✏️ Edit Entities", expanded=edit_mode):
        st.markdown("Modify entities below. Add or remove rows as needed.")

        existing = []
        if task.ner and task.ner.entities:
            for e in task.ner.entities:
                existing.append({
                    "text": e.text,
                    "label": e.label if isinstance(e.label, str) else e.label.value,
                    "start": e.start,
                    "end": e.end,
                })

        if existing:
            df = pd.DataFrame(existing)
        else:
            df = pd.DataFrame(columns=["text", "label", "start", "end"])

        edited_df = st.data_editor(
            df,
            column_config={
                "label": st.column_config.SelectboxColumn(
                    "Label",
                    options=[e.value for e in NERLabel],
                    required=True,
                ),
                "start": st.column_config.NumberColumn("Start", min_value=0),
                "end": st.column_config.NumberColumn("End", min_value=0),
            },
            num_rows="dynamic",
            use_container_width=True,
            key=f"ner_editor_{task.id}",
        )

        if st.button("💾 Save Entity Edits", key=f"save_ner_{task.id}", type="primary"):
            edited_entities = []
            for _, row in edited_df.iterrows():
                try:
                    entity = NEREntity(
                        text=str(row["text"]),
                        label=row["label"],
                        start=int(row["start"]),
                        end=int(row["end"]),
                        confidence=1.0,  # Human-edited = full confidence
                    )
                    edited_entities.append(entity)
                except Exception:
                    continue
            _submit_review(task, ReviewAction.EDIT, edited_entities=edited_entities)

    comments = st.text_area(
        "Annotator Comments / Critique",
        key=f"comments_{task.id}",
        placeholder="Optional: explain your decision...",
    )

    if accept:
        _submit_review(task, ReviewAction.ACCEPT, comments_text=comments)
    elif reject:
        _submit_review(task, ReviewAction.REJECT, comments_text=comments)


def _submit_review(
    task: AnnotationTask,
    action: ReviewAction,
    edited_sentiment: Optional[str] = None,
    edited_intent: Optional[str] = None,
    edited_entities: Optional[list[NEREntity]] = None,
    comments_text: str = "",
):
    """Submit a human review for a task."""
    elapsed = time.time() - (st.session_state.review_start_time or time.time())

    review = HumanReview(
        task_id=task.id,
        annotator_id=st.session_state.annotator_name,
        action=action,
        edited_sentiment=edited_sentiment,
        edited_intent=edited_intent,
        edited_entities=edited_entities,
        comments=comments_text,
        time_spent_seconds=round(elapsed, 1),
    )

    st.session_state.reviews[task.id] = review
    st.session_state.review_start_time = time.time()

    action_name = action.value if isinstance(action, ReviewAction) else action
    st.toast(f"✅ Review submitted: {action_name}", icon="✅")

    # Auto-advance to next unreviewed task
    tasks = st.session_state.tasks
    current = st.session_state.current_index
    for i in range(current + 1, len(tasks)):
        if tasks[i].id not in st.session_state.reviews:
            st.session_state.current_index = i
            break

    st.rerun()


# ──────────────────────────────────────────────
# Tab 3: Quality & Agreement
# ──────────────────────────────────────────────

def render_quality_tab():
    """Render quality metrics and inter-annotator agreement."""
    st.markdown('<div class="section-header">📈 Quality & Agreement Metrics</div>', unsafe_allow_html=True)

    tasks = st.session_state.tasks
    reviews = st.session_state.reviews

    if not tasks:
        st.info("📂 No data available. Process some records first.")
        return

    if not reviews:
        st.info("📝 No reviews yet. Complete some annotations to see quality metrics.")
        return

    # Generate quality report
    report = generate_quality_report(tasks, list(reviews.values()))

    # Review statistics
    rev_stats = report["reviews"]
    st.markdown("### 👥 Review Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_metric_card(
            str(rev_stats["total_reviews"]), "Total Reviews"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card(
            f"{rev_stats['accepted_pct']}%", "Accepted",
            f"{rev_stats['accepted']} records"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card(
            f"{rev_stats['edited_pct']}%", "Edited",
            f"{rev_stats['edited']} records"
        ), unsafe_allow_html=True)
    with col4:
        st.markdown(render_metric_card(
            f"{rev_stats['rejected_pct']}%", "Rejected",
            f"{rev_stats['rejected']} records"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Human-vs-AI Alignment
    st.markdown("### 🤖↔👤 Human-vs-AI Alignment")

    # Build label pairs for reviewed tasks
    ai_labels = []
    human_labels = []

    for task in tasks:
        if task.id not in reviews:
            continue
        review = reviews[task.id]

        if task.task_type == TaskType.CLASSIFICATION or task.task_type == "classification":
            if task.classification:
                ai_label = (
                    task.classification.sentiment.value
                    if isinstance(task.classification.sentiment, SentimentLabel)
                    else str(task.classification.sentiment)
                )
                ai_labels.append(ai_label)

                if review.action == "edit" and review.edited_sentiment:
                    human_label = (
                        review.edited_sentiment.value
                        if isinstance(review.edited_sentiment, SentimentLabel)
                        else str(review.edited_sentiment)
                    )
                    human_labels.append(human_label)
                elif review.action == "accept":
                    human_labels.append(ai_label)
                else:
                    human_labels.append("rejected")

    if ai_labels and human_labels:
        alignment = human_ai_alignment(ai_labels, human_labels)

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown(render_metric_card(
                f"{alignment['accuracy']:.1%}", "AI Accuracy",
                "vs. human decisions"
            ), unsafe_allow_html=True)
        with col_a2:
            st.markdown(render_metric_card(
                f"{alignment['f1_macro']:.1%}", "Macro F1",
                "Across all labels"
            ), unsafe_allow_html=True)

        # Per-label breakdown
        if alignment.get("per_label"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Per-Label Performance:**")
            label_rows = []
            for label, metrics in alignment["per_label"].items():
                label_rows.append({
                    "Label": label,
                    "Precision": f"{metrics['precision']:.1%}",
                    "Recall": f"{metrics['recall']:.1%}",
                    "F1": f"{metrics['f1']:.1%}",
                    "Support": metrics["support"],
                })
            st.dataframe(pd.DataFrame(label_rows), use_container_width=True, hide_index=True)

        # Cohen's Kappa (AI vs Human as "two annotators")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📐 Inter-Annotator Agreement")
        kappa = cohens_kappa(ai_labels, human_labels)
        kappa_interpretation = (
            "Almost Perfect" if kappa > 0.8 else
            "Substantial" if kappa > 0.6 else
            "Moderate" if kappa > 0.4 else
            "Fair" if kappa > 0.2 else
            "Slight" if kappa > 0 else
            "Poor"
        )

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown(render_metric_card(
                f"{kappa:.3f}", "Cohen's Kappa (AI vs Human)",
                kappa_interpretation
            ), unsafe_allow_html=True)
        with col_k2:
            st.info(
                "📖 **Interpretation Guide:**\n"
                "- **> 0.8**: Almost perfect agreement\n"
                "- **0.6–0.8**: Substantial agreement\n"
                "- **0.4–0.6**: Moderate agreement\n"
                "- **0.2–0.4**: Fair agreement\n"
                "- **< 0.2**: Slight/Poor agreement"
            )
    else:
        st.warning("Not enough reviewed classification tasks to compute alignment metrics.")

    # Fleiss' Kappa demo
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📐 Multi-Annotator Agreement (Fleiss' Kappa)")
    st.info(
        "Fleiss' Kappa requires multiple annotators reviewing the same items. "
        "Upload a multi-annotator dataset or invite additional reviewers to enable this metric."
    )

    # Demo calculation with synthetic data
    with st.expander("🧪 Demo: Fleiss' Kappa with Synthetic Data"):
        st.markdown("Sample 5-item × 3-category ratings matrix (3 annotators):")
        demo_matrix = [
            [3, 0, 0],
            [2, 1, 0],
            [0, 1, 2],
            [1, 1, 1],
            [0, 0, 3],
        ]
        demo_df = pd.DataFrame(
            demo_matrix,
            columns=["Category A", "Category B", "Category C"],
            index=[f"Item {i+1}" for i in range(5)],
        )
        st.dataframe(demo_df, use_container_width=True)
        fk = fleiss_kappa(demo_matrix)
        st.markdown(f"**Fleiss' Kappa:** `{fk:.4f}`")


# ──────────────────────────────────────────────
# Tab 4: Data Management
# ──────────────────────────────────────────────

def render_data_management():
    """Render data ingestion and export tab."""
    st.markdown('<div class="section-header">📤 Data Management</div>', unsafe_allow_html=True)

    # ── Upload Section ──
    st.markdown("### 📂 Data Ingestion")

    upload_col1, upload_col2 = st.columns([2, 1])

    with upload_col1:
        uploaded_file = st.file_uploader(
            "Upload raw text data (CSV or JSONL)",
            type=["csv", "jsonl", "json"],
            key="file_uploader",
        )

    with upload_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button(
            "📋 Load Sample Data",
            key="load_sample",
            use_container_width=True,
        )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                text_col = st.selectbox(
                    "Select text column",
                    df.columns.tolist(),
                    key="text_column",
                )
                records = []
                for _, row in df.iterrows():
                    records.append(RawTextRecord(
                        text=str(row[text_col]),
                        metadata=row.to_dict(),
                    ))
                st.session_state.uploaded_records = records
            else:
                content = uploaded_file.read().decode("utf-8")
                raw_records = []
                for line in content.strip().split("\n"):
                    if line.strip():
                        raw_records.append(json.loads(line))

                records = [RawTextRecord.model_validate(r) for r in raw_records]
                st.session_state.uploaded_records = records

            st.success(f"✅ Loaded {len(records)} records from `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    if use_sample:
        sample_path = Path("data/sample_raw.jsonl")
        if sample_path.exists():
            raw = load_jsonl(sample_path)
            records = [RawTextRecord.model_validate(r) for r in raw]
            st.session_state.uploaded_records = records
            st.success(f"✅ Loaded {len(records)} sample records")
            st.rerun()
        else:
            st.error("❌ Sample data file not found at `data/sample_raw.jsonl`")

    # Show loaded records preview
    if st.session_state.uploaded_records:
        st.markdown(f"**{len(st.session_state.uploaded_records)} records loaded**")
        with st.expander("Preview records"):
            preview = []
            for r in st.session_state.uploaded_records[:10]:
                preview.append({
                    "ID": r.id[:12] + "...",
                    "Text": r.text[:80] + "..." if len(r.text) > 80 else r.text,
                    "Source": r.source or "N/A",
                    "Language": r.language or "en",
                })
            st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

    # ── Run Pipeline ──
    st.markdown("---")
    st.markdown("### 🚀 Run AI Pre-Labeling Pipeline")

    if st.session_state.uploaded_records:
        run_col1, run_col2 = st.columns([1, 2])
        with run_col1:
            run_pipeline = st.button(
                "🤖 Run Pipeline",
                key="run_pipeline",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing,
            )

        if run_pipeline:
            st.session_state.processing = True
            config = BatchConfig(
                model=st.session_state.model,
                task_type=st.session_state.task_type,
                confidence_threshold=st.session_state.confidence_threshold,
            )

            progress_bar = st.progress(0, text="Starting pipeline...")
            status_text = st.empty()

            def update_progress(done: int, total: int):
                progress_bar.progress(done / total, text=f"Processing {done}/{total}...")

            try:
                result = asyncio.run(process_batch(
                    st.session_state.uploaded_records,
                    config,
                    progress_callback=update_progress,
                ))

                st.session_state.tasks = result.tasks
                st.session_state.batch_processed = True
                st.session_state.current_index = 0
                st.session_state.reviews = {}
                st.session_state.processing = False

                progress_bar.progress(1.0, text="✅ Pipeline complete!")
                st.success(
                    f"✅ Processed **{result.total_records}** records in **{result.processing_time_seconds}s**\n\n"
                    f"- Auto-approved: **{result.auto_approved}** ({result.auto_approved / result.total_records * 100:.0f}%)\n"
                    f"- Human review: **{result.human_review}** ({result.human_review / result.total_records * 100:.0f}%)\n"
                    f"- Avg confidence: **{result.avg_confidence:.1%}**"
                )
                st.rerun()

            except Exception as e:
                st.session_state.processing = False
                st.error(f"❌ Pipeline error: {e}")
    else:
        st.warning("⬆️ Upload data above or load sample data to run the pipeline.")

    # ── Export Section ──
    st.markdown("---")
    st.markdown("### 📦 Export Annotations")

    if not st.session_state.tasks:
        st.info("No annotations to export yet.")
        return

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        st.markdown("**🤗 Hugging Face JSONL**")
        st.caption("Text classification format")
        if st.button("Export HF JSONL", key="export_hf", use_container_width=True):
            try:
                path = export_huggingface_jsonl(
                    st.session_state.tasks,
                    st.session_state.reviews,
                )
                summary = get_export_summary(path)
                st.success(f"✅ Exported {summary['records']} records ({summary['size_kb']} KB)")
                with open(path, "r") as f:
                    st.download_button(
                        "⬇️ Download",
                        f.read(),
                        file_name=summary["filename"],
                        mime="application/jsonl",
                    )
            except Exception as e:
                st.error(f"Export error: {e}")

    with export_col2:
        st.markdown("**🏷️ CoNLL / BIO Format**")
        st.caption("NER token tagging format")
        if st.button("Export CoNLL", key="export_conll", use_container_width=True):
            try:
                path = export_conll_bio(
                    st.session_state.tasks,
                    st.session_state.reviews,
                )
                summary = get_export_summary(path)
                st.success(f"✅ Exported ({summary['size_kb']} KB)")
                with open(path, "r") as f:
                    st.download_button(
                        "⬇️ Download",
                        f.read(),
                        file_name=summary["filename"],
                        mime="text/plain",
                    )
            except Exception as e:
                st.error(f"Export error: {e}")

    with export_col3:
        st.markdown("**🎯 DPO / RLHF Pairs**")
        st.caption("Preference optimization format")
        if st.button("Export DPO", key="export_dpo", use_container_width=True):
            try:
                path = export_dpo_pairs(
                    st.session_state.tasks,
                    st.session_state.reviews,
                )
                summary = get_export_summary(path)
                st.success(f"✅ Exported {summary['records']} pairs ({summary['size_kb']} KB)")
                with open(path, "r") as f:
                    st.download_button(
                        "⬇️ Download",
                        f.read(),
                        file_name=summary["filename"],
                        mime="application/jsonl",
                    )
            except Exception as e:
                st.error(f"Export error: {e}")


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

def main():
    """Main application entry point."""
    render_sidebar()

    # Header
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
            <h1 style="
                font-size: 2.2rem;
                font-weight: 800;
                background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.3rem;
            ">🏷️ SmartAnnotate-AI</h1>
            <p style="color: #94a3b8; font-size: 1rem; margin-top: 0;">
                Human-in-the-Loop AI Annotation Pipeline with Automated Confidence Routing
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "📝 Annotation Workspace",
        "📈 Quality & Agreement",
        "📤 Data Management",
    ])

    with tab1:
        render_dashboard()

    with tab2:
        render_annotation_workspace()

    with tab3:
        render_quality_tab()

    with tab4:
        render_data_management()


if __name__ == "__main__":
    main()
