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
from pipeline import process_batch, AVAILABLE_MODELS, fetch_available_models, search_models
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

    /* Progress steps & Stepper */
    .stepper-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 1rem 0 1rem 0;
    }
    .step-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: all 0.2s ease;
    }
    .step-card.active {
        background: rgba(49, 46, 129, 0.45);
        border: 1px solid #818cf8;
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.2);
    }
    .step-card.done {
        background: rgba(6, 78, 59, 0.35);
        border: 1px solid #10b981;
    }
    .step-card.pending {
        opacity: 0.6;
    }
    .step-icon-badge {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    .step-icon-badge.done {
        background: #10b981;
        color: #ffffff;
    }
    .step-icon-badge.active {
        background: #6366f1;
        color: #ffffff;
        box-shadow: 0 0 8px #6366f1;
    }
    .step-icon-badge.pending {
        background: rgba(148, 163, 184, 0.2);
        color: #94a3b8;
    }
    .step-text {
        overflow: hidden;
    }
    .step-name {
        font-size: 0.85rem;
        font-weight: 700;
        color: #f1f5f9;
        white-space: nowrap;
        text-overflow: ellipsis;
    }
    .step-desc {
        font-size: 0.74rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Action guidance alert box */
    .action-guidance-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(49, 46, 129, 0.45));
        border-left: 4px solid #818cf8;
        border-top: 1px solid rgba(129, 140, 248, 0.25);
        border-right: 1px solid rgba(129, 140, 248, 0.25);
        border-bottom: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 0 12px 12px 0;
        padding: 0.9rem 1.4rem;
        margin: 0.3rem 0 1.2rem 0;
    }
    .action-guidance-title {
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 4px;
    }
    .action-guidance-text {
        font-size: 0.95rem;
        color: #e2e8f0;
        line-height: 1.45;
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


def load_sample_records() -> list[RawTextRecord]:
    """Load sample raw records from data/sample_raw.jsonl."""
    sample_path = Path("data/sample_raw.jsonl")
    if sample_path.exists():
        raw = load_jsonl(sample_path)
        return [RawTextRecord.model_validate(r) for r in raw]
    return []


def execute_pipeline(records: list[RawTextRecord]):
    """Execute AI pre-labeling pipeline on the given records with progress reporting."""
    st.session_state.processing = True
    config = BatchConfig(
        model=st.session_state.model,
        task_type=st.session_state.task_type,
        confidence_threshold=st.session_state.confidence_threshold,
    )

    progress_bar = st.progress(0, text="Pornire pipeline de adnotare AI...")

    def update_progress(done: int, total: int):
        progress_bar.progress(done / total, text=f"AI Pre-labeling: {done}/{total} înregistrări...")

    try:
        result = asyncio.run(process_batch(
            records,
            config,
            progress_callback=update_progress,
        ))

        st.session_state.tasks = result.tasks
        st.session_state.batch_processed = True
        st.session_state.current_index = 0
        st.session_state.reviews = {}
        st.session_state.processing = False

        progress_bar.progress(1.0, text="✅ Pipeline finalizat!")
        st.toast("✅ Pre-etichetarea AI a fost finalizată cu succes!", icon="🚀")
        st.rerun()

    except Exception as e:
        st.session_state.processing = False
        st.error(f"❌ Eroare la rularea pipeline-ului: {e}")


def render_pipeline_stepper():
    """Render a visual 4-step progress tracker and next-action guidance banner."""
    tasks = st.session_state.tasks
    uploaded = st.session_state.uploaded_records
    reviews = st.session_state.reviews
    pending = get_pending_tasks()
    total_tasks = len(tasks)
    reviewed_count = len(reviews)
    pending_count = len(pending)

    # Step status calculations
    has_data = total_tasks > 0 or len(uploaded) > 0
    s1_status = "done" if has_data else "active"
    s1_badge = "✓" if has_data else "1"
    s1_desc = f"{total_tasks or len(uploaded)} mostre gata" if has_data else "În așteptare date"

    s2_done = total_tasks > 0
    s2_status = "done" if s2_done else ("active" if len(uploaded) > 0 else "pending")
    s2_badge = "✓" if s2_done else "2"
    auto_count = sum(1 for t in tasks if t.routing == RoutingDecision.AUTO_APPROVED or t.routing == "auto_approved")
    s2_desc = f"{auto_count} auto-aprobate" if s2_done else ("Gata de rulare" if len(uploaded) > 0 else "În așteptare")

    if not s2_done:
        s3_status = "pending"
        s3_badge = "3"
        s3_desc = "În așteptare AI"
    elif pending_count == 0:
        s3_status = "done"
        s3_badge = "✓"
        s3_desc = f"Toate revizuite ({reviewed_count})"
    else:
        s3_status = "active"
        s3_badge = "3"
        s3_desc = f"{pending_count} în coadă"

    s4_status = "done" if (s2_done and pending_count == 0) else ("active" if s2_done else "pending")
    s4_badge = "✓" if (s2_done and pending_count == 0) else "4"
    s4_desc = "3 formate gata" if s2_done else "În așteptare"

    st.markdown(
        f"""
        <div class="stepper-container">
            <div class="step-card {s1_status}">
                <div class="step-icon-badge {s1_status}">{s1_badge}</div>
                <div class="step-text">
                    <div class="step-name">1. Ingestie Date</div>
                    <div class="step-desc">{s1_desc}</div>
                </div>
            </div>
            <div class="step-card {s2_status}">
                <div class="step-icon-badge {s2_status}">{s2_badge}</div>
                <div class="step-text">
                    <div class="step-name">2. Pre-Etichetare AI</div>
                    <div class="step-desc">{s2_desc}</div>
                </div>
            </div>
            <div class="step-card {s3_status}">
                <div class="step-icon-badge {s3_status}">{s3_badge}</div>
                <div class="step-text">
                    <div class="step-name">3. Verificare Umană</div>
                    <div class="step-desc">{s3_desc}</div>
                </div>
            </div>
            <div class="step-card {s4_status}">
                <div class="step-icon-badge {s4_status}">{s4_badge}</div>
                <div class="step-text">
                    <div class="step-name">4. Calitate & Export</div>
                    <div class="step-desc">{s4_desc}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Next recommended action guidance
    if total_tasks == 0 and len(uploaded) == 0:
        action_title = "👉 Pasul 1: Încarcă date sau încearcă Demo-ul rapid (1-Click)"
        action_text = (
            "Nu există date încărcate în pipeline. Apasă pe butonul "
            "<strong>⚡ 1-Click Demo</strong> din tab-ul de mai jos pentru a încărca 15 mostre și a rula pipeline-ul AI instantaneu, "
            "sau încarcă propriul fișier CSV / JSONL."
        )
    elif total_tasks == 0 and len(uploaded) > 0:
        action_title = "👉 Pasul 2: Rulează Pre-Etichetarea AI"
        action_text = (
            f"Au fost încărcate <strong>{len(uploaded)} înregistrări</strong>. "
            "Apasă pe <strong>🤖 Rulează Pipeline AI</strong> pentru a genera predicțiile și a calcula scorurile de încredere."
        )
    elif pending_count > 0:
        action_title = "👉 Pasul 3: Validează predicțiile cu scor scăzut în Workspace"
        action_text = (
            f"Pipeline-ul AI a procesat datele! Au fost identificate <strong>{pending_count} mostre incerte</strong> "
            f"(scor de încredere sub pragul de <strong>{st.session_state.confidence_threshold:.0%}</strong>). "
            "Mergi la tab-ul <strong>✍️ Verificare Umană (HITL)</strong> pentru a le valida sau corecta."
        )
    else:
        action_title = "👉 Pasul 4: Pipeline finalizat! Descarcă datele sau verifică metricile"
        action_text = (
            "Toate sarcinile din coadă au fost revizuite cu succes! "
            "Mergi la tab-ul <strong>📈 Calitate & Acord AI</strong> pentru raportul detaliat de acord inter-adnotatori (Cohen's Kappa) "
            "sau la <strong>📤 Export & Date</strong> pentru a descărca dataset-ul final (HuggingFace, CoNLL, DPO)."
        )

    st.markdown(
        f"""
        <div class="action-guidance-box">
            <div class="action-guidance-title">{action_title}</div>
            <div class="action-guidance-text">{action_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

        # Fetch all models from OpenRouter (cached after first call)
        if "all_models" not in st.session_state:
            with st.spinner("Loading models from OpenRouter..."):
                st.session_state.all_models = fetch_available_models()

        if st.button("🔄 Refresh Models", key="refresh_models", width="stretch"):
            with st.spinner("Refreshing..."):
                st.session_state.all_models = fetch_available_models(force_refresh=True)
            st.rerun()

        model_search = st.text_input(
            "🔍 Search models",
            value=st.session_state.get("model_search_query", ""),
            key="model_search_input",
            placeholder="e.g. llama, :free, mistral, gpt...",
            help="Type to filter. Use ':free' to show free models only.",
        )
        st.session_state.model_search_query = model_search

        filtered = search_models(model_search or "", st.session_state.all_models)

        if filtered:
            # Build display labels: "🆓 Model Name (id)" or "Model Name (id)"
            model_options = []
            model_id_map = {}
            for m in filtered:
                free_badge = "🆓 " if m["is_free"] else ""
                ctx = f"{m['context_length'] // 1000}k" if m['context_length'] else "?"
                display = f"{free_badge}{m['name']}  [{ctx} ctx]"
                model_options.append(display)
                model_id_map[display] = m["id"]

            # Try to preserve current selection
            current_display = None
            for disp, mid in model_id_map.items():
                if mid == st.session_state.model:
                    current_display = disp
                    break

            default_idx = (
                model_options.index(current_display)
                if current_display and current_display in model_options
                else 0
            )

            selected_display = st.selectbox(
                "Select Model",
                model_options,
                index=default_idx,
                key="sidebar_model",
            )
            st.session_state.model = model_id_map[selected_display]

            # Show selected model ID
            st.caption(f"`{st.session_state.model}`")
        else:
            st.warning("No models match your search.")
            st.caption(f"Current: `{st.session_state.model}`")

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
    """Render the executive pipeline dashboard and quick-start onboarding tab."""
    st.markdown('<div class="section-header">🚀 Start Rapid & Pipeline Dashboard</div>', unsafe_allow_html=True)

    tasks = st.session_state.tasks
    reviews = st.session_state.reviews

    if not tasks:
        # Onboarding Hero Container
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(49, 46, 129, 0.4));
                border: 1px solid rgba(129, 140, 248, 0.3);
                border-radius: 16px;
                padding: 1.8rem;
                margin-bottom: 1.5rem;
            ">
                <h3 style="color: #f8fafc; margin-top: 0; font-size: 1.35rem;">
                    👋 Bun venit în SmartAnnotate-AI!
                </h3>
                <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6; margin-bottom: 1.2rem;">
                    Acest sistem implementează un pipeline inteligent <strong>Human-in-the-Loop (HITL)</strong>
                    cu clasificare automată și rutare pe baza scorurilor de încredere (Confidence Routing):
                </p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 0.5rem;">
                    <div style="background: rgba(15, 23, 42, 0.7); padding: 1.1rem; border-radius: 10px; border-left: 3px solid #6366f1;">
                        <div style="font-weight: 700; color: #818cf8; margin-bottom: 5px;">1. 🤖 AI Pre-labeling</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                            LLM-ul analizează textele brute și generează etichete de sentiment/intenție sau entități NER cu probabilități calibrate.
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.7); padding: 1.1rem; border-radius: 10px; border-left: 3px solid #10b981;">
                        <div style="font-weight: 700; color: #34d399; margin-bottom: 5px;">2. ⚡ Confidence Routing</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                            Predicțiile cu scor ≥ 85% sunt <strong>auto-aprobate instantaneu</strong>, economisind peste 70% din costul adnotării manuale.
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.7); padding: 1.1rem; border-radius: 10px; border-left: 3px solid #f97316;">
                        <div style="font-weight: 700; color: #fb923c; margin-bottom: 5px;">3. ✍️ Active Human Review</div>
                        <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                            Doar predicțiile ambigue sau incerte (&lt; 85%) sunt direcționate către adnotatori umani pentru acceptare, editare sau respingere.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### ⚡ Testează instantaneu (Demo 1-Click)")
        col_demo1, col_demo2 = st.columns([2, 1])
        with col_demo1:
            demo_clicked = st.button(
                "⚡ 1-Click Demo: Încarcă 15 Mostre & Rulează Pipeline AI",
                key="btn_1click_demo",
                type="primary",
                width="stretch",
                help="Încarcă setul de date demonstrativ din data/sample_raw.jsonl și execută pre-etichetarea imediat.",
            )
            if demo_clicked:
                records = load_sample_records()
                if records:
                    st.session_state.uploaded_records = records
                    execute_pipeline(records)
                else:
                    st.error("❌ Nu s-a găsit fișierul demonstrativ `data/sample_raw.jsonl`.")
        with col_demo2:
            st.info("💡 **Fără cheie API?** Nicio problemă: pipeline-ul rulează automat în **mod Simulare Heuristică** cu scoruri de încredere reale.")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📂 Sau încarcă propriul fișier de date (CSV / JSONL)"):
            up_file = st.file_uploader(
                "Selectează un fișier CSV sau JSONL",
                type=["csv", "jsonl", "json"],
                key="dashboard_uploader",
            )
            if up_file:
                try:
                    if up_file.name.endswith(".csv"):
                        df = pd.read_csv(up_file)
                        col = st.selectbox("Selectează coloana cu textul", df.columns.tolist(), key="dash_text_col")
                        rec_list = [RawTextRecord(text=str(row[col]), metadata=row.to_dict()) for _, row in df.iterrows()]
                    else:
                        content = up_file.read().decode("utf-8")
                        raw_lines = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
                        rec_list = [RawTextRecord.model_validate(r) for r in raw_lines]

                    st.session_state.uploaded_records = rec_list
                    st.success(f"✅ S-au încărcat {len(rec_list)} înregistrări din `{up_file.name}`.")

                    if st.button("🤖 Rulează Pipeline AI pe fișierul încărcat", type="primary", width="stretch"):
                        execute_pipeline(rec_list)
                except Exception as e:
                    st.error(f"Eroare la procesarea fișierului: {e}")
        return

    stats = triage_statistics(tasks)
    pending_tasks = get_pending_tasks()

    # Pipeline Status Banner
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(99, 102, 241, 0.12));
            border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 12px;
            padding: 1rem 1.4rem;
            margin-bottom: 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        ">
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #34d399;">
                    ✅ Pipeline Activ: {stats['total']} Înregistrări Procesate
                </div>
                <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 3px;">
                    Model: <code style="color: #a78bfa;">{st.session_state.model}</code> • 
                    Prag Încredere: <strong style="color: #fb923c;">{st.session_state.confidence_threshold:.0%}</strong> • 
                    Sarcina: <strong style="color: #38bdf8;">{st.session_state.task_type.value if hasattr(st.session_state.task_type, 'value') else st.session_state.task_type}</strong>
                </div>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span style="background: rgba(16, 185, 129, 0.25); border: 1px solid #10b981; padding: 4px 12px; border-radius: 20px; color: #6ee7b7; font-size: 0.85rem; font-weight: 700;">
                    🟢 {stats['auto_approved']} Auto-Aprobate ({stats['auto_approved_pct']}%)
                </span>
                <span style="background: rgba(249, 115, 22, 0.25); border: 1px solid #f97316; padding: 4px 12px; border-radius: 20px; color: #fdba74; font-size: 0.85rem; font-weight: 700;">
                    🟡 {stats['human_review']} Verificare Umană ({100 - stats['auto_approved_pct']:.1f}%)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # If pending tasks remain, display a prominent guidance alert
    if pending_tasks:
        st.info(
            f"👉 **Acțiune necesară:** Ai **{len(pending_tasks)} mostre** direcționate către verificarea umană din cauza scorului de încredere scăzut. "
            "Mergi în tab-ul **✍️ Verificare Umană (HITL)** din bara de sus pentru a le valida/edita."
        )

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
        backlog = len(pending_tasks)
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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig2, width="stretch")

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

    # Preview table of processed records
    with st.expander("🔍 Vezi toate înregistrările procesate în pipeline", expanded=False):
        task_summary = []
        for i, t in enumerate(tasks):
            task_summary.append({
                "#": i + 1,
                "Text": t.record.text[:75] + "..." if len(t.record.text) > 75 else t.record.text,
                "Încredere": f"{t.overall_confidence:.1%}",
                "Rutare": "🟢 Auto-Aprobat" if (t.routing == RoutingDecision.AUTO_APPROVED or t.routing == "auto_approved") else "🟡 Verificare Umană",
                "Status": "✅ Revizuit" if t.id in reviews else "⏳ În așteptare",
            })
        st.dataframe(pd.DataFrame(task_summary), width="stretch", hide_index=True)


# ──────────────────────────────────────────────
# Tab 2: Annotation Workspace
# ──────────────────────────────────────────────

def render_annotation_workspace():
    """Render the interactive annotation workspace."""
    st.markdown('<div class="section-header">✍️ Spațiu de Adnotare & Validare Umană (HITL)</div>', unsafe_allow_html=True)

    tasks = st.session_state.tasks
    if not tasks:
        st.info("📂 Nu există sarcini încărcate în pipeline. Mergi la tab-ul **🚀 Start Rapid & Dashboard** și apasă pe **⚡ 1-Click Demo** pentru a rula pipeline-ul!")
        return

    # Queue selector
    queue_option = st.radio(
        "Filtru Coadă de Lucru",
        ["Toate Sarcinile", "Doar Verificare Umană (Incertitudine AI)", "Auto-Aprobate (Spot Check)", "Doar Nerevizuite"],
        horizontal=True,
        key="queue_filter",
    )

    if queue_option == "Doar Verificare Umană (Incertitudine AI)":
        filtered = [t for t in tasks if t.routing == RoutingDecision.HUMAN_REVIEW or t.routing == "human_review"]
    elif queue_option == "Auto-Aprobate (Spot Check)":
        filtered = [t for t in tasks if t.routing == RoutingDecision.AUTO_APPROVED or t.routing == "auto_approved"]
    elif queue_option == "Doar Nerevizuite":
        filtered = get_pending_tasks()
    else:
        filtered = tasks

    if not filtered:
        st.success("✅ Toate sarcinile din această coadă au fost revizuite!")
        return

    # Navigation
    total = len(filtered)
    idx = st.session_state.current_index % total if total > 0 else 0

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 3])
    with nav_col1:
        if st.button("⬅ Înapoi", key="btn_prev", width="stretch"):
            st.session_state.current_index = max(0, idx - 1)
            st.session_state.review_start_time = time.time()
            st.rerun()
    with nav_col2:
        if st.button("Înainte ➡", key="btn_next", width="stretch"):
            st.session_state.current_index = min(total - 1, idx + 1)
            st.session_state.review_start_time = time.time()
            st.rerun()
    with nav_col3:
        # Jump to specific record
        jump = st.number_input(
            "Sari la #", min_value=1, max_value=total, value=idx + 1, key="jump_to"
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
    st.markdown(f"**Înregistrarea {idx + 1} din {total}** — ID: `{task.record.id}`", unsafe_allow_html=True)
    header_col1, header_col2, header_col3 = st.columns([2, 2, 2])
    with header_col1:
        st.markdown(render_confidence_gauge(task.overall_confidence), unsafe_allow_html=True)
    with header_col2:
        st.markdown(render_routing_badge(task.routing), unsafe_allow_html=True)
    with header_col3:
        if is_reviewed:
            review = st.session_state.reviews[task.id]
            st.success(f"✅ Revizuit: {review.action}")
        else:
            st.warning("⏳ În așteptare verificare")

    # Routing explanation banner
    if task.routing == RoutingDecision.HUMAN_REVIEW or task.routing == "human_review":
        st.markdown(
            f"""
            <div style="background: rgba(249, 115, 22, 0.12); border-left: 4px solid #f97316; border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; margin: 0.8rem 0 1rem 0;">
                <strong style="color: #fb923c;">⚠️ De ce este această mostră în coada de revizuire?</strong>
                <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 3px;">
                    Modelul AI are un scor de încredere de <strong>{task.overall_confidence:.1%}</strong>, fiind <strong>sub pragul de {st.session_state.confidence_threshold:.0%}</strong>.
                    Modelul este nesigur. Verifică predicția și alege o acțiune: <strong>Acceptă</strong> (dacă e corectă), <strong>Editează</strong> (corectează etichetele) sau <strong>Respinge</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border-left: 4px solid #10b981; border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; margin: 0.8rem 0 1rem 0;">
                <strong style="color: #34d399;">🟢 Mostră Auto-Aprobată de AI (Spot Check / Audit)</strong>
                <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 3px;">
                    Scor de încredere ridicat: <strong>{task.overall_confidence:.1%}</strong> (≥ {st.session_state.confidence_threshold:.0%}).
                    Această înregistrare a fost aprobată automat fără cost de muncă manuală. Ești în modul de auditare a calității.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Text display
    main_col, side_col = st.columns([3, 1])

    with main_col:
        st.markdown("**Text Original:**")
        if task.task_type == TaskType.NER or task.task_type == "ner":
            entities = task.ner.entities if task.ner else []
            st.markdown(render_ner_text(task.record.text, entities), unsafe_allow_html=True)

            # Entity table
            if entities:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Entități Extrase:**")
                ent_data = []
                for e in entities:
                    ent_data.append({
                        "Entitate": e.text,
                        "Etichetă": e.label if isinstance(e.label, str) else e.label.value,
                        "Start": e.start,
                        "End": e.end,
                        "Încredere": f"{e.confidence:.1%}",
                    })
                st.dataframe(pd.DataFrame(ent_data), width="stretch", hide_index=True)
        else:
            st.markdown(f'<div class="text-display">{task.record.text}</div>', unsafe_allow_html=True)

    with side_col:
        st.markdown("**Predicție AI:**")
        if task.task_type == TaskType.CLASSIFICATION or task.task_type == "classification":
            if task.classification:
                st.markdown(f"🏷️ **Sentiment:** `{task.classification.sentiment}`")
                st.markdown(f"🎯 **Intenție:** `{task.classification.intent}`")
                st.markdown(f"📊 **Încredere:** `{task.classification.confidence:.1%}`")
                with st.expander("💭 Raționament AI"):
                    st.write(task.classification.reasoning)
        elif task.ner and task.ner.entities:
            st.markdown(f"📊 **Încredere Generală:** `{task.ner.confidence:.1%}`")
            st.markdown(f"🔍 **Entități găsite:** `{len(task.ner.entities)}`")

        # Metadata
        st.markdown("---")
        st.markdown("**Metadate:**")
        st.caption(f"Sursă: {task.record.source or 'N/A'}")
        st.caption(f"Limbă: {task.record.language or 'N/A'}")
        st.caption(f"Model utilizat: {task.model_used}")

    st.markdown("---")

    # Annotation Actions
    st.markdown("**Acțiuni de Decizie & Revizuire:**")

    if task.task_type == TaskType.CLASSIFICATION or task.task_type == "classification":
        _render_classification_actions(task)
    else:
        _render_ner_actions(task)

    # Guidelines expander
    with st.expander("📖 Ghid Rapid de Adnotare (SOP & Criterii)", expanded=False):
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown(
                """
                **🏷️ Sentiment & Intenție:**
                - **Positive**: client mulțumit, apreciere, laudă
                - **Negative**: frustrare, defecte, reclamații, erori
                - **Neutral**: întrebări simple, fapte obiective
                - **Intenții**: `bug_report`, `feature_request`, `complaint`, `general_inquiry`, `feedback`
                """
            )
        with g_col2:
            st.markdown(
                """
                **🎯 Rolul Butoanelor de Decizie:**
                - **✅ Acceptă Predicția**: Aprobă eticheta generată de AI fără nicio modificare.
                - **✏️ Editează Etichetele**: Corectează valorile de sentiment/intenție dacă AI a greșit.
                - **❌ Respinge Mostra**: Text ilizibil, spam sau complet irelevant.
                
                **⌨️ Taste rapide:** `A` = Acceptă, `E` = Editează, `R` = Respinge
                """
            )


def _render_classification_actions(task: AnnotationTask):
    """Render classification-specific review actions."""
    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        accept = st.button(
            "✅ Acceptă Predicția AI (A)",
            key=f"accept_{task.id}",
            width="stretch",
            type="primary",
            help="Aprobă etichetele AI fără modificări",
        )

    with action_col2:
        edit_mode = st.button(
            "✏️ Modifică Etichetele (E)",
            key=f"edit_{task.id}",
            width="stretch",
            help="Deschide editorul pentru a corecta valorile AI",
        )

    with action_col3:
        reject = st.button(
            "❌ Respinge Mostra (R)",
            key=f"reject_{task.id}",
            width="stretch",
            help="Marchează ca nevalidă sau predicție eronată",
        )

    # Edit form (always visible, used when Edit is clicked)
    with st.expander("✏️ Formular de Corectare Etichete", expanded=edit_mode):
        edit_col1, edit_col2 = st.columns(2)
        with edit_col1:
            sentiment_options = [e.value for e in SentimentLabel]
            current_sent = task.classification.sentiment if task.classification else "neutral"
            if isinstance(current_sent, SentimentLabel):
                current_sent = current_sent.value
            edited_sentiment = st.selectbox(
                "Sentiment Corectat",
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
                "Intenție Corectată",
                intent_options,
                index=intent_options.index(current_intent) if current_intent in intent_options else 0,
                key=f"edit_int_{task.id}",
            )
        if st.button("💾 Salvează Modificările", key=f"save_edit_{task.id}", type="primary", width="stretch"):
            _submit_review(task, ReviewAction.EDIT, edited_sentiment, edited_intent)

    # Comment field
    comments = st.text_area(
        "Comentarii Adnotator / Observații",
        key=f"comments_{task.id}",
        placeholder="Opțional: explică decizia sau notează ambiguități...",
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
            "✅ Acceptă Entitățile AI (A)",
            key=f"accept_{task.id}",
            width="stretch",
            type="primary",
            help="Aprobă entitățile extrase de AI",
        )

    with action_col2:
        edit_mode = st.button(
            "✏️ Editează Entitățile (E)",
            key=f"edit_{task.id}",
            width="stretch",
            help="Adaugă sau modifică entități",
        )

    with action_col3:
        reject = st.button(
            "❌ Respinge Mostra (R)",
            key=f"reject_{task.id}",
            width="stretch",
        )

    # NER editing interface
    with st.expander("✏️ Editor Tabelar Entități", expanded=edit_mode):
        st.markdown("Modifică entitățile de mai jos. Poți adăuga sau șterge rânduri.")

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
                    "Etichetă",
                    options=[e.value for e in NERLabel],
                    required=True,
                ),
                "start": st.column_config.NumberColumn("Start", min_value=0),
                "end": st.column_config.NumberColumn("End", min_value=0),
            },
            num_rows="dynamic",
            width="stretch",
            key=f"ner_editor_{task.id}",
        )

        if st.button("💾 Salvează Entitățile Modificate", key=f"save_ner_{task.id}", type="primary", width="stretch"):
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
            st.dataframe(pd.DataFrame(label_rows), width="stretch", hide_index=True)

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
                "📖 **Ghid de Interpretare Acord:**\n"
                "- **> 0.8**: Acord aproape perfect (AI este gata de producție)\n"
                "- **0.6–0.8**: Acord substanțial (aliniere foarte bună)\n"
                "- **0.4–0.6**: Acord moderat (sunt necesare câteva revizuiri)\n"
                "- **0.2–0.4**: Acord rezonabil\n"
                "- **< 0.2**: Acord slab (AI necesită re-antrenare/prompting nou)"
            )
    else:
        st.warning("Nu există suficiente sarcini de clasificare revizuite pentru calcularea acordului.")

    # Fleiss' Kappa demo
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📐 Acord Multi-Adnotator (Fleiss' Kappa)")
    st.info(
        "Fleiss' Kappa măsoară acordul când mai mulți adnotatori umani evaluează aceleași mostre. "
        "Încarcă un set de date multi-adnotator sau invită colegi pentru a activa metrica live."
    )

    # Demo calculation with synthetic data
    with st.expander("🧪 Demo: Calcul Fleiss' Kappa cu Date Sintetice"):
        st.markdown("Matrice eșantion 5 mostre × 3 categorii (3 adnotatori):")
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
        st.dataframe(demo_df, width="stretch")
        fk = fleiss_kappa(demo_matrix)
        st.markdown(f"**Fleiss' Kappa:** `{fk:.4f}`")


# ──────────────────────────────────────────────
# Tab 4: Data Management
# ──────────────────────────────────────────────

def render_data_management():
    """Render data ingestion and export tab."""
    st.markdown('<div class="section-header">📤 Ingestie & Export Seturi de Date</div>', unsafe_allow_html=True)

    # ── Upload Section ──
    st.markdown("### 📂 Ingestie Date Brute")

    upload_col1, upload_col2 = st.columns([2, 1])

    with upload_col1:
        uploaded_file = st.file_uploader(
            "Încarcă date text brute (CSV sau JSONL)",
            type=["csv", "jsonl", "json"],
            key="file_uploader",
        )

    with upload_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button(
            "📋 Încarcă Date Sample",
            key="load_sample",
            width="stretch",
        )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                text_col = st.selectbox(
                    "Selectează coloana de text",
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

            st.success(f"✅ S-au încărcat {len(records)} înregistrări din `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"❌ Eroare la citirea fișierului: {e}")

    if use_sample:
        records = load_sample_records()
        if records:
            st.session_state.uploaded_records = records
            st.success(f"✅ S-au încărcat {len(records)} înregistrări demonstrative.")
            st.rerun()
        else:
            st.error("❌ Fișierul `data/sample_raw.jsonl` nu a fost găsit.")

    # Show loaded records preview
    if st.session_state.uploaded_records:
        st.markdown(f"**{len(st.session_state.uploaded_records)} înregistrări pregătite**")
        with st.expander("Previzualizare date încărcate"):
            preview = []
            for r in st.session_state.uploaded_records[:10]:
                preview.append({
                    "ID": r.id[:12] + "...",
                    "Text": r.text[:80] + "..." if len(r.text) > 80 else r.text,
                    "Sursă": r.source or "N/A",
                    "Limbă": r.language or "en",
                })
            st.dataframe(pd.DataFrame(preview), width="stretch", hide_index=True)

    # ── Run Pipeline ──
    st.markdown("---")
    st.markdown("### 🚀 Execută Pipeline-ul de Pre-Etichetare AI")

    if st.session_state.uploaded_records:
        run_col1, run_col2 = st.columns([1, 2])
        with run_col1:
            run_pipeline = st.button(
                "🤖 Rulează Pipeline AI",
                key="run_pipeline",
                type="primary",
                width="stretch",
                disabled=st.session_state.processing,
            )

        if run_pipeline:
            execute_pipeline(st.session_state.uploaded_records)
    else:
        st.warning("⬆️ Încarcă date mai sus sau apasă pe 'Încarcă Date Sample' pentru a rula pipeline-ul.")

    # ── Export Section ──
    st.markdown("---")
    st.markdown("### 📦 Exportă Adnotările Finale")

    if not st.session_state.tasks:
        st.info("Nu există date adnotate gata de export. Rulează mai întâi pipeline-ul AI.")
        return

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        st.markdown("**🤗 Hugging Face JSONL**")
        st.caption("Format ideal pentru text classification și fine-tuning Transformers.")
        if st.button("Export HF JSONL", key="export_hf", width="stretch"):
            try:
                path = export_huggingface_jsonl(
                    st.session_state.tasks,
                    st.session_state.reviews,
                )
                summary = get_export_summary(path)
                st.success(f"✅ Exportat {summary['records']} înregistrări ({summary['size_kb']} KB)")
                with open(path, "r") as f:
                    st.download_button(
                        "⬇️ Descarcă JSONL",
                        f.read(),
                        file_name=summary["filename"],
                        mime="application/jsonl",
                        width="stretch",
                    )
            except Exception as e:
                st.error(f"Eroare la export: {e}")

    with export_col2:
        st.markdown("**🏷️ CoNLL / BIO Format**")
        st.caption("Format standard token-level pentru modele NER (SpaCy, Stanza).")
        if st.button("Export CoNLL", key="export_conll", width="stretch"):
            try:
                path = export_conll_bio(
                    st.session_state.tasks,
                    st.session_state.reviews,
                )
                summary = get_export_summary(path)
                st.success(f"✅ Exportat ({summary['size_kb']} KB)")
                with open(path, "r") as f:
                    st.download_button(
                        "⬇️ Descarcă CoNLL",
                        f.read(),
                        file_name=summary["filename"],
                        mime="text/plain",
                        width="stretch",
                    )
            except Exception as e:
                st.error(f"Eroare la export: {e}")

    with export_col3:
        st.markdown("**🎯 DPO / RLHF Pairs**")
        st.caption("Perechi (Prompt, Chosen, Rejected) pentru Direct Preference Optimization.")
        if st.button("Export DPO", key="export_dpo", width="stretch"):
            try:
                path = export_dpo_pairs(
                    st.session_state.tasks,
                    st.session_state.reviews,
                )
                summary = get_export_summary(path)
                st.success(f"✅ Exportat {summary['records']} perechi ({summary['size_kb']} KB)")
                with open(path, "r") as f:
                    st.download_button(
                        "⬇️ Descarcă DPO",
                        f.read(),
                        file_name=summary["filename"],
                        mime="application/jsonl",
                        width="stretch",
                    )
            except Exception as e:
                st.error(f"Eroare la export: {e}")


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

def main():
    """Main application entry point."""
    render_sidebar()

    # Header
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 0.3rem 0;">
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

    # Visual Pipeline Stepper & Action Guidance
    render_pipeline_stepper()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Start Rapid & Dashboard",
        "✍️ Verificare Umană (HITL)",
        "📈 Calitate & Acord AI",
        "📤 Export & Date",
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
