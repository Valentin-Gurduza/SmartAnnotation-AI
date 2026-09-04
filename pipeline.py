"""
SmartAnnotate-AI — AI Pre-Labeling Pipeline
=============================================
Asynchronous API client for OpenRouter with confidence scoring,
batch processing, and automated confidence-based routing.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import httpx
from dotenv import load_dotenv
from rich.console import Console

from schemas import (
    AnnotationTask,
    BatchConfig,
    BatchResult,
    ClassificationResult,
    IntentLabel,
    NEREntity,
    NERLabel,
    NERResult,
    RawTextRecord,
    RoutingDecision,
    SentimentLabel,
    TaskType,
)

load_dotenv()
console = Console()

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.3-70b-instruct")
DEFAULT_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-4o-mini",
    "mistralai/mistral-7b-instruct",
    "google/gemini-flash-1.5",
    "anthropic/claude-3.5-haiku",
]

# Module-level cache for fetched models
_models_cache: dict = {"models": [], "fetched": False}


def fetch_available_models(force_refresh: bool = False) -> list[dict]:
    """
    Fetch all available models from OpenRouter's /models endpoint.

    Returns a list of dicts with keys: id, name, pricing, context_length.
    Results are cached after the first successful fetch.
    """
    if _models_cache["fetched"] and not force_refresh:
        return _models_cache["models"]

    try:
        with httpx.Client(timeout=15) as client:
            response = client.get(f"{OPENROUTER_BASE_URL}/models")
            response.raise_for_status()
            data = response.json()

        models = []
        for m in data.get("data", []):
            model_id = m.get("id", "")
            name = m.get("name", model_id)
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt", "0") or "0")
            completion_price = float(pricing.get("completion", "0") or "0")
            is_free = prompt_price == 0 and completion_price == 0
            context_length = m.get("context_length", 0)

            models.append({
                "id": model_id,
                "name": name,
                "is_free": is_free,
                "context_length": context_length,
                "prompt_price": prompt_price,
                "completion_price": completion_price,
            })

        # Sort: free models first, then by name
        models.sort(key=lambda x: (not x["is_free"], x["name"].lower()))

        _models_cache["models"] = models
        _models_cache["fetched"] = True
        console.print(f"[green]✓ Fetched {len(models)} models from OpenRouter[/green]")
        return models

    except Exception as e:
        console.print(f"[yellow]⚠ Could not fetch models from OpenRouter: {e}[/yellow]")
        # Return fallback models as dicts
        fallback = [
            {"id": mid, "name": mid.split("/")[-1], "is_free": False,
             "context_length": 0, "prompt_price": 0, "completion_price": 0}
            for mid in FALLBACK_MODELS
        ]
        _models_cache["models"] = fallback
        _models_cache["fetched"] = True
        return fallback


def search_models(query: str, models: list[dict] | None = None) -> list[dict]:
    """
    Filter models by search query (matches against id and name).

    Args:
        query: Search string (case-insensitive). Use ':free' to filter free models.
        models: Optional pre-fetched model list; fetches if None.

    Returns:
        Filtered list of model dicts.
    """
    if models is None:
        models = fetch_available_models()

    if not query.strip():
        return models

    query_lower = query.lower().strip()
    tokens = query_lower.split()

    results = []
    for m in models:
        searchable = f"{m['id']} {m['name']}".lower()
        if m["is_free"]:
            searchable += " :free free"

        # All tokens must match
        if all(token in searchable for token in tokens):
            results.append(m)

    return results


# Keep a simple list accessor for backward compatibility
AVAILABLE_MODELS = FALLBACK_MODELS

# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────

CLASSIFICATION_SYSTEM_PROMPT = """You are an expert text annotation assistant. Your task is to classify the given text with a sentiment label, an intent label, and provide your reasoning.

You MUST respond with a valid JSON object and nothing else. Do not include markdown formatting, code fences, or any other text outside the JSON.

The JSON must have exactly these fields:
{
  "sentiment": one of ["positive", "negative", "neutral", "mixed"],
  "intent": one of ["complaint", "inquiry", "feedback", "request", "escalation", "other"],
  "confidence": a float between 0.0 and 1.0 representing your certainty,
  "reasoning": a brief explanation of your classification decision
}

Be precise with confidence scoring:
- 0.9-1.0: Very clear, unambiguous text
- 0.7-0.89: Reasonably clear with minor ambiguity
- 0.5-0.69: Significant ambiguity or mixed signals
- Below 0.5: Very unclear or insufficient information"""

CLASSIFICATION_USER_TEMPLATE = """Classify the following text:

---
{text}
---

Respond with a JSON object only."""

NER_SYSTEM_PROMPT = """You are an expert Named Entity Recognition (NER) annotator. Extract all named entities from the given text with their exact character offsets.

You MUST respond with a valid JSON object and nothing else. Do not include markdown formatting, code fences, or any other text outside the JSON.

The JSON must have exactly these fields:
{
  "entities": [
    {
      "text": "exact entity text as it appears",
      "label": one of ["PERSON", "ORG", "LOC", "DATE", "PRODUCT", "MONETARY", "EVENT", "MISC"],
      "start": character offset where entity starts (0-indexed),
      "end": character offset where entity ends (exclusive),
      "confidence": float between 0.0 and 1.0
    }
  ],
  "confidence": overall confidence float between 0.0 and 1.0,
  "reasoning": brief explanation of extraction decisions
}

Rules:
- Character offsets must exactly match the entity text position in the input
- Use exclusive end offsets (text[start:end] == entity text)
- Include ALL entities, even low-confidence ones
- Be precise with confidence scoring based on entity boundary clarity"""

NER_USER_TEMPLATE = """Extract all named entities from the following text:

---
{text}
---

Respond with a JSON object only."""


# ──────────────────────────────────────────────
# OpenRouter API Client
# ──────────────────────────────────────────────

class OpenRouterClient:
    """Async HTTP client for the OpenRouter chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            console.print(
                "[yellow]⚠ OPENROUTER_API_KEY not set. "
                "Pipeline will use simulated responses.[/yellow]"
            )

    async def chat_completion(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Send a chat completion request to OpenRouter.
        Returns the parsed JSON response content.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/smartannotate-ai",
            "X-Title": "SmartAnnotate-AI",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()

                    content = data["choices"][0]["message"]["content"]
                    return json.loads(content)

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    # Rate limited — exponential backoff
                    wait = 2 ** attempt
                    console.print(
                        f"[yellow]Rate limited. Retrying in {wait}s "
                        f"(attempt {attempt}/{self.max_retries})...[/yellow]"
                    )
                    await asyncio.sleep(wait)
                elif e.response.status_code >= 500:
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                else:
                    raise
            except (httpx.RequestError, json.JSONDecodeError, KeyError) as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(1)

        raise RuntimeError(
            f"OpenRouter API call failed after {self.max_retries} retries: "
            f"{last_error}"
        )


# ──────────────────────────────────────────────
# Simulated Responses (for demo / no API key)
# ──────────────────────────────────────────────

def _simulate_classification(text: str) -> dict:
    """Generate a simulated classification response for demo mode."""
    import hashlib
    import random

    # Deterministic seed from text for consistency
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    text_lower = text.lower()

    # Heuristic-based simulation
    if any(w in text_lower for w in ["angry", "terrible", "worst", "broken", "fail", "hate", "awful"]):
        sentiment = "negative"
        intent = rng.choice(["complaint", "escalation"])
        confidence = rng.uniform(0.82, 0.96)
    elif any(w in text_lower for w in ["love", "great", "excellent", "amazing", "thank", "perfect"]):
        sentiment = "positive"
        intent = "feedback"
        confidence = rng.uniform(0.85, 0.97)
    elif any(w in text_lower for w in ["how", "where", "when", "what", "can you", "?"]):
        sentiment = "neutral"
        intent = "inquiry"
        confidence = rng.uniform(0.75, 0.92)
    elif any(w in text_lower for w in ["but", "however", "although", "mixed"]):
        sentiment = "mixed"
        intent = "feedback"
        confidence = rng.uniform(0.55, 0.78)
    else:
        sentiment = rng.choice(["neutral", "mixed"])
        intent = rng.choice(["inquiry", "feedback", "other"])
        confidence = rng.uniform(0.50, 0.80)

    return {
        "sentiment": sentiment,
        "intent": intent,
        "confidence": round(confidence, 3),
        "reasoning": f"Text analysis indicates {sentiment} sentiment with {intent} intent. "
                     f"Key signals: tone, word choice, and context.",
    }


def _simulate_ner(text: str) -> dict:
    """Generate simulated NER results using simple pattern matching."""
    import re

    entities = []

    # Simple date patterns
    for match in re.finditer(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4}\b",
        text,
    ):
        entities.append({
            "text": match.group(),
            "label": "DATE",
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.92,
        })

    # Money patterns
    for match in re.finditer(r"\$[\d,]+\.?\d*|\b\d+\s*(?:USD|EUR|dollars)\b", text):
        entities.append({
            "text": match.group(),
            "label": "MONETARY",
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.95,
        })

    # Capitalized word sequences (potential names / orgs)
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
        word = match.group()
        # Skip common non-entity phrases
        if word.lower() in {"the", "this", "that", "these", "please", "thank you"}:
            continue
        # Heuristic: longer = more likely ORG, shorter = PERSON
        label = "ORG" if len(word.split()) > 2 else "PERSON"
        entities.append({
            "text": word,
            "label": label,
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.72,
        })

    import hashlib
    import random
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    overall_conf = round(rng.uniform(0.65, 0.93), 3) if entities else round(rng.uniform(0.50, 0.75), 3)

    return {
        "entities": entities,
        "confidence": overall_conf,
        "reasoning": f"Extracted {len(entities)} entities using contextual analysis.",
    }


# ──────────────────────────────────────────────
# Parsing & Validation
# ──────────────────────────────────────────────

def parse_classification(raw: dict) -> ClassificationResult:
    """Parse and validate a raw JSON dict into a ClassificationResult."""
    # Normalize keys to lowercase
    normalized = {k.lower(): v for k, v in raw.items()}

    # Handle potential enum mismatches gracefully
    if "sentiment" in normalized:
        normalized["sentiment"] = str(normalized["sentiment"]).lower().strip()
    if "intent" in normalized:
        normalized["intent"] = str(normalized["intent"]).lower().strip()

    return ClassificationResult.model_validate(normalized)


def parse_ner(raw: dict) -> NERResult:
    """Parse and validate a raw JSON dict into an NERResult."""
    normalized = {k.lower(): v for k, v in raw.items()}

    # Validate each entity
    valid_entities = []
    for ent in normalized.get("entities", []):
        try:
            entity = NEREntity.model_validate(ent)
            valid_entities.append(entity)
        except Exception:
            # Skip invalid entities rather than failing the whole record
            continue

    return NERResult(
        entities=valid_entities,
        confidence=normalized.get("confidence", 0.5),
        reasoning=normalized.get("reasoning", ""),
    )


# ──────────────────────────────────────────────
# Confidence Scoring & Routing
# ──────────────────────────────────────────────

def compute_confidence(
    result: ClassificationResult | NERResult,
    task_type: TaskType,
) -> float:
    """
    Compute a normalized confidence score.

    For classification: uses the model's reported confidence with
    a penalty if reasoning is very short (likely hallucinated).

    For NER: averages entity-level confidences weighted by the
    overall model confidence.
    """
    if task_type == TaskType.CLASSIFICATION:
        assert isinstance(result, ClassificationResult)
        score = result.confidence

        # Penalty for very short reasoning (< 20 chars) — likely low quality
        if len(result.reasoning) < 20:
            score *= 0.85

        return round(max(0.0, min(1.0, score)), 4)

    else:
        assert isinstance(result, NERResult)
        if not result.entities:
            return round(result.confidence * 0.9, 4)

        entity_avg = sum(e.confidence for e in result.entities) / len(result.entities)
        # Weighted blend: 60% overall, 40% entity average
        score = 0.6 * result.confidence + 0.4 * entity_avg
        return round(max(0.0, min(1.0, score)), 4)


def route_task(
    confidence: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> RoutingDecision:
    """Route a task based on confidence threshold."""
    if confidence >= threshold:
        return RoutingDecision.AUTO_APPROVED
    return RoutingDecision.HUMAN_REVIEW


# ──────────────────────────────────────────────
# Single Record Processing
# ──────────────────────────────────────────────

async def process_single_record(
    record: RawTextRecord,
    config: BatchConfig,
    client: OpenRouterClient,
) -> AnnotationTask:
    """Process a single record through the pre-labeling pipeline."""
    task = AnnotationTask(
        record=record,
        task_type=config.task_type,
        model_used=config.model,
    )

    try:
        if config.task_type == TaskType.CLASSIFICATION:
            # Build prompts
            messages = [
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": CLASSIFICATION_USER_TEMPLATE.format(text=record.text)},
            ]

            # Call API or simulate
            if client.api_key:
                raw = await client.chat_completion(
                    messages=messages,
                    model=config.model,
                    temperature=config.temperature,
                )
            else:
                raw = _simulate_classification(record.text)

            result = parse_classification(raw)
            task.classification = result
            task.overall_confidence = compute_confidence(result, TaskType.CLASSIFICATION)

        else:  # NER
            messages = [
                {"role": "system", "content": NER_SYSTEM_PROMPT},
                {"role": "user", "content": NER_USER_TEMPLATE.format(text=record.text)},
            ]

            if client.api_key:
                raw = await client.chat_completion(
                    messages=messages,
                    model=config.model,
                    temperature=config.temperature,
                )
            else:
                raw = _simulate_ner(record.text)

            result = parse_ner(raw)
            task.ner = result
            task.overall_confidence = compute_confidence(result, TaskType.NER)

        # Route
        task.routing = route_task(task.overall_confidence, config.confidence_threshold)
        task.prelabeled_at = datetime.now(timezone.utc)

    except Exception as e:
        console.print(f"[red]Error processing record {record.id}: {e}[/red]")
        task.overall_confidence = 0.0
        task.routing = RoutingDecision.HUMAN_REVIEW

    return task


# ──────────────────────────────────────────────
# Batch Processing
# ──────────────────────────────────────────────

async def process_batch(
    records: list[RawTextRecord],
    config: BatchConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> BatchResult:
    """
    Process a batch of records through the pre-labeling pipeline.
    Uses semaphore-based concurrency control.
    """
    start_time = time.time()
    client = OpenRouterClient()
    semaphore = asyncio.Semaphore(config.max_concurrent)
    completed = 0
    tasks: list[AnnotationTask] = []

    async def _process_with_semaphore(record: RawTextRecord) -> AnnotationTask:
        nonlocal completed
        async with semaphore:
            result = await process_single_record(record, config, client)
            completed += 1
            if progress_callback:
                progress_callback(completed, len(records))
            return result

    # Run all records concurrently (bounded by semaphore)
    tasks = await asyncio.gather(
        *[_process_with_semaphore(r) for r in records],
        return_exceptions=False,
    )

    # Compute batch statistics
    elapsed = time.time() - start_time
    successful = [t for t in tasks if t.overall_confidence > 0]
    auto_approved = [t for t in tasks if t.routing == RoutingDecision.AUTO_APPROVED]

    avg_conf = (
        sum(t.overall_confidence for t in successful) / len(successful)
        if successful
        else 0.0
    )

    return BatchResult(
        total_records=len(records),
        successful=len(successful),
        failed=len(records) - len(successful),
        auto_approved=len(auto_approved),
        human_review=len(successful) - len(auto_approved),
        avg_confidence=round(avg_conf, 4),
        processing_time_seconds=round(elapsed, 2),
        tasks=list(tasks),
    )


# ──────────────────────────────────────────────
# CLI Entry Point (for testing)
# ──────────────────────────────────────────────

def main():
    """Quick CLI test of the pipeline."""
    from rich.table import Table

    sample_texts = [
        "Your product is absolutely terrible. I want a full refund immediately!",
        "Can you tell me how to reset my password?",
        "I love the new update! The dark mode is beautiful.",
        "The delivery was late but the product quality is decent I guess.",
        "I need to speak with a manager about my account being charged twice.",
    ]

    records = [RawTextRecord(text=t) for t in sample_texts]
    config = BatchConfig(task_type=TaskType.CLASSIFICATION)

    console.print("\n[bold cyan]SmartAnnotate-AI Pipeline Test[/bold cyan]\n")

    result = asyncio.run(process_batch(records, config))

    table = Table(title="Pre-Labeling Results")
    table.add_column("Text (truncated)", width=40)
    table.add_column("Sentiment", style="cyan")
    table.add_column("Intent", style="green")
    table.add_column("Confidence", justify="right")
    table.add_column("Routing", style="bold")

    for task in result.tasks:
        text_short = task.record.text[:37] + "..." if len(task.record.text) > 40 else task.record.text
        cls = task.classification
        conf_color = "green" if task.overall_confidence >= 0.85 else "yellow"
        routing_color = "green" if task.routing == "auto_approved" else "red"

        table.add_row(
            text_short,
            cls.sentiment if cls else "N/A",
            cls.intent if cls else "N/A",
            f"[{conf_color}]{task.overall_confidence:.3f}[/{conf_color}]",
            f"[{routing_color}]{task.routing}[/{routing_color}]",
        )

    console.print(table)
    console.print(f"\n[dim]Processed {result.total_records} records in {result.processing_time_seconds}s[/dim]")
    console.print(f"[dim]Auto-approved: {result.auto_approved} | Human review: {result.human_review}[/dim]\n")


if __name__ == "__main__":
    main()
