# 📋 SmartAnnotate-AI — Annotation Guidelines

## Standard Operating Procedure (SOP) for Human Annotators

**Version:** 1.0  
**Effective Date:** 2024-01-01  
**Last Updated:** 2024-09-04

---

## 1. Overview

This document defines the annotation standards, decision procedures, and quality expectations for human reviewers working with the SmartAnnotate-AI pipeline. All annotators must read and acknowledge these guidelines before beginning review work.

**Key Principle:** The AI pre-labeling system handles approximately 60% of annotations automatically. Your role is to verify high-confidence predictions and manually annotate ambiguous cases that require human judgment.

---

## 2. Label Taxonomy

### 2.1 Sentiment Labels

| Label | Definition | Examples |
|-------|-----------|----------|
| **Positive** | Text expresses satisfaction, praise, or positive emotion | "Love the new feature!", "Excellent support" |
| **Negative** | Text expresses dissatisfaction, frustration, or complaint | "Terrible experience", "Product is broken" |
| **Neutral** | Text is factual, procedural, or emotionally flat | "How do I reset my password?", "My order number is 4521" |
| **Mixed** | Text contains both positive AND negative signals | "Product is great but shipping was awful" |

### 2.2 Intent Labels

| Label | Definition | Examples |
|-------|-----------|----------|
| **Complaint** | Expressing dissatisfaction about a specific issue | "The product arrived damaged" |
| **Inquiry** | Asking a question or seeking information | "What are your business hours?" |
| **Feedback** | Providing an opinion or review (positive or negative) | "I think the UI could be improved" |
| **Request** | Asking for a specific action or change | "Please cancel my subscription" |
| **Escalation** | Demanding higher-level intervention | "I need to speak with a manager" |
| **Other** | Doesn't fit any of the above categories | Spam, unrelated content |

### 2.3 NER Entity Labels

| Label | Definition | Boundary Rules |
|-------|-----------|----------------|
| **PERSON** | Full name of a person | Include first + last name; exclude titles (Mr., Dr.) |
| **ORG** | Organization, company, or institution | Include legal suffixes (Inc., LLC) if present |
| **LOC** | Geographic location (city, country, address) | Include full location name; exclude prepositions |
| **DATE** | Specific date or time reference | Include full date expression ("January 15, 2024") |
| **PRODUCT** | Product name, service, or brand | Include version numbers if part of the name |
| **MONETARY** | Money amounts with currency | Include currency symbol/code |
| **EVENT** | Named event or occurrence | Include full event name |
| **MISC** | Other notable entities | Use sparingly for named entities that don't fit above |

---

## 3. Confidence-Based Review Protocol

### 3.1 Routing Logic

```
┌──────────────────────────────────────────────────────┐
│                   AI Prediction                       │
│              (Confidence Score: 0.0-1.0)             │
└───────────────────────┬──────────────────────────────┘
                        │
                ┌───────┴───────┐
                │               │
        ≥ 0.85 (High)    < 0.85 (Low)
                │               │
    ┌───────────┴──┐    ┌──────┴───────────┐
    │ AUTO-APPROVED │    │  HUMAN REVIEW    │
    │ (Spot Check)  │    │  (Full Review)   │
    └──────────────┘    └──────────────────┘
```

### 3.2 Queue-Specific Instructions

#### Auto-Approved Queue (Spot Check)
- **Time target:** ~10 seconds per record
- **Action:** Quick visual verification of AI labels
- **Accept** if the labels are clearly correct
- **Edit** only if you spot an obvious error
- **Reject** if the text is spam, garbled, or the AI is completely wrong

#### Human Review Queue (Full Review)
- **Time target:** ~60-120 seconds per record
- **Action:** Careful evaluation of text and proposed labels
- Read the full text carefully
- Evaluate the AI's reasoning (click "AI Reasoning" to expand)
- Consider the confidence score context
- Choose Accept, Edit, or Reject with justification

### 3.3 Decision Matrix for Edge Cases

| Scenario | Recommended Action | Example |
|----------|-------------------|---------|
| AI is correct but confidence is low | **Accept** + comment on why you agree | Technical jargon the AI wasn't sure about |
| AI label is close but not ideal | **Edit** to the correct label | "Mixed" → "Negative" when negativity dominates |
| Sarcasm detected that AI missed | **Edit** + note sarcasm in comments | "Oh great, another update 🙄" labeled as positive |
| Text is ambiguous even for humans | **Accept AI** + flag in comments | Could be complaint OR feedback |
| Multiple intents in one text | Choose the **primary** intent + note secondary | Complaint + escalation → pick stronger signal |
| Code-switching (multilingual) | Annotate based on **overall** meaning | Spanish complaint with English product names |
| AI produced invalid entities | **Edit** entities, fix spans | Entity offset doesn't match text |
| No entities to extract | **Accept** empty entity list | Purely abstract text with no named entities |

---

## 4. NER Span Boundary Rules

### 4.1 General Rules

1. **Exact match:** Entity text must exactly match the source text substring at the specified offsets
2. **Minimal spans:** Include only the entity itself, not surrounding context
3. **Complete names:** Include full names — don't split "John Smith" into two entities
4. **No nested entities:** Choose the most specific label when entities overlap

### 4.2 Common Boundary Decisions

| Text | Correct Span | Incorrect Span | Label |
|------|-------------|----------------|-------|
| "CEO of Google Inc." | "Google Inc." | "CEO of Google Inc." | ORG |
| "Dr. Sarah Johnson" | "Sarah Johnson" | "Dr. Sarah Johnson" | PERSON |
| "$1,299.99 USD" | "$1,299.99 USD" | "1,299.99" | MONETARY |
| "on January 15th, 2024" | "January 15th, 2024" | "on January 15th" | DATE |
| "the new iPhone 15 Pro" | "iPhone 15 Pro" | "new iPhone 15 Pro" | PRODUCT |

---

## 5. Quality Standards

### 5.1 Minimum Accuracy Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Annotation accuracy | ≥ 92% | Spot-check by lead annotator |
| Inter-annotator agreement (Cohen's κ) | ≥ 0.75 | Pairwise comparison |
| Average review time (Human Queue) | ≤ 120s | Per record |
| Average review time (Spot Check) | ≤ 15s | Per record |
| Rejection rate (without comments) | 0% | All rejections must be justified |

### 5.2 Comment Requirements

- **Accept:** Comments optional but encouraged for edge cases
- **Edit:** Brief explanation of what was changed and why (REQUIRED)
- **Reject:** Detailed reason for rejection (REQUIRED)

---

## 6. Escalation Protocol

If you encounter any of the following, flag the record and notify the team lead:

1. **Personally identifiable information (PII)** that should be redacted
2. **Offensive or harmful content** that violates content policy
3. **Systematic AI errors** — same mistake pattern across multiple records
4. **Ambiguous guidelines** — scenarios not covered by this SOP
5. **Data quality issues** — truncated text, encoding errors, duplicate records

**Escalation process:**
1. Click **Reject / Flag** on the record
2. Add detailed comment with the escalation reason
3. Tag with `[ESCALATION]` prefix in the comment field
4. Continue with remaining tasks — don't block on escalations

---

## 7. Glossary

| Term | Definition |
|------|-----------|
| **Pre-labeling** | AI-generated initial annotation before human review |
| **Confidence score** | Model's self-assessed certainty (0.0 = no confidence, 1.0 = fully certain) |
| **Routing** | Automated decision to send records to spot-check or full review queues |
| **Cohen's Kappa** | Statistical measure of agreement between 2 annotators, corrected for chance |
| **Fleiss' Kappa** | Extension of Cohen's Kappa for 3+ annotators |
| **BIO tagging** | Named entity tagging scheme: Begin, Inside, Outside |
| **DPO** | Direct Preference Optimization — a method for RLHF fine-tuning |
| **Active Learning** | Iterative process of selecting informative samples for human annotation |

---

*This document is maintained by the SmartAnnotate-AI project team. For questions or suggested updates, please open an issue on the project repository.*
