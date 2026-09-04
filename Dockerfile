# ──────────────────────────────────────────────
# SmartAnnotate-AI — Docker Configuration
# ──────────────────────────────────────────────
# Multi-stage build for production deployment
# ──────────────────────────────────────────────

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# ── Dependencies Stage ───────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Stage ────────────────────────
FROM deps AS app

# Copy application code
COPY schemas.py pipeline.py metrics.py exporters.py app.py ./
COPY data/ ./data/

# Create export directory
RUN mkdir -p data/exports && chown -R appuser:appuser /app

# Streamlit configuration
RUN mkdir -p /home/appuser/.streamlit
RUN echo '\
[server]\n\
port = 8501\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base = "dark"\n\
primaryColor = "#818cf8"\n\
backgroundColor = "#0f0f23"\n\
secondaryBackgroundColor = "#1a1a3e"\n\
textColor = "#e2e8f0"\n\
' > /home/appuser/.streamlit/config.toml

RUN chown -R appuser:appuser /home/appuser/.streamlit

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
ENTRYPOINT ["streamlit", "run", "app.py"]
