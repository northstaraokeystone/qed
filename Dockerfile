FROM python:3.11-slim AS builder

# Environment for deterministic, quiet Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies (no recommends to keep image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy metadata first for better layer caching (any that exist)
COPY pyproject.toml* setup.py* setup.cfg* requirements.txt* ./ 

# Install Python dependencies (if requirements.txt is present)
RUN pip install --upgrade pip \
 && if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy full source (qed.py + hooks/)
COPY . .

# ----------------------------------------------------------------------------- 

FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QED_VERSION=5.0.0

# Minimal runtime OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python artifacts from builder (numpy etc.)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source (qed.py + hooks/*)
COPY --from=builder /app /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash qeduser \
 && chown -R qeduser:qeduser /app
USER qeduser

# Labels for provenance
LABEL org.opencontainers.image.source="https://github.com/xai/qed-v5" \
      org.opencontainers.image.description="QED v5.0 Telemetry Compressor Hooks" \
      org.opencontainers.image.version="5.0.0"

# Healthcheck: verify qed is importable
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import qed, sys; sys.exit(0)"

# Default entrypoint: QED CLI (overrideable for hook-specific runs)
ENTRYPOINT ["python", "-m", "qed"]
CMD ["--help"]
