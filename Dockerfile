# Stochastic F1 Race Strategist — containerised runtime (NFR-11).
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the package itself (src layout)
COPY pyproject.toml README.md schema.sql ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Config + dashboard + tests
COPY config ./config
COPY dashboard.py ./
COPY tests ./tests

# Non-root user for hygiene
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import f1strategist; import dashboard" || exit 1

# Default: launch the dashboard. Override with e.g. "python -m f1strategist.cli ..."
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
