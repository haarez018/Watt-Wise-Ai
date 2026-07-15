# Build context must be the repo root, not backend/ — requirements.txt
# installs `-e ../libs/wattwise_tariffs` and `-e ../libs/wattwise_climate`
# (see docs/ARCHITECTURE.md's "ML pipeline" section), which needs those
# directories present in the build context alongside backend/. Build with:
#   docker build -f backend/Dockerfile .
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
COPY libs/ /libs/
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
