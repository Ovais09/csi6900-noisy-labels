FROM python:3.11-slim

ARG VCS_REF=unknown

LABEL org.opencontainers.image.revision=$VCS_REF

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    CSI6900_GIT_COMMIT=$VCS_REF

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

RUN mkdir -p datasets/raw datasets/processed results/tables results/figures results/logs

CMD ["python", "-m", "pytest", "-q"]
