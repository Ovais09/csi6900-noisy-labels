FROM python:3.11-slim@sha256:bab1b7ef4b450c81002278d035eff85ebe394ae94df904f7a3ba14f7e16e487b

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-docker.lock ./
RUN python -m pip install pip==26.2.1 \
    && python -m pip install -r requirements.txt -c requirements-docker.lock \
    && python -m pip check

ARG VCS_REF=unknown

LABEL org.opencontainers.image.revision=$VCS_REF

ENV CSI6900_GIT_COMMIT=$VCS_REF

COPY . .

RUN mkdir -p datasets/raw datasets/processed results/tables results/figures results/logs

CMD ["python", "-m", "pytest", "-q"]
