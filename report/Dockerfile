FROM python:3.11-slim@sha256:bab1b7ef4b450c81002278d035eff85ebe394ae94df904f7a3ba14f7e16e487b

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        latexmk texlive-latex-base texlive-latex-recommended \
        texlive-fonts-recommended texlive-publishers lmodern poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
CMD ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=report/build", "report/main.tex"]
