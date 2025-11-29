# Use your chosen base (kept 3.10-slim as requested)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
WORKDIR /app

# --- Install system packages needed by Playwright, Tesseract, ffmpeg, fonts, etc. ---
# Keep minimal but include packages you commonly need (ffmpeg + fonts + dev tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl gnupg ca-certificates unzip git \
    # Playwright and headless Chromium dependencies
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
    libgtk-3-0 libgbm1 libasound2 libxcomposite1 libxdamage1 libxrandr2 \
    libxfixes3 libpango-1.0-0 libcairo2 \
    # multimedia and OCR
    ffmpeg tesseract-ocr \
    # fonts to avoid missing-glyph issues in headless browsers and images
    fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# --- Ensure pip tooling is fresh before installing Python packages ---
RUN python -m pip install --upgrade pip setuptools wheel

# --- Install playwright CLI first so it's available during build (keeps behavior similar) ---
# We will also install chromium browsers below.
RUN pip install --no-cache-dir playwright

# --- Install uv package manager (you already use it) ---
RUN pip install --no-cache-dir uv

# --- Copy project into container (do this before dependency sync to include pyproject/uv files) ---
COPY . /app

# --- Install python dependencies via uv (same as your workflow) ---
# --frozen uses the lockfile; this expects pyproject.lock to exist in repo
RUN uv sync --frozen

# --- Install Playwright browsers (must be done AFTER Playwright installed) ---
# Use the playwright CLI to download browser engine(s). --with-deps is useful on Debian-slim.
RUN python -m playwright install --with-deps chromium

# Optional: create runtime directory for file downloads etc.
RUN mkdir -p /app/LLMFiles && chmod -R 755 /app/LLMFiles

# Expose the port used by your FastAPI app
EXPOSE 7860

# --- Runtime command ---
# Keep using uv run main.py but ensure we bind to 0.0.0.0 and correct port for Spaces.
# If your uv config already sets host/port, you can remove the flags.
CMD ["uv", "run", "main.py", "--host", "0.0.0.0", "--port", "7860"]
