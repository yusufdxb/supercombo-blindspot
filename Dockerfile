# Reproducible test image for supercombo-blindspot.
#
# Mirrors the CI job exactly: Python 3.10 + requirements-ci.txt + pytest.
# The unit-test closure is CPU-only (numpy / OpenCV / sklearn); the CARLA
# integration suite skips itself when no simulator is reachable, and the
# onnxruntime-GPU inference paths are exercised separately on a GPU host.
#
#   docker build -t supercombo-blindspot .
#   docker run --rm supercombo-blindspot          # runs the test suite
#
FROM python:3.10-slim

# OpenCV (headless) still links libglib at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer caches across source edits.
COPY requirements-ci.txt .
RUN pip install --no-cache-dir -r requirements-ci.txt

COPY . .

# Default: reproduce the CI test run.
CMD ["python", "-m", "pytest", "-q"]
