# Runs the checks in docs/REPRODUCE.md (Levels 1 and 2) on Linux.
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir --editable .
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src
CMD ["bash", "scripts/docker_check.sh"]
