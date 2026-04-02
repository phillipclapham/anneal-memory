FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI (native installer)
WORKDIR /tmp
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

# Install anneal-memory from PyPI
RUN pip install --no-cache-dir anneal-memory

# Create a workspace
WORKDIR /workspace

# Persist Claude auth + AM data between runs
VOLUME /root/.claude
VOLUME /root/.anneal-memory

ENTRYPOINT ["/bin/bash"]
