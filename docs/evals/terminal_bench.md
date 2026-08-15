# TerminalBench 2.1 (TB2) Evaluation Setup Guide

## Overview
`terminal_bench` evaluates autonomous agent execution across interactive Linux terminal troubleshooting tasks, Bash scripts, and CLI workflows.

## Prerequisites
Terminal-Bench executes candidate commands inside isolated Docker sandbox containers managed by the **Harbor** agent evaluation framework.

---

### 1. System Requirements (Docker Engine)
`pip install docker` only installs the Python SDK. The actual **Docker Engine daemon** must be installed on your Linux host:

```bash
# 1. Check if Docker is already installed
docker --version && docker info

# 2. If Docker is NOT installed, install Docker Engine:
# Option A: Standard Ubuntu/Debian package (zero config)
sudo apt-get update && sudo apt-get install -y docker.io
# Option B: Upstream Docker CE (if preferred)
# Follow https://docs.docker.com/engine/install/

# 3. Ensure Docker service is running
sudo systemctl enable --now docker

# 4. Grant non-root user permissions to access /var/run/docker.sock
sudo groupadd -f docker
sudo usermod -aG docker $USER
newgrp docker
```

---

### 2. Python & Framework Dependencies
Install the official Docker Python SDK and Harbor framework:

```bash
# Via pip
pip install docker harbor

# Or via uv (recommended)
uv tool install harbor
```

---

### 3. Install Terminal-Bench 2.1 (TB2.1) Dataset
Download TB2.1 tasks directly from the [Harbor Registry](https://hub.harborframework.com/datasets):

```bash
harbor dataset download terminal-bench/terminal-bench-2-1
```

---

## Running the Evaluation in gbench

### Full Benchmark Run (All Tasks)
```bash
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals terminal_bench \
       --eval-thinking \
       --sandboxes 32
```

### Fast Smoke Test (`--eval-limit`)
Run a quick single-task or small-sample smoke test to verify setup:

```bash
# Smoke test on 1 task
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals terminal_bench \
       --eval-thinking \
       --eval-limit 1

# Quick test on 5 tasks with 16 parallel sandbox containers
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals terminal_bench \
       --eval-thinking \
       --eval-limit 5 \
       --sandboxes 16
```

### Concurrency Separation (`--sandboxes` vs `--batch-sizes`)
When running mixed evaluation campaigns (e.g. `--evals all` or combining academic QA evals with agent benchmarks):
- `--batch-sizes 128` controls concurrent HTTP API requests for high-throughput academic benchmarks (`mmlu_pro`, `aime`, `gsm8k`).
- `--sandboxes 32` or `--sandboxes 64` explicitly caps the number of simultaneous Docker containers spawned on the host system to prevent container daemon exhaustion while maintaining maximum throughput.
