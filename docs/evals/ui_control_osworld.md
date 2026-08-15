# OSWorld Desktop UI Control Evaluation Setup Guide

## Overview
`ui_control_osworld` evaluates multimodal computer-use agents performing desktop tasks (Ubuntu OS, LibreOffice, Chrome, VSCode, GIMP) using screenshot observations and OS action primitives (click, drag, type, hotkey).

## Prerequisites
OSWorld requires a virtual machine or container running an X11 virtual display buffer (Xvfb) and GUI automation tools.

---

### 1. System Requirements (Docker Engine & Xvfb)
`pip install docker` only installs the Python SDK. The actual **Docker Engine daemon** and virtual display buffer must be configured on your host:

```bash
# 1. Check if Docker is already installed
docker --version && docker info

# 2. If Docker is NOT installed, install Docker Engine:
# Option A: Standard Ubuntu/Debian package (zero config)
sudo apt-get update && sudo apt-get install -y docker.io xvfb
# Option B: Upstream Docker CE (if preferred)
# Follow https://docs.docker.com/engine/install/

# 3. Ensure Docker service is running
sudo systemctl enable --now docker

# 4. Grant non-root user permissions to access /var/run/docker.sock
sudo groupadd -f docker
sudo usermod -aG docker $USER
newgrp docker

# 5. Start virtual display buffer (if running headless)
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
```

---

### 2. Python Dependencies
Install the required packages:
```bash
pip install docker pyautogui pillow osworld
```

---

## Running the Evaluation in gbench

### Full Benchmark Run
```bash
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals ui_control_osworld \
       --eval-thinking \
       --sandboxes 16
```

### Fast Smoke Test (`--eval-limit`)
```bash
# Smoke test on 1 GUI task
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals ui_control_osworld \
       --eval-thinking \
       --eval-limit 1
```

### Concurrency Separation (`--sandboxes` vs `--batch-sizes`)
- `--batch-sizes 128`: Used for standard HTTP API requests.
- `--sandboxes 16`: Controls simultaneous X11 / OSWorld desktop instances.
