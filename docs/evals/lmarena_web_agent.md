# WebArena / LMArena Web Agent Evaluation Setup Guide

## Overview
`lmarena_web_agent` evaluates autonomous agents browsing live interactive web applications (e-commerce storefronts, social platforms, CMS dashboards, ReactJS apps) via Playwright headless browser control.

## Prerequisites
WebArena requires headless Chromium and the Playwright browser automation framework.

### 1. System Requirements
- **Node.js / Chromium dependencies**
- **Python 3.10+**

### 2. Python & Browser Dependencies
Install the required packages and download browser binaries:
```bash
pip install playwright webarena
playwright install --with-deps chromium
```

---

## Running the Evaluation in gbench

### Full Benchmark Run
```bash
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals lmarena_web_agent \
       --eval-thinking \
       --sandboxes 32
```

### Fast Smoke Test (`--eval-limit`)
```bash
# Smoke test on 1 web task
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals lmarena_web_agent \
       --eval-thinking \
       --eval-limit 1
```

### Concurrency Separation (`--sandboxes` vs `--batch-sizes`)
- `--batch-sizes 128`: Used for standard HTTP API requests.
- `--sandboxes 32`: Controls concurrent Playwright headless browser contexts.
