# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Endpoint verification utilities for checking liveness and multimodal support."""

import json
import logging
import urllib.request
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

TINY_1X1_PNG_BASE64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def verify_endpoint_functional(base_url: str, timeout: int = 10) -> Tuple[bool, str, Optional[int]]:
    """Check if an OpenAI/vLLM endpoint is functional and answering requests.

    Args:
        base_url: Base URL of the endpoint (e.g. 'http://127.0.0.1:8000' or 'http://127.0.0.1:8000/v1')
        timeout: Timeout in seconds for the verification requests

    Returns:
        (is_functional, status_msg, max_model_len)
    """
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        models_url = f"{url}/models"
    else:
        models_url = f"{url}/v1/models"

    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                max_len = None
                if models and isinstance(models, list) and len(models) > 0 and "max_model_len" in models[0]:
                    max_len = int(models[0]["max_model_len"])
                return True, "Endpoint is functional and answering /v1/models requests", max_len
    except Exception as e1:
        # Fallback check /health
        try:
            health_url = f"{url}/health" if not url.endswith("/v1") else f"{url[:-3]}/health"
            with urllib.request.urlopen(health_url, timeout=timeout) as resp:
                if resp.status == 200:
                    return True, "Endpoint is functional (answered /health)", None
        except Exception as e2:
            return False, f"Unreachable (/v1/models: {e1}; /health: {e2})", None
    return False, "Endpoint returned unexpected response format", None


def probe_multimodal_support(base_url: str, model_id: str, timeout: int = 10) -> bool:
    """Dynamically probe if an endpoint/model supports multimodal image inputs.

    Sends a 1x1 pixel Base64 PNG request to /v1/chat/completions with max_tokens=1.
    If the endpoint responds HTTP 200 OK, the model supports multimodal.

    Args:
        base_url: Base URL of the HTTP endpoint.
        model_id: Model ID or tag name.
        timeout: Timeout in seconds.

    Returns:
        True if the endpoint accepts image payloads for this model, False otherwise.
    """
    url = base_url.rstrip("/")
    chat_url = f"{url}/chat/completions" if url.endswith("/v1") else f"{url}/v1/chat/completions"

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "test"},
                    {"type": "image_url", "image_url": {"url": TINY_1X1_PNG_BASE64}},
                ],
            }
        ],
        "max_tokens": 1,
    }

    try:
        req = urllib.request.Request(
            chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception as e:
        logger.info(f"Multimodal probe for {model_id} on {chat_url} returned: {e}")
        return False
