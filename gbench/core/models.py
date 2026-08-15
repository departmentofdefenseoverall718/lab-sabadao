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

"""Model registry with auto-enrichment from HuggingFace config.json.

Models are defined as slim entries (short_name, hf_model_id) with optional
overrides. At registry init, each model's config.json is fetched from the
HuggingFace cache to derive: total params, MoE topology, multimodal support,
category, and max context length.

Local paths follow a convention: models/gbench/<short_name> for third-party
models, models/<short_name> for Gemma4 models.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional

from huggingface_hub import hf_hub_download, model_info

logger = logging.getLogger(__name__)

# Workspace root (parent of gbench repo)
# models.py is at gbench/gbench/core/models.py → core/ → gbench/ → gbench/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Get models directory - can be overridden with GEMMA_MODELS_DIR env var
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR = Path(os.getenv("GEMMA_MODELS_DIR", DEFAULT_MODELS_DIR))




def is_tfhub_path(path: str) -> bool:
    """Check if the path is a TFHub path."""
    return path.startswith("/tfhub/") or path.startswith("tfhub://")


def normalize_tfhub_path(path: str) -> str:
    """Normalize tfhub:// paths to /tfhub/prod/ paths."""
    if path.startswith("tfhub://"):
        parts = path[8:].strip("/").split("/")
        if len(parts) >= 3:
            publisher = parts[0]
            asset = parts[1]
            version = parts[2]
            return f"/tfhub/prod/{publisher}/{asset}/{version}"
        elif len(parts) == 2:
            return f"/tfhub/prod/{parts[0]}/{parts[1]}/1"
    elif path.startswith("/tfhub/"):
        return path
    raise ValueError(f"Not a valid TFHub path: {path}")


def tfhub_path_to_short_name(path: str) -> str:
    """Extract a clean short name from a TFHub path.

    E.g. /tfhub/prod/ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1
    -> ml-gemma-GEMMA-4.0-2B-IT-G755-SAFETENSORS-v1
    """
    normalized = normalize_tfhub_path(path)
    parts = normalized.strip("/").split("/")
    if len(parts) >= 5:
        publisher = parts[2]
        asset = parts[3]
        version = parts[4]
        return f"{publisher}-{asset}-v{version}"
    return parts[-1]


def _clean_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def download_from_tfhub(tfhub_id: str, target_dir: Path) -> None:
    """Download entire model directory from TFHub using fileutil cp -R."""
    normalized_src = normalize_tfhub_path(tfhub_id)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    _clean_path(target_dir)

    logger.info(f"Running: fileutil cp -R {normalized_src} {target_dir}")
    try:
        subprocess.run(
            ["/google/data/ro/teams/tf-hub/fileutil", "cp", "-R", normalized_src, str(target_dir)],
            check=True,
            timeout=600
        )
        logger.info(f"Successfully downloaded {tfhub_id} to {target_dir}")
    except subprocess.TimeoutExpired as e:
        logger.error(f"Download timed out after 600 seconds: {e}")
        _clean_path(target_dir)
        raise RuntimeError(f"Failed to download model from TFHub: timed out") from e
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to download model from TFHub: {e}")
        _clean_path(target_dir)
        raise RuntimeError(f"Failed to download model from TFHub: {e}")


class ModelFormat(Enum):
    """Supported model formats."""

    HF = "hf"
    GGUF = "gguf"
    REMOTE = "remote-endpoint"


class ModelCategory(Enum):
    """Model categories."""

    TEXT = "text"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"


class Priority(Enum):
    """Benchmark priority levels."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


@dataclass
class ModelConfig:
    """Configuration for a specific model variant.

    Core fields (short_name, hf_model_id) are required. Everything else
    is either auto-derived from config.json or uses sensible defaults.
    """

    # ── Required ─────────────────────────────────────────────
    short_name: str
    hf_model_id: str

    # ── Auto-derived from config.json (populated by _enrich) ─
    total_params_b: float = 0.0
    is_moe: bool = False
    num_experts: int = 0
    num_active_experts: int = 0
    supports_multimodal: bool = False
    category: ModelCategory = ModelCategory.TEXT
    max_context_length: int = 128_000

    # ── Optional overrides ───────────────────────────────────
    name: str = ""  # Human-readable name (defaults to short_name)
    local_path: Optional[str] = None
    gguf_model_id: Optional[str] = None
    gguf_file: Optional[str] = None
    priority: Priority = Priority.P0
    supports_audio: bool = False

    def __post_init__(self):
        if not self.name:
            self.name = self.short_name

    def get_model_path(self, format: ModelFormat) -> str:
        """Get model path for specified format.

        For HF models, returns local_path (resolved to absolute) if set,
        otherwise HuggingFace Hub model ID.
        For GGUF models, returns absolute path to local file.
        """
        if format == ModelFormat.REMOTE:
            return self.hf_model_id
        elif format == ModelFormat.HF:
            if self.local_path:
                resolved = PROJECT_ROOT / self.local_path
                if resolved.exists():
                    return str(resolved)
                if is_tfhub_path(self.hf_model_id):
                    logger.info(f"Model not found locally at {resolved}. Downloading from TFHub...")
                    download_from_tfhub(self.hf_model_id, resolved)
                    return str(resolved)
            return self.hf_model_id
        elif format == ModelFormat.GGUF:
            if not self.gguf_file:
                raise ValueError(
                    f"GGUF format not available for {self.short_name}"
                )

            gguf_path = MODELS_DIR / self.gguf_file

            if not gguf_path.exists():
                raise FileNotFoundError(
                    f"GGUF model file not found: {gguf_path}\n\n"
                    f"Download it with:\n"
                    f"  ./utils/download_gemma.sh {self.short_name}\n\n"
                    f"Or set GEMMA_MODELS_DIR environment variable.\n"
                    f"Current GEMMA_MODELS_DIR: {MODELS_DIR}"
                )

            return str(gguf_path)
        raise ValueError(f"Unknown format: {format}")


# ── Auto-enrichment from HuggingFace config.json ────────────

def _enrich_from_config(model: ModelConfig) -> None:
    """Populate auto-derived fields from cached HuggingFace config.json.

    Uses huggingface_hub to locate the cached config.json. If not cached,
    downloads it (~1KB). Sets: total_params_b, is_moe, num_experts,
    num_active_experts, supports_multimodal, category, max_context_length.
    """
    try:
        if model.hf_model_id.startswith("gs://"):
            temp_dir = tempfile.mkdtemp(prefix="gbench_gcs_config_")
            config_path = os.path.join(temp_dir, "config.json")
            gcs_url = model.hf_model_id.rstrip("/") + "/config.json"
            try:
                subprocess.run(
                    ["gcloud", "storage", "cp", gcs_url, config_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                with open(config_path) as f:
                    config = json.load(f)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        elif is_tfhub_path(model.hf_model_id):
            normalized_path = normalize_tfhub_path(model.hf_model_id)
            config_src = f"{normalized_path}/config.json"
            temp_dir = tempfile.mkdtemp(prefix="gbench_tfhub_config_")
            config_path = os.path.join(temp_dir, "config.json")
            try:
                subprocess.run(
                    ["cp", "-f", config_src, config_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                with open(config_path) as f:
                    config = json.load(f)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        elif ":" in model.hf_model_id or "/" not in model.hf_model_id:
            # Model ID is an Ollama tag (e.g. gemma4-qat:4b) or short name, not an HF repo ID
            m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model.hf_model_id)
            if m:
                model.total_params_b = float(m.group(1))
            return
        else:
            try:
                # First try local cache only to avoid network spam/401s on gated models
                path = hf_hub_download(model.hf_model_id, "config.json", local_files_only=True)
            except Exception:
                try:
                    path = hf_hub_download(model.hf_model_id, "config.json")
                except Exception as dl_err:
                    logger.debug(f"Could not fetch config.json for {model.hf_model_id}: {dl_err}")
                    return
            with open(path) as f:
                config = json.load(f)
    except Exception as e:
        logger.debug(
            f"Could not fetch config.json for {model.hf_model_id}: {e}. "
            f"Using defaults."
        )
        return

    tc = config.get("text_config", config)

    # ── Architecture fields ──────────────────────────────────
    hidden = tc.get("hidden_size", 0)
    layers = tc.get("num_hidden_layers", 0)
    heads = tc.get("num_attention_heads", 0)
    kv_heads = tc.get("num_key_value_heads", heads)
    intermediate = tc.get("intermediate_size", 0)
    vocab = tc.get("vocab_size", config.get("vocab_size", 0))

    # MoE detection (field names vary across vendors)
    num_experts = (
        tc.get("num_experts")
        or tc.get("num_local_experts")
        or tc.get("n_routed_experts")
        or 0
    )
    num_active = (
        tc.get("num_experts_per_tok")
        or tc.get("top_k_experts")
        or 0
    )
    moe_intermediate = tc.get("moe_intermediate_size", 0)
    n_shared = tc.get("n_shared_experts", 0)
    is_moe = num_experts > 0

    model.is_moe = is_moe
    model.num_experts = num_experts
    model.num_active_experts = num_active

    # ── Multimodal detection ─────────────────────────────────
    arch_str = str(config.get("architectures", []))
    has_vision = (
        "vision_config" in config
        or "ConditionalGeneration" in arch_str
    )
    model.supports_multimodal = has_vision
    model.category = (
        ModelCategory.MULTIMODAL if has_vision else ModelCategory.TEXT
    )

    # ── Max context length ───────────────────────────────────
    max_pos = tc.get(
        "max_position_embeddings",
        config.get("max_position_embeddings", 128_000),
    )
    model.max_context_length = max_pos

    # ── Total params estimation ──────────────────────────────
    # Prefer safetensors metadata (exact), fall back to architecture estimate.
    st_total = None
    if not model.hf_model_id.startswith("gs://") and not is_tfhub_path(model.hf_model_id):
        try:
            info = model_info(model.hf_model_id)
            if info.safetensors and info.safetensors.total:
                st_total = info.safetensors.total / 1e9
        except Exception:
            pass

    if st_total:
        model.total_params_b = round(st_total, 1)
    else:
        # Estimate from architecture
        head_dim = hidden // heads if heads else 0
        embed_params = vocab * hidden
        attn_per_layer = (
            hidden * heads * head_dim
            + hidden * kv_heads * head_dim * 2
            + heads * head_dim * hidden
        )
        if is_moe:
            expert_inter = moe_intermediate if moe_intermediate else intermediate
            expert_ffn = 3 * hidden * expert_inter
            ffn_per_layer = num_experts * expert_ffn + n_shared * expert_ffn
        else:
            ffn_per_layer = 3 * hidden * intermediate

        est = (embed_params + layers * (attn_per_layer + ffn_per_layer)) / 1e9
        if est > 0:
            model.total_params_b = round(est, 1)
        else:
            name_to_search = f"{model.short_name} {model.hf_model_id or ''} {model.name or ''}"
            m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", name_to_search)
            if m:
                model.total_params_b = float(m.group(1))
            else:
                model.total_params_b = 0.0

    # ── Name fallback ────────────────────────────────────────
    if not model.name:
        model.name = model.short_name


def _hf_id_to_short_name(hf_model_id: str) -> str:
    """Derive a model ID from a HuggingFace model ID or TFHub path.

    Preserves original casing from HuggingFace.
    E.g. 'google/gemma-4-E4B-it'  → 'gemma-4-E4B-it'
         'google/gemma-4-31B-it'  → 'gemma-4-31B-it'
    """
    if is_tfhub_path(hf_model_id):
        return tfhub_path_to_short_name(hf_model_id)
    return hf_model_id.rstrip("/").split("/")[-1]


# ── Slim model definitions ──────────────────────────────────
# Only hf_model_id is required. short_name is derived automatically.
# Optional GGUF fields for models that support quantized formats.

_MODEL_DEFS: list[dict] = [
    # ── Gemma4 ───────────────────────────────────────────────
    dict(
        hf_model_id="google/gemma-4-E2B-it",
        total_params_b=2.0,
        max_context_length=128_000,
        category=ModelCategory.TEXT,
    ),
    dict(
        hf_model_id="google/gemma-4-E4B-it",
        total_params_b=4.0,
        max_context_length=128_000,
        category=ModelCategory.TEXT,
    ),
    dict(
        hf_model_id="google/gemma-4-26B-A4B-it",
        total_params_b=26.0,
        max_context_length=128_000,
        category=ModelCategory.TEXT,
        is_moe=True,
        num_experts=16,
        num_active_experts=2,
    ),
    dict(
        hf_model_id="google/gemma-4-31B-it",
        total_params_b=31.0,
        max_context_length=128_000,
        category=ModelCategory.TEXT,
    ),
]


def _build_models() -> list[ModelConfig]:
    """Build ModelConfig list from slim definitions + HF enrichment."""
    models = []
    for defn in _MODEL_DEFS:
        hf_id = defn["hf_model_id"]
        short = _hf_id_to_short_name(hf_id)

        model = ModelConfig(
            short_name=short,
            local_path=f"models/{short}",
            **defn,
        )
        if not model.total_params_b:
            _enrich_from_config(model)
        models.append(model)
    return models


# Build on import — config.json files are tiny and cached by huggingface_hub
MODELS = _build_models()


class ModelRegistry:
    """Registry for accessing model configurations.

    Built-in models (Gemma3/4) are pre-loaded at import time.
    Any other HuggingFace model can be dynamically registered via
    ``register_hf_model()`` or by passing an HF model ID to ``get()``.
    """

    def __init__(self):
        """Initialize the model registry.

        Lookups are case-insensitive — 'gemma-4-E2B-it' and 'gemma-4-e2b-it'
        both resolve to the same model.
        """
        self._models: dict[str, ModelConfig] = {}
        for model in MODELS:
            self._models[model.short_name.lower()] = model
        self._by_category: dict[ModelCategory, list[ModelConfig]] = {}
        for model in MODELS:
            self._by_category.setdefault(model.category, []).append(model)

    def register_hf_model(self, hf_model_id: str) -> ModelConfig:
        """Dynamically register a model from its HuggingFace model ID.

        Creates a ModelConfig with auto-derived fields from config.json.
        If already registered, returns the existing entry.

        Args:
            hf_model_id: HuggingFace model ID (e.g. 'google/gemma-4-31b-it').

        Returns:
            The registered ModelConfig.
        """
        # Check if already registered (by HF ID)
        for m in self._models.values():
            if m.hf_model_id == hf_model_id:
                return m

        short = _hf_id_to_short_name(hf_model_id)
        key = short.lower()

        # Avoid short_name collision
        if key in self._models:
            return self._models[key]

        model = ModelConfig(
            short_name=short,
            hf_model_id=hf_model_id,
            local_path=f"models/gbench/{short}",
        )
        _enrich_from_config(model)

        self._models[key] = model
        MODELS.append(model)
        self._by_category.setdefault(model.category, []).append(model)

        logger.info(
            f"Registered {short} from {hf_model_id} "
            f"({model.total_params_b:.1f}B, "
            f"{'MoE' if model.is_moe else 'dense'}, "
            f"{'multimodal' if model.supports_multimodal else 'text'})"
        )
        return model

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get model by short name (case-insensitive) or HuggingFace model ID.

        If not found and the name contains '/', automatically registers
        it as a new model from HuggingFace.
        """
        model = self._models.get(name.lower())
        if model is not None:
            return model

        # Auto-register if it looks like an HF model ID
        if "/" in name:
            return self.register_hf_model(name)

        return None

    def get_by_category(
        self, category: ModelCategory
    ) -> list[ModelConfig]:
        """Get all models in a category."""
        return self._by_category.get(category, [])

    def get_by_priority(self, priority: Priority) -> list[ModelConfig]:
        """Get all models at a priority level."""
        return [m for m in self._models.values() if m.priority == priority]

    def list_all(self) -> list[ModelConfig]:
        """Get all registered models."""
        return list(self._models.values())

    def filter(
        self,
        category: Optional[ModelCategory] = None,
        priority: Optional[Priority] = None,
        supports_gguf: bool = False,
    ) -> list[ModelConfig]:
        """Filter models by criteria."""
        filtered = list(self._models.values())

        if category is not None:
            filtered = [m for m in filtered if m.category == category]

        if priority is not None:
            filtered = [m for m in filtered if m.priority == priority]

        if supports_gguf:
            filtered = [
                m for m in filtered if m.gguf_model_id is not None
            ]

        return filtered


# Global registry instance
registry = ModelRegistry()
