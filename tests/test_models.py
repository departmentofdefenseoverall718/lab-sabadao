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

"""Unit tests for model registry and param-based config functions."""

from gbench.core.models import (
    ModelCategory,
    ModelConfig,
    ModelFormat,
    ModelRegistry,
    Priority,
    registry,
    MODELS,
    PROJECT_ROOT,
)
from gbench.core.config import (
    get_batch_sizes,
    get_num_gpus,
    get_server_timeout,
    get_tensor_parallel,
    get_gpu_memory_utilization,
    get_max_model_len,
)


# ── Registry tests ──────────────────────────────────────────

def test_registry_not_empty():
    """Registry contains models."""
    assert len(MODELS) > 0


def test_all_models_accessible():
    """Every model in MODELS is accessible via registry.get()."""
    for model in MODELS:
        found = registry.get(model.short_name)
        assert found is not None, f"Model '{model.short_name}' not found"
        assert found.short_name == model.short_name


def test_unique_short_names():
    """All short names are unique."""
    names = [m.short_name for m in MODELS]
    assert len(names) == len(set(names)), "Duplicate short_name found"


def test_local_paths_set():
    """All models have local_path configured."""
    for model in MODELS:
        assert model.local_path is not None, (
            f"{model.short_name} missing local_path"
        )


def test_hf_model_ids_set():
    """All models have hf_model_id configured."""
    for model in MODELS:
        assert model.hf_model_id, (
            f"{model.short_name} missing hf_model_id"
        )


def test_auto_enrichment_params():
    """All models have total_params_b populated by auto-enrichment."""
    for model in MODELS:
        assert model.total_params_b > 0, (
            f"{model.short_name} has total_params_b={model.total_params_b}"
        )


def test_multimodal_flags_consistent():
    """Models in MULTIMODAL category have supports_multimodal=True."""
    for model in MODELS:
        if model.category == ModelCategory.MULTIMODAL:
            assert model.supports_multimodal, (
                f"{model.short_name} is MULTIMODAL but supports_multimodal=False"
            )


def test_text_models_not_multimodal():
    """Models in TEXT category have supports_multimodal=False."""
    for model in MODELS:
        if model.category == ModelCategory.TEXT:
            assert not model.supports_multimodal, (
                f"{model.short_name} is TEXT but supports_multimodal=True"
            )


def test_filter_by_category():
    """Filtering by category returns correct subsets."""
    text_models = registry.filter(category=ModelCategory.TEXT)
    mm_models = registry.filter(category=ModelCategory.MULTIMODAL)

    for m in text_models:
        assert m.category == ModelCategory.TEXT
    for m in mm_models:
        assert m.category == ModelCategory.MULTIMODAL


def test_filter_gguf_support():
    """GGUF filter only returns models with gguf_model_id."""
    gguf_models = registry.filter(supports_gguf=True)
    for m in gguf_models:
        assert m.gguf_model_id is not None


def test_gemma_models_present():
    """Gemma4 models are registered."""
    gemma4 = [m for m in MODELS if m.short_name.startswith("gemma-4-")]
    assert len(gemma4) > 0, "No Gemma4 models found"


def test_gemma4_paths_not_under_gbench():
    """Gemma4 models use models/gemma4-* not models/gbench/."""
    for model in MODELS:
        if model.short_name.startswith("gemma-4-"):
            assert "gbench" not in model.local_path, (
                f"{model.short_name} should not be under models/gbench/"
            )


# ── Param-based config function tests ───────────────────────

def test_gpu_allocation_fairness():
    """Models of similar size get the same GPU count."""
    # All ≤20B → 1 GPU
    assert get_num_gpus(1.0) == 1
    assert get_num_gpus(14.0) == 1
    assert get_num_gpus(20.0) == 1

    # All 20-80B → 2 GPUs
    assert get_num_gpus(25.0) == 2
    assert get_num_gpus(36.0) == 2
    assert get_num_gpus(80.0) == 2

    # All >80B → 8 GPUs
    assert get_num_gpus(100.0) == 8
    assert get_num_gpus(235.0) == 8


def test_tensor_parallel_matches_gpus():
    """TP size always equals GPU count."""
    for params in [1.0, 14.0, 30.0, 120.0, 235.0]:
        assert get_tensor_parallel(params) == get_num_gpus(params)


def test_batch_sizes_decrease_with_model_size():
    """Larger models get smaller batch sizes."""
    small = get_batch_sizes(4.0, "default")
    large = get_batch_sizes(30.0, "default")
    xlarge = get_batch_sizes(120.0, "default")

    assert max(small) > max(large) > max(xlarge)


def test_timeout_increases_with_params():
    """Larger models get longer timeouts."""
    t_small = get_server_timeout(4.0, False)
    t_large = get_server_timeout(30.0, False)
    t_xlarge = get_server_timeout(120.0, False)

    assert t_small < t_large < t_xlarge


def test_timeout_moe_multiplier():
    """MoE models get longer timeouts than dense at same size."""
    t_dense = get_server_timeout(30.0, False)
    t_moe = get_server_timeout(30.0, True)

    assert t_moe > t_dense


def test_timeout_bounds():
    """Timeouts are within floor/cap bounds."""
    assert get_server_timeout(1.0, False) >= 600
    assert get_server_timeout(500.0, True) <= 3600


def test_max_model_len_uniform():
    """max_model_len is always 4096."""
    assert get_max_model_len() == 4096


def test_gpu_memory_utilization():
    """GPU memory utilization is uniform 0.90."""
    assert get_gpu_memory_utilization(4.0) == 0.90
    assert get_gpu_memory_utilization(120.0) == 0.90


def test_tfhub_path_helpers():
    from gbench.core.models import is_tfhub_path, normalize_tfhub_path, tfhub_path_to_short_name

    assert is_tfhub_path("tfhub://ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1")
    assert is_tfhub_path("/tfhub/prod/ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1")
    assert not is_tfhub_path("google/gemma-3-1b-it")

    assert normalize_tfhub_path("tfhub://ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1") == "/tfhub/prod/ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1"
    assert normalize_tfhub_path("tfhub://ml-gemma-experimental/Gemma4_4p5B_PT/2") == "/tfhub/prod/ml-gemma-experimental/Gemma4_4p5B_PT/2"
    assert normalize_tfhub_path("/tfhub/prod/ml-gemma/GEMMA-4.0-2B") == "/tfhub/prod/ml-gemma/GEMMA-4.0-2B"

    assert tfhub_path_to_short_name("tfhub://ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1") == "ml-gemma-GEMMA-4.0-2B-IT-G755-SAFETENSORS-v1"
    assert tfhub_path_to_short_name("/tfhub/prod/ml-gemma-experimental/Gemma4_4p5B_PT/2") == "ml-gemma-experimental-Gemma4_4p5B_PT-v2"


from unittest.mock import patch, MagicMock
import json

@patch("gbench.core.models.subprocess.run")
def test_tfhub_model_registration(mock_run):
    from gbench.core.models import registry

    def side_effect(args, **kwargs):
        dest_path = args[3]
        mock_config = {
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "vocab_size": 256000,
            },
            "architectures": ["Gemma3ForConditionalGeneration"]
        }
        with open(dest_path, "w") as f:
            json.dump(mock_config, f)
        res = MagicMock()
        res.returncode = 0
        return res

    mock_run.side_effect = side_effect

    tfhub_id = "tfhub://ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1"
    model = registry.register_hf_model(tfhub_id)

    assert model is not None
    assert model.short_name == "ml-gemma-GEMMA-4.0-2B-IT-G755-SAFETENSORS-v1"
    assert model.hf_model_id == tfhub_id
    assert model.local_path == "models/gbench/ml-gemma-GEMMA-4.0-2B-IT-G755-SAFETENSORS-v1"
    assert model.total_params_b > 0

    # Clean up registry
    key = model.short_name.lower()
    if key in registry._models:
        del registry._models[key]
    from gbench.core.models import MODELS
    if model in MODELS:
        MODELS.remove(model)


@patch("gbench.core.models.subprocess.run")
@patch("shutil.rmtree")
def test_download_from_tfhub(mock_rmtree, mock_run):
    from gbench.core.models import download_from_tfhub
    from pathlib import Path

    tfhub_id = "tfhub://ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1"
    target_path = Path("/tmp/gbench_test_download")

    download_from_tfhub(tfhub_id, target_path)

    mock_run.assert_called_once_with(
        ["/google/data/ro/teams/tf-hub/fileutil", "cp", "-R", "/tfhub/prod/ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1", "/tmp/gbench_test_download"],
        check=True,
        timeout=600
    )


@patch("gbench.core.models.subprocess.run")
def test_download_from_tfhub_failure(mock_run):
    from gbench.core.models import download_from_tfhub
    from pathlib import Path
    import subprocess
    import pytest

    mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="fileutil")

    tfhub_id = "tfhub://ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1"
    target_path = Path("/tmp/gbench_test_download_fail")

    with pytest.raises(RuntimeError, match="Failed to download model from TFHub"):
        download_from_tfhub(tfhub_id, target_path)


@patch("gbench.core.models.download_from_tfhub")
def test_get_model_path_tfhub_download(mock_download_from_tfhub):
    from gbench.core.models import ModelConfig, ModelFormat
    from pathlib import Path

    tfhub_id = "tfhub://ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1"
    model = ModelConfig(
        short_name="test-tfhub-model",
        hf_model_id=tfhub_id,
        local_path="models/nonexistent_test_tfhub_path",
    )

    from gbench.core.models import PROJECT_ROOT
    expected_path = PROJECT_ROOT / model.local_path
    assert not expected_path.exists(), "Expected path should not exist before test"

    path = model.get_model_path(ModelFormat.HF)

    mock_download_from_tfhub.assert_called_once_with(tfhub_id, expected_path)
    assert path == str(expected_path)


@patch("gbench.core.models.download_from_tfhub")
def test_get_model_path_tfhub_no_redownload(mock_download):
    from gbench.core.models import ModelConfig, ModelFormat
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        tfhub_id = "tfhub://ml-gemma/GEMMA-4.0-2B-IT-G755-SAFETENSORS/1"
        model = ModelConfig(
            short_name="test-tfhub-model",
            hf_model_id=tfhub_id,
            local_path=temp_dir,
        )

        path = model.get_model_path(ModelFormat.HF)

        mock_download.assert_not_called()
        assert path == temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

