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

"""Unit tests for the CLI module."""

import argparse
from unittest.mock import patch, MagicMock
import pytest
from gbench.cli import create_parser, main, _split_golden_tasks


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ([], []),
        (["math_canonical"], ["math_canonical"]),
        (["a", "b"], ["a", "b"]),
        (["a,b"], ["a", "b"]),
        (["a, b", "c"], ["a", "b", "c"]),
        (["a,,b", " "], ["a", "b"]),
    ],
)
def test_split_golden_tasks(raw, expected):
    """Comma and space separated task lists both work.

    nargs="+" alone turns "a,b" into one token that matches no task, so
    the natural thing to type would fail with a harness error.
    """
    assert _split_golden_tasks(raw) == expected


def test_parser_stage_to_gcs():
    """Verify --stage-to-gcs argument is parsed correctly."""
    parser = create_parser()
    args = parser.parse_args(["--models", "gemma-4-E4B-it", "--stage-to-gcs", "gs://my-bucket/path"])

    assert args.models == ["gemma-4-E4B-it"]
    assert args.stage_to_gcs == "gs://my-bucket/path"


@patch("gbench.cli.stage_models_to_gcs")
@patch("gbench.cli.get_models_from_args")
@patch("gbench.cli.check_gpu_ready")
def test_main_stage_to_gcs_flow(mock_gpu_ready, mock_get_models, mock_stage):
    """Verify main() runs the staging flow and bypasses GPU checks."""
    mock_model = MagicMock()
    mock_get_models.return_value = [mock_model]

    # Run main with --stage-to-gcs
    retval = main(["--models", "some-model", "--stage-to-gcs", "gs://my-bucket/path"])

    assert retval == 0
    # check_gpu_ready should NOT be called
    mock_gpu_ready.assert_not_called()
    # stage_models_to_gcs should be called
    mock_stage.assert_called_once_with([mock_model], "gs://my-bucket/path")


def test_parser_max_output_tokens():
    """Verify --max-output-tokens is parsed and populated into config."""
    from gbench.cli import get_config_from_args
    parser = create_parser()
    
    # 1. Custom override
    args = parser.parse_args(["--models", "some-model", "--max-output-tokens", "4096"])
    cfg = get_config_from_args(args)
    assert cfg.eval_max_output_tokens == 4096

    # 2. None default
    args_default = parser.parse_args(["--models", "some-model"])
    cfg_default = get_config_from_args(args_default)
    assert cfg_default.eval_max_output_tokens is None

