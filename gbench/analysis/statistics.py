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

"""Statistical analysis utilities for benchmark results."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute statistical measures for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistical measures
    """
    if not values:
        return {}
    
    arr = np.array(values)
    
    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(values),
    }
    
    # Coefficient of Variation (CV%)
    if stats["mean"] > 0:
        stats["cv_percent"] = (stats["std"] / stats["mean"]) * 100
    else:
        stats["cv_percent"] = 0.0
    
    return stats


def aggregate_benchmark_results(
    results: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, Any]:
    """Aggregate multiple benchmark results with statistics.
    
    Args:
        results: List of individual benchmark result dictionaries
        metrics: List of metric names to aggregate
        
    Returns:
        Aggregated results with statistics for each metric
    """
    if not results:
        return {}
    
    aggregated = {
        "iterations": results,  # Keep raw data
        "num_iterations": len(results),
    }
    
    # Compute statistics for each metric
    for metric in metrics:
        values = []
        for result in results:
            if metric in result and result[metric] is not None:
                values.append(result[metric])
        
        if values:
            stats = compute_statistics(values)
            # Add metric prefix to stat names
            for stat_name, stat_value in stats.items():
                aggregated[f"{metric}_{stat_name}"] = stat_value
    
    return aggregated


def validate_repeatability(
    aggregated: dict[str, Any],
    metric: str,
    max_cv_percent: float = 5.0,
) -> tuple[bool, str]:
    """Validate benchmark repeatability using CV%.
    
    Args:
        aggregated: Aggregated results dictionary
        metric: Metric name to check
        max_cv_percent: Maximum acceptable CV%
        
    Returns:
        Tuple of (is_valid, message)
    """
    cv_key = f"{metric}_cv_percent"
    
    if cv_key not in aggregated:
        return False, f"No CV% data for {metric}"
    
    cv = aggregated[cv_key]
    
    if cv <= max_cv_percent:
        return True, f"✓ CV% = {cv:.2f}% (≤ {max_cv_percent}%)"
    elif cv <= max_cv_percent * 2:
        return (
            True,
            f"⚠ CV% = {cv:.2f}% (acceptable but high)",
        )
    else:
        return (
            False,
            f"✗ CV% = {cv:.2f}% (> {max_cv_percent}%, poor repeatability)",
        )


def format_statistics_summary(
    aggregated: dict[str, Any],
    metric: str,
) -> str:
    """Format statistics summary for a metric.
    
    Args:
        aggregated: Aggregated results dictionary
        metric: Metric name
        
    Returns:
        Formatted summary string
    """
    mean = aggregated.get(f"{metric}_mean")
    std = aggregated.get(f"{metric}_std")
    cv = aggregated.get(f"{metric}_cv_percent")
    min_val = aggregated.get(f"{metric}_min")
    max_val = aggregated.get(f"{metric}_max")
    
    if mean is None:
        return f"{metric}: N/A"
    
    summary = f"{metric}: {mean:.2f} ± {std:.2f}"
    
    if cv is not None:
        summary += f" (CV={cv:.1f}%)"
    
    if min_val is not None and max_val is not None:
        summary += f" [min={min_val:.2f}, max={max_val:.2f}]"
    
    return summary
