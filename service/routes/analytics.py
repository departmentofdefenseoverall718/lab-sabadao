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

from fastapi import APIRouter, HTTPException
from service.utils.storage import get_run_details, list_runs

router = APIRouter()

@router.get("/summary")
def get_analytics_summary():
    # Return a summary of all completed runs
    runs = list_runs()
    summary = []
    for run in runs:
        if run.get("status") != "completed":
            continue
            
        details = get_run_details(run["run_id"])
        if not details:
            continue
            
        results = details.get("results", {})
        quality = results.get("quality")
        if not quality:
            continue
            
        raw_scenarios = quality.get("raw_results", {}).get("scenarios", [])
        scenarios_summary = []
        for s in raw_scenarios:
            scenarios_summary.append({
                "name": s.get("name"),
                "path": s.get("path"),
                "status": s.get("status")
            })

        summary.append({
            "run_id": run["run_id"],
            "created_at": run.get("created_at"),
            "tags": run.get("tags", []),
            "model": (details.get("models", [None])[0] or details.get("model_name") or "Unknown"),
            "pass_rate": quality.get("pass_rate", 0.0) / 100.0,
            "total_scenarios": quality.get("total_scenarios", 0),
            "scenarios": scenarios_summary
        })
        
    return summary

@router.get("/{run_id}")
def get_run_analytics(run_id: str):
    details = get_run_details(run_id)
    if not details:
        raise HTTPException(status_code=404, detail="Run not found")
        
    results = details.get("results", {})
    quality = results.get("quality")
    if not quality:
        # If quality is missing, it might still be running or it failed, or it was performance only.
        return {
            "run_id": run_id,
            "status": details.get("status"),
            "tags": details.get("tags", []),
            "metrics": {},
            "scenarios": []
        }
        
    # Parse quality.json data into a chart-friendly format
    metrics = {
        "pass_rate": quality.get("pass_rate", 0.0) / 100.0,
        "total_scenarios": quality.get("total_scenarios", 0),
        "passed_scenarios": quality.get("passed_scenarios", 0)
    }
    
    scenarios = []
    raw_scenarios = quality.get("raw_results", {}).get("scenarios", [])
    for sc in raw_scenarios:
        status = sc.get("status", "unknown")
        scenarios.append({
            "category": "General",
            "name": sc.get("name"),
            "status": status,
            "score": 1.0 if status == "pass" else 0.0,
            "duration": 0
        })
            
    return {
        "run_id": run_id,
        "status": details.get("status"),
        "tags": details.get("tags", []),
        "model": (details.get("models", [None])[0] or details.get("model_name") or "Unknown"),
        "metrics": metrics,
        "scenarios": scenarios
    }
