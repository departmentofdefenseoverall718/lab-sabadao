# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Helper module to robustly load and parse gbench datasets from JSONL/CSV/Textproto formats.

import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Allow large CSV fields (MRCR 128k text)
try:
    csv.field_size_limit(sys.maxsize)
except Exception:
    csv.field_size_limit(2147483647)


def extract_lossless_image_b64(img_val: Any) -> Optional[str]:
    """Extract base64 string from raw image bytes or PIL object losslessly without re-compression overhead."""
    import base64
    import io
    if img_val is None:
        return None
    if isinstance(img_val, bytes):
        return base64.b64encode(img_val).decode("utf-8")
    if isinstance(img_val, dict) and "bytes" in img_val and img_val["bytes"]:
        return base64.b64encode(img_val["bytes"]).decode("utf-8")
    if hasattr(img_val, "filename") and img_val.filename:
        try:
            with open(img_val.filename, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass
    if hasattr(img_val, "save"):
        im = img_val
        if hasattr(im, "mode") and im.mode not in ("RGB", "L", "RGBA"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        fmt = getattr(im, "format", None) or "PNG"
        try:
            im.save(buf, format=fmt)
        except Exception:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    return None
