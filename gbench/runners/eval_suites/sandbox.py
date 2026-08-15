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

"""Process isolation for eval suites that execute model-written code (audit CC5).

`--sandboxes` only bounds how many of these run at once; it does not isolate anything.
Until now every code-executing suite ran `sys.executable -c <model output>` directly on the
host, with the user's full filesystem and network.

This wraps those calls in `bubblewrap`: read-only root, private `/tmp`, no network, and an
explicit writable bind for the scratch directory the suite needs.

    GBENCH_SANDBOX=auto   (default) use bwrap when it is present and works, else warn
    GBENCH_SANDBOX=bwrap            require bwrap; raise if it is unavailable
    GBENCH_SANDBOX=none             run directly on the host (previous behaviour)

`auto` self-probes once rather than assuming: a bwrap that is installed but blocked (no
user namespaces, a restrictive container) would otherwise make every execution suite fail
and read as a model that cannot write code.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["sandbox_mode", "sandbox_available", "wrap_argv", "run_sandboxed"]

_PROBE_TIMEOUT = 10


def sandbox_mode() -> str:
    mode = (os.environ.get("GBENCH_SANDBOX") or "auto").strip().lower()
    return mode if mode in ("auto", "bwrap", "none") else "auto"


@functools.lru_cache(maxsize=1)
def _probe_bwrap() -> bool:
    """Does bwrap actually run here? Cached: this is asked once per code-executing sample."""
    if not shutil.which("bwrap"):
        return False
    try:
        proc = subprocess.run(
            wrap_argv([sys.executable, "-c", "print(1)"], force=True),
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT)
        return proc.returncode == 0 and proc.stdout.strip() == "1"
    except Exception as e:  # pragma: no cover - environment dependent
        logger.debug("bwrap probe failed: %s", e)
        return False


@functools.lru_cache(maxsize=1)
def sandbox_available() -> bool:
    """True when execution will be isolated; logs once when it will not be."""
    mode = sandbox_mode()
    if mode == "none":
        logger.warning("GBENCH_SANDBOX=none: model-written code runs directly on this host "
                       "with full filesystem and network access.")
        return False
    if _probe_bwrap():
        return True
    if mode == "bwrap":
        raise RuntimeError(
            "GBENCH_SANDBOX=bwrap was requested but bubblewrap is unavailable or blocked "
            "here. Install `bubblewrap` and ensure unprivileged user namespaces are "
            "permitted, or set GBENCH_SANDBOX=none to run unsandboxed.")
    logger.warning("bubblewrap unavailable; code-executing eval suites will run "
                   "UNSANDBOXED on this host. Install `bubblewrap` for isolation, or set "
                   "GBENCH_SANDBOX=none to silence this.")
    return False


def wrap_argv(argv: Sequence[str], writable: Sequence[str] = (), network: bool = False,
              force: bool = False) -> List[str]:
    """Prefix `argv` with a bubblewrap jail, or return it unchanged when disabled.

    `writable` lists directories the command legitimately needs to write (a compiler's
    scratch dir, the temp file holding the program). Everything else is read-only.
    """
    if not force and not sandbox_available():
        return list(argv)

    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
           "--tmpfs", "/tmp", "--die-with-parent", "--new-session"]
    if not network:
        cmd += ["--unshare-net"]
    for path in writable:
        if path and os.path.isdir(path):
            cmd += ["--bind", path, path]
    return cmd + list(argv)


def run_sandboxed(argv: Sequence[str], writable: Sequence[str] = (), network: bool = False,
                  **kwargs: Any) -> "subprocess.CompletedProcess":
    """`subprocess.run` for model-written code, isolated when bubblewrap is available."""
    return subprocess.run(wrap_argv(argv, writable=writable, network=network), **kwargs)
