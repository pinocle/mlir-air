# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# From Triton-XDNA amd_triton_npu/backend/config.py, changed for AIE1.

"""
Process-global configuration for the AIE1 Triton backend (from Triton-XDNA).

Provides a Python API to control settings that were previously only
available via environment variables. Programmatic values always take
priority over environment variables, which in turn take priority over
built-in defaults.

Usage::

    from aiec_air.triton.config import aie_config, config_context

    aie_config.compile_only = True

    with config_context(compile_only=True):
        kernel[grid](a, b, c)

    # Environment variables still work as fallback
    # AIE_TRITON_COMPILE_ONLY=1 python my_script.py
"""

import contextlib
import logging
import os
from dataclasses import MISSING
from pathlib import Path

from aiec_air.toolchain.environment import ROOT


class _AIEConfig:
    """Process-global configuration for the AIE backend.

    Each property falls back to its corresponding environment variable
    when no programmatic value has been set. Call ``reset()`` to clear
    all programmatic overrides.
    """

    def __init__(self):
        self._compile_only = MISSING
        self._transform_tiling_script = MISSING
        self._air_project_path = MISSING
        self._debug = MISSING

    # ---- compile_only ----

    @property
    def compile_only(self) -> bool:
        """If True, compile kernels but skip XRT launch.

        Env var fallback: ``AIE_TRITON_COMPILE_ONLY`` (``"1"`` to enable).
        """
        if self._compile_only is not MISSING:
            return self._compile_only
        return os.getenv("AIE_TRITON_COMPILE_ONLY", "0") == "1"

    @compile_only.setter
    def compile_only(self, value: bool):
        self._compile_only = bool(value)

    # ---- transform_tiling_script ----

    @property
    def transform_tiling_script(self):
        """Path to a custom MLIR Transform dialect tiling script.

        When set, the backend loads this file instead of the built-in
        default tiling strategy. Set to ``None`` to use the default.

        Env var fallback: ``AIR_TRANSFORM_TILING_SCRIPT``.
        """
        if self._transform_tiling_script is not MISSING:
            return self._transform_tiling_script
        return os.getenv("AIR_TRANSFORM_TILING_SCRIPT")

    @transform_tiling_script.setter
    def transform_tiling_script(self, value):
        self._transform_tiling_script = value

    # ---- air_project_path ----

    @property
    def air_project_path(self) -> Path:
        """Directory where intermediate IR files and compiled artifacts are written.

        Defaults to ``build/air_project`` in the project.

        Env var fallback: ``AIE_TRITON_AIR_PROJECT_PATH``.
        """
        if self._air_project_path is not MISSING:
            return Path(self._air_project_path)
        custom = os.getenv("AIE_TRITON_AIR_PROJECT_PATH")
        if custom:
            return Path(custom)
        return ROOT / "build/air_project"

    @air_project_path.setter
    def air_project_path(self, value):
        if value is None:
            self._air_project_path = MISSING
            return
        self._air_project_path = value

    # ---- debug ----

    @property
    def debug(self) -> bool:
        """If True, enable verbose logging from subprocesses and the C++ launcher.

        Env var fallback: ``AIE_TRITON_DEBUG`` (``"1"`` to enable).
        """
        if self._debug is not MISSING:
            return self._debug
        return os.getenv("AIE_TRITON_DEBUG", "0") == "1"

    @debug.setter
    def debug(self, value: bool):
        self._debug = bool(value)
        # Keep the driver logger level in sync so logger.debug() calls
        # are enabled/suppressed when the flag is toggled programmatically.
        _drv = logging.getLogger("aiec_air.triton.driver")
        _drv.setLevel(logging.DEBUG if self._debug else logging.CRITICAL)

    def reset(self):
        """Clear all programmatic overrides, reverting to env var / default values."""
        self._compile_only = MISSING
        self._transform_tiling_script = MISSING
        self._air_project_path = MISSING
        self._debug = MISSING


# Module-level singleton
aie_config = _AIEConfig()


def set_config(**kwargs):
    """Set multiple configuration options at once.

    Example::

        set_config(compile_only=True, debug=True)

    Values that are ``dataclasses.MISSING`` (identity comparison) are
    skipped, so callers can forward optional overrides without branching::

        from dataclasses import MISSING
        set_config(debug=user_opts.get("debug", MISSING))

    Raises ``ValueError`` for unknown keys.
    """
    valid_keys = {
        "compile_only",
        "debug",
        "transform_tiling_script",
        "air_project_path",
    }
    for key, value in kwargs.items():
        if key not in valid_keys:
            raise ValueError(
                f"Unknown config key: {key!r}. Valid keys: {sorted(valid_keys)}"
            )
        if value is MISSING:
            continue
        setattr(aie_config, key, value)


@contextlib.contextmanager
def config_context(**kwargs):
    """Temporarily override configuration settings, restoring on exit.

    Example::

        with config_context(compile_only=True):
            kernel[grid](a, b, c)
    """
    saved = aie_config.__dict__.copy()
    try:
        set_config(**kwargs)
        yield aie_config
    finally:
        aie_config.__dict__.clear()
        aie_config.__dict__.update(saved)
