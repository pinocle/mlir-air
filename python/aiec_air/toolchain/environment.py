"""Where the compilers and the AMD tools are, from the process environment.

As in mlir-aie and mlir-air, tools are found on PATH and the Python bindings
on PYTHONPATH: ``python tools/build_upstream.py`` builds the mlir-aie and
mlir-air forks, ``source env.sh`` puts their installs on both. The AMD release
(Vitis, aietools, the AArch64 GCC, EmbeddedSW headers) is ``AIEC_VITIS_ROOT``,
or the parent of ``XILINX_VITIS`` that Vitis's settings64.sh sets.
"""
from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(os.environ.get("AIEC_PROJECT_ROOT", Path(__file__).resolve().parents[4])).resolve()
SOURCES = {"mlir-air": ROOT / "mlir-air", "mlir-aie": ROOT / "third_party/mlir-aie"}


class ToolNotFound(RuntimeError):
    pass


def vitis_root() -> Path:
    """The AMD release directory that holds Vitis/, Vivado/, data/ and gnu/."""
    configured = os.environ.get("AIEC_VITIS_ROOT")
    if configured:
        return Path(configured)
    if os.environ.get("XILINX_VITIS"):
        return Path(os.environ["XILINX_VITIS"]).parent
    raise ToolNotFound("No AMD tool install: set AIEC_VITIS_ROOT (see env.sh) or source Vitis's settings64.sh")


def aietools() -> Path:
    return vitis_root() / "Vitis/aietools"


def tool(name: str) -> Path:
    """A compiler tool (air-opt, aircc, aiecc, ...) from PATH."""
    found = shutil.which(name)
    if found is None:
        raise ToolNotFound(f"{name} is not on PATH: python tools/build_upstream.py, then source env.sh")
    return Path(found)


@functools.cache
def revisions() -> dict[str, str]:
    """The source revision of each fork the tools are built from: the commit,
    and a digest of uncommitted changes when there are any."""
    result = {}
    for name, source in SOURCES.items():
        def git(*arguments):
            return subprocess.run(["git", "-C", str(source), *arguments], capture_output=True, check=True).stdout
        try:
            revision = git("rev-parse", "--short=12", "HEAD").decode().strip()
            diff = git("diff", "HEAD") + git("ls-files", "--others", "--exclude-standard")
        except (OSError, subprocess.CalledProcessError):
            result[name] = "unknown"
            continue
        result[name] = revision + (f"+dirty.{hashlib.sha256(diff).hexdigest()[:12]}" if diff else "")
    return result
