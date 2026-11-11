"""Run a compiler tool from PATH over textual MLIR."""

from __future__ import annotations

import subprocess

from aiec_air.toolchain.environment import tool


class NativeCompileError(RuntimeError):
    """A native compiler tool failed."""


def air_opt(text: str, *arguments: str, timeout: int = 300) -> str:
    """`text` transformed by air-opt (the mlir-air fork, on PATH through
    env.sh), in a separate process."""
    done = subprocess.run([str(tool("air-opt")), "-", *arguments], input=text, capture_output=True, text=True,
                          timeout=timeout)
    if done.returncode:
        raise NativeCompileError(f"air-opt {' '.join(arguments)} failed:\n{done.stderr[-4000:]}")
    return done.stdout
