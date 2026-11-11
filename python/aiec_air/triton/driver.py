# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# From Triton-XDNA amd_triton_npu/backend/driver.py, changed for AIE1: NPU
# detection, Peano, XRT, HSA, MSVC and launcher generation are gone, pointer
# arguments are ranked by their launch buffers, the transform script's L2
# buffers are removed (air-eliminate-staging-buffers), and the device build is
# an AIE1 bundle (deployment.aie1: aircc --device=xcvc1902).

import hashlib
import json
import logging
import tempfile
import sys

import os
import shutil

from math import prod
from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget


L1_BYTES = 32 * 1024  # AIE1 core-tile data memory

from aiec_air.toolchain.environment import revisions
from aiec_air.triton.config import aie_config
from aiec_air.triton.launch import LaunchUnsupported, LOWERING_VERSION, herd_shapes, _air_to_herds
from aiec_air.triton.codegen import (
    written_pointer_args,
    extract_actual_sizes,
    extract_signature_and_constants,
    ty_to_cpp,
)


def _make_active_driver_per_thread():
    """Keep Triton's active driver per thread instead of per process.

    Triton holds one active driver for the whole process, and a hetero model
    switches it per scope: the AIE driver around AIE launches and AIEChain
    warmups, the AMD one around iGPU work. With those in two threads, a switch
    in one thread changes the backend the other compiles and launches for.

    Triton's ``DriverConfig`` keeps the choice in ``_active``. Its one instance
    gets a subclass that keeps the choice in a ContextVar instead. A thread
    that has not chosen a driver since then sees the one that was active
    process-wide when this ran.

    Called from ``AIEDriver.__init__``, which every switch to the AIE goes
    through, rather than at import: this module is imported while Triton is
    still discovering backends, before ``triton.runtime.driver`` can load.
    """
    import contextvars

    from triton.runtime.driver import driver as config

    if getattr(type(config), "per_thread", False):
        return
    unset = object()
    active = contextvars.ContextVar("triton_active_driver", default=unset)
    shared = config.__dict__.pop("_active", None)

    class PerThreadDriverConfig(type(config)):
        per_thread = True

        @property
        def _active(self):
            driver = active.get()
            return shared if driver is unset else driver

        @_active.setter
        def _active(self, driver):
            active.set(driver)

    config.__class__ = PerThreadDriverConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
if aie_config.debug:
    logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


# Kernels compiled by this process, by kernel and grid.
_global_module_cache = {}


def compile_module(signature, *, lower_common_ir, actual_sizes=None, on_cache_resolved=None):
    """Lower a kernel at launch, when its grid is known."""

    def launch(
        gridX,
        gridY,
        gridZ,
        stream,
        cu_function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        if prod((gridX, gridY, gridZ)) < 1:
            raise ValueError(
                "Must set at least 1 program in the launch grid; to launch one "
                "program, invoke the kernel as kernel[(1,)](...)"
            )
        asm_src = cu_function.encode() if isinstance(cu_function, str) else cu_function
        kernel_name = kernel_metadata[6]  # see pack_metadata in compiler.py
        return_types = kernel_metadata[7]
        user_script = aie_config.transform_tiling_script
        toolchain = json.dumps(revisions(), sort_keys=True)  # the forks the tools are built from

        # Element count of each pointer argument's buffer (see _rank_pointer_arguments).
        arguments = [arg for i, arg in enumerate(args) if signature[i] != "constexpr"]
        sizes = tuple(
            arg.numel() if ty.startswith("*") and hasattr(arg, "numel") else None
            for ty, arg in zip((t for t in signature.values() if t != "constexpr"), arguments)
        )
        input_key = hashlib.md5(
            asm_src + f"_{gridX}_{gridY}_{gridZ}_{kernel_name}_{sizes}"
            f"_{actual_sizes}_{user_script or ''}_{toolchain}_{LOWERING_VERSION}".encode()
        ).hexdigest()

        if input_key not in _global_module_cache:
            air_proj_path = aie_config.air_project_path
            os.makedirs(air_proj_path, exist_ok=True)
            conversion_dir = Path(tempfile.mkdtemp(prefix="common-", dir=air_proj_path))
            source = conversion_dir / "common.mlir"
            source.write_bytes(asm_src)
            common_air = lower_common_ir(source, conversion_dir / "air",
                grid=(gridX, gridY, gridZ), actual_sizes=actual_sizes)
            air_output = _air_to_herds(Path(common_air["output"]).read_text(), sizes=sizes)
            key_data = (
                str(air_output)
                + f"_toolchain_{toolchain}"
                + f"_lowering_{LOWERING_VERSION}"
            )
            key = hashlib.md5(key_data.encode("utf-8")).hexdigest()
            cache = get_cache_manager(key)
            cached = cache.get_file("aie-kernel.json")
            if cached is not None:
                report = json.loads(Path(cached).read_text())
            else:
                from deployment.aie1 import build_bundle

                # build_bundle (aircc --device=xcvc1902) writes a new
                # directory; the report, written last, is what makes the
                # cache entry complete.
                build = Path(tempfile.mkdtemp(prefix="aie-", dir=cache.cache_dir))
                try:
                    (build / "air.mlir").write_text(str(air_output))
                    bundle = build_bundle(build / "air.mlir", build / "bundle")
                except BaseException:
                    shutil.rmtree(build, ignore_errors=True)
                    raise
                report = {
                    "schema_version": 2,
                    "cache_dir": cache.cache_dir,
                    "grid": [gridX, gridY, gridZ],
                    "herds": [list(shape) for shape in herd_shapes(air_output)],
                    "air": str(build / "air.mlir"),
                    "common_air_compilation": str(conversion_dir / "air/compilation.json"),
                    "bundle": str(build / "bundle"),
                    "functions": [f["name"] for f in bundle["functions"]],
                    "completed_stage": "aie1_bundle",
                    "board_execution_performed": False,
                }
                cache.put(json.dumps(report, indent=2) + "\n", "aie-kernel.json")
            _global_module_cache[input_key] = report

        report = _global_module_cache[input_key]
        if on_cache_resolved is not None:
            on_cache_resolved(report)
        if return_types:
            # Triton's launch has no path for a kernel's results to its caller.
            raise LaunchUnsupported(f"{kernel_name} returns {return_types}; AIE kernels write through pointers")
        if aie_config.compile_only:
            logger.debug("Compile-only mode: %s", report["cache_dir"])
            return None
        raise LaunchUnsupported(
            f"{kernel_name} compiled for grid {(gridX, gridY, gridZ)} ({report}); the board runtime "
            f"has no kernel ABI for {len(arguments)} pointer/scalar arguments yet, so it is not executed"
        )

    return launch


class AIELauncher(object):
    aie_cache_dir: "str | None" = None
    report = None
    launch = None  # callable(gridX, gridY, gridZ, stream, function, *args)

    def __init__(self, src, metadata):
        """Build the launcher for ``src``; compilation waits for the grid."""
        constants, signature = extract_signature_and_constants(src)
        # Pointers the kernel stores to; None when that cannot be established.
        self.written = written_pointer_args(src)
        actual_sizes = extract_actual_sizes(src)

        def _on_cache_resolved(report):
            self.report = report
            self.aie_cache_dir = report["cache_dir"]

        self.launch = compile_module(signature, lower_common_ir=self.lower_common_ir,
                                     actual_sizes=actual_sizes, on_cache_resolved=_on_cache_resolved)

    def lower_common_ir(self, source, out_dir, **options):
        raise LaunchUnsupported("Use the registered AIE compiler driver for common-IR -> AIR conversion")

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        self.launch(gridX, gridY, gridZ, stream, function, *args)


def get_aie_cache_dir(compiled_kernel):
    """The AIE build directory of a compiled kernel, or None before its first
    launch: AIE compilation needs the grid, so it happens at launch."""
    launcher = getattr(compiled_kernel, "_run", None)
    if launcher is None:
        return None
    if not isinstance(launcher, AIELauncher):
        raise TypeError(
            f"Expected an AIELauncher but got {type(launcher).__name__}. "
            "Is the AIE backend active?"
        )
    return launcher.aie_cache_dir


class AIEUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(AIEUtils, cls).__new__(cls)
        return cls.instance

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the npu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": L1_BYTES,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # assembly source code so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_asm, shared, device):
        return (
            name,  # module: CompiledKernel makes a new launcher while it is None
            kernel_asm,  # function
            None,  # n_regs
            None,  # n_spills
            sys.maxsize,  # n_max_threads
        )

    @staticmethod
    def unload_module(module):
        return


class AIEDriver(DriverBase):
    launcher_cls = AIELauncher

    def __init__(self):
        _make_active_driver_per_thread()
        super().__init__()
        self.utils = AIEUtils()
        self.binary_ext = "ttsharedir"

    # AIE driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(AIEDriver()) or driver_scope("aie").
    @staticmethod
    def is_active():
        return False

    def do_bench(self, fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
        raise LaunchUnsupported("AIE kernel timing needs board execution, which the driver does not run yet")

    def get_benchmarker(self):
        return self.do_bench

    def get_device_capability(self):
        return ("aie", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # AIE doesn't have a device to return. Return something.
        return "aie"

    def set_current_device(self, device):
        # AIE doesn't have a device to set
        assert device == "aie"
        return

    def get_current_target(self):
        return GPUTarget("aie", 0, 1)

    def get_active_torch_device(self):
        import torch

        # Kernel buffers are host tensors: the board's ARM and AIE share DDR.
        return torch.device("cpu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)
