# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Extracted from the Triton-XDNA AIE1 driver adapter.

"""Target tiling of already converted AIR; grid mapping belongs to the compiler."""

import logging
import os
import re
import shutil
import tempfile

from air.compiler.util import run_transform
from air.ir import Context, Module
import air.passmanager

from aiec_air.triton.config import aie_config
from aiec_air.triton.transform_inject import _inject_transform_library

logger = logging.getLogger(__name__)


class LaunchUnsupported(RuntimeError):
    pass


def _dump_ir_if_needed(files):
    """
    Dump intermediate IR files to the air_project directory.

    Files are always dumped to the air_project path (controlled by
    ``aie_config.air_project_path`` or defaulting to ./air_project/).
    """
    air_proj_path = aie_config.air_project_path
    os.makedirs(air_proj_path, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(air_proj_path, os.path.basename(f)))


def _get_transform_ir_string():
    """
    Get the transform IR string for tiling operations.

    Priority: (1) a user-supplied script via
    ``aie_config.transform_tiling_script`` / ``AIR_TRANSFORM_TILING_SCRIPT``;
    (2) the built-in default tiling IR string. The auto-generated matmul
    schedule of the NPU backend stages through L2, which AIE1 does not have.

    If the script uses `transform.include`, the shared transform library
    (transform_library.mlir) is automatically injected.

    Returns:
        str: The transform IR string to use for tiling
    """
    custom_script_path = aie_config.transform_tiling_script

    if custom_script_path:
        if not os.path.isfile(custom_script_path):
            raise FileNotFoundError(
                f"transform_tiling_script / AIR_TRANSFORM_TILING_SCRIPT is set to "
                f"'{custom_script_path}' but the file was not found "
                f"(cwd: {os.getcwd()}). Use an absolute path or run from the "
                f"directory containing the script."
            )
        with open(custom_script_path, "r") as f:
            logger.debug("Using custom tiling script from: %s", custom_script_path)
            user_script = f.read()
        return _inject_transform_library(user_script)

    # Default hardcoded transform IR string
    matmul_tiling_size_l1_m = 32
    matmul_tiling_size_l1_n = 32
    matmul_tiling_size_l1_k = 32
    elemwise_tiling_size_l1_m = 32
    elemwise_tiling_size_l1_n = 32

    return f"""
        module attributes {{transform.with_named_sequence}} {{
          transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
                %mul = transform.structured.match ops{{["linalg.mul"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %mul_1, %loop = transform.air.linalg_tile %mul [{elemwise_tiling_size_l1_m}, {elemwise_tiling_size_l1_n}]
                transform.air.linalg_promote %mul_1 <{{"operands_to_promote"=[2], "memory_space"="L1"}}> : (!transform.any_op) -> !transform.any_op
                transform.air.linalg_promote %mul_1 <{{"operands_to_promote"=[0,1], "memory_space"="L1"}}> : (!transform.any_op) -> !transform.any_op

                %add = transform.structured.match ops{{["linalg.add"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %add_1, %add_loop = transform.air.linalg_tile %add [{elemwise_tiling_size_l1_m}, {elemwise_tiling_size_l1_n}]
                transform.air.linalg_promote %add_1 <{{"operands_to_promote"=[2], "memory_space"="L1"}}> : (!transform.any_op) -> !transform.any_op
                transform.air.linalg_promote %add_1 <{{"operands_to_promote"=[0,1], "memory_space"="L1"}}> : (!transform.any_op) -> !transform.any_op

                %matmul = transform.structured.match ops{{["linalg.matmul"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %fill = transform.structured.match ops{{["linalg.fill"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %matmul_1, %matmul_loop = transform.air.linalg_tile %matmul [{matmul_tiling_size_l1_m}, {matmul_tiling_size_l1_n}]
                %fill_1 = transform.air.fuse_into_containing_op %fill into %matmul_loop : (!transform.any_op, !transform.any_op) -> !transform.any_op
                transform.air.linalg_promote %fill_1 <{{"operands_to_promote"=[1], "memory_space"="L1"}}> : (!transform.any_op) -> !transform.any_op
                transform.air.linalg_promote %matmul_1 <{{"operands_to_promote"=[2], "memory_space"="L1"}}> : (!transform.any_op) -> !transform.any_op
                %matmul_2, %reduction_loop = transform.air.linalg_tile %matmul_1 [0, 0, {matmul_tiling_size_l1_k}]
                transform.air.linalg_promote %matmul_2 <{{"operands_to_promote"=[0,1], "memory_space"="L1"}}> : (!transform.any_op) -> !transform.any_op
            transform.yield
          }}
        }}
        """


# Part of every compiled-kernel key: bump when the lowering below changes.
LOWERING_VERSION = 4


def herd_shapes(air_module):
    """The constant (x, y) size of every air.herd, in program order."""
    from air.ir import DenseI32ArrayAttr, IntegerAttr

    shapes = []

    def visit(operation):
        for region in operation.regions:
            for block in region.blocks:
                for op in block.operations:
                    op = op.operation
                    if op.name == "air.herd":
                        # operands: async dependencies, sizes, kernel operands
                        deps, count = list(DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))[:2]
                        sizes = op.operands[deps : deps + count]
                        shapes.append(tuple(IntegerAttr(s.owner.attributes["value"]).value for s in sizes))
                    visit(op)

    visit(air_module.operation)
    return shapes


def _rank_pointer_arguments(air_module, sizes):
    """Give each unranked pointer argument the static size of its launch buffer.

    triton-shared passes a pointer as memref<*xT>; air-to-aie for AIE1 builds
    shim DMAs from ranked memrefs only and asserts on unranked ones. The size is
    the element count of the buffer the launch passes, as the XDNA launcher
    sizes its XRT buffers. ``sizes`` holds one count, or None, per argument.
    """
    from air.dialects import memref as memref_d
    from air.ir import FunctionType, InsertionPoint, MemRefType, TypeAttr, UnrankedMemRefType

    funcs = [op for op in air_module.body.operations if op.operation.name == "func.func"]
    assert len(funcs) == 1
    func = funcs[0]
    entry = func.regions[0].blocks[0]
    function_type = FunctionType(TypeAttr(func.attributes["function_type"]).value)
    inputs = list(function_type.inputs)
    for i, size in enumerate(sizes):
        if size is None or inputs[i].typeid != UnrankedMemRefType.static_typeid:
            continue
        argument = entry.arguments[i]
        with argument.location:
            unranked = UnrankedMemRefType(inputs[i])
            ranked = MemRefType.get([size], unranked.element_type)
            argument.set_type(ranked)
            inputs[i] = ranked
            # The body keeps its unranked view; canonicalize folds the cast away.
            with InsertionPoint.at_block_begin(entry):
                view = memref_d.CastOp(unranked, argument)
        argument.replace_all_uses_except(view.result, view.operation)
    func.attributes["function_type"] = TypeAttr.get(FunctionType.get(inputs, function_type.results))


def _air_to_herds(mod, sizes=()):
    """Promote and tile an existing AIR launch for the AIE1 target."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = os.path.join(tmpdir, "airinput.mlir")
        air_context = Context()
        air_module = Module.parse(mod, context=air_context)
        launches = []
        def visit(op):
            if op.name == "air.launch":
                launches.append(op)
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations:
                        visit(child.operation)
        visit(air_module.operation)
        if not launches:
            raise LaunchUnsupported("Target tiling requires AIR launches from the compiler")
        with air_context:
            _rank_pointer_arguments(air_module, sizes)
        pipeline = "builtin.module(air-override-memref-memory-space{scope=func memory-space=1})"
        pm = air.passmanager.PassManager.parse(pipeline, context=air_context)
        pm.run(air_module.operation)
        # MLIR-AIR compilation step 2: tiling the launch body
        transform_ir_string = _get_transform_ir_string()
        transform_ir = Module.parse(transform_ir_string, context=air_context)
        run_transform(transform_ir, air_module)
        # AIE1 has no L2: the herds move their parts of the block between L3
        # and L1 themselves instead of through the script's L2 buffers.
        from aiec_air.toolchain.runner import air_opt

        staged = air_opt(str(air_module), "--air-eliminate-staging-buffers")
        if re.search(r"memref<[^>]*, 1( : i32)?>", staged):
            # e.g. a masked load fills only part of its staging buffer.
            raise LaunchUnsupported(
                "An L2 buffer of the transform script is still in use after "
                "air-eliminate-staging-buffers; AIE1 has no L2 memory"
            )
        air_module = Module.parse(staged, context=air_context)
        # Copies introduced by target promotion become DMA operations.
        pipeline = "builtin.module(canonicalize,cse,air-copy-to-dma)"
        pm = air.passmanager.PassManager.parse(pipeline, context=air_context)
        pm.run(air_module.operation)
        with open(dst_path, "w") as f:
            f.write(str(air_module))
        _dump_ir_if_needed([dst_path])
        return air_module


