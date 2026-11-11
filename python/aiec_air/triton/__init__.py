"""Triton launch lowering and driver support owned by MLIR-AIR."""

# Complete Triton plugin discovery before a direct driver submodule import.
# During discovery this returns the already-initializing Triton package.
import triton as triton
