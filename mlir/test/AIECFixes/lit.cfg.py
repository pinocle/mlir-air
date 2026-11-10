"""Tests of the fork's AIE1 passes, linked into the companion aiec-opt."""
import os

import lit.formats

config.name = "aie-compiler-air-fixes"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.aiec_binary_dir, "air-fix-tests")
config.environment["PATH"] = os.pathsep.join([
    config.aiec_tools_dir, config.llvm_tools_dir, config.environment.get("PATH", "")])
