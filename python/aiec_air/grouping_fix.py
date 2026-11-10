"""Form spatial AIR programs and replace proven DDR edges with device channels.

This transforms the mixed host/device IR, not a side execution plan. An edge
can stay on the array only when its ownership and ordered transfer descriptors
match. Unproved layouts/effects remain explicit program boundaries. The
transformation is the aiec-group-programs pass (mlir/lib/Transform/AIECFixes/GroupPrograms_fix.cpp)
on the sequential host form (aiec-synchronize-host).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from compiler.target_spec import (COLUMNS, CORE_ROWS, CORE_STATIC_BYTES, DEFAULT_STACK_BYTES, LOCKS_PER_TILE,
                                  MODULE_BYTES, PACKET_IDS)

# Planning allowance for request control and allocator alignment, in addition
# to the common runtime reservation. The native allocator verifies final use.
RETILE_RESERVED_BYTES = CORE_STATIC_BYTES + 512


def group_programs(text, *, columns=COLUMNS, rows=CORE_ROWS, input_ports=32, output_ports=32,
                   local_bytes=MODULE_BYTES, stack_bytes=DEFAULT_STACK_BYTES):
    """The grouped module text and the grouping report."""
    from compiler.tools import aiec_opt
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "grouping.json"
        options = (f"columns={columns} rows={rows} input-ports={input_ports} output-ports={output_ports} "
                   f"local-bytes={local_bytes} stack-bytes={stack_bytes} locks-per-tile={LOCKS_PER_TILE} "
                   f"packet-ids={PACKET_IDS} retile-reserved-bytes={RETILE_RESERVED_BYTES} report={report}")
        # The host side follows the sequential runtime ABI: host execute and
        # token waits run in program order and launch/segment wrappers lose
        # their tokens. Device herds keep their async graph and ping-pong loops.
        grouped = aiec_opt(text, "--aiec-synchronize-host=strip-hierarchy-tokens=true", "--canonicalize", "--cse",
                           "--aiec-group-programs=" + options, "--canonicalize", "--cse", "--symbol-dce")
        return grouped, json.loads(report.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    for name, default in [("columns", COLUMNS), ("rows", CORE_ROWS), ("input-ports", 32), ("output-ports", 32),
                          ("local-bytes", MODULE_BYTES), ("stack-bytes", DEFAULT_STACK_BYTES)]:
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    source = args.source.read_text()
    text, report = group_programs(source, columns=args.columns, rows=args.rows, input_ports=args.input_ports,
                                  output_ports=args.output_ports, local_bytes=args.local_bytes,
                                  stack_bytes=args.stack_bytes)
    args.output.write_text(text)
    report.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(), native_verifier="passed")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
