"""Prove uniform initializers and keep promoted reductions in local memory.

This is the contract between native Linalg promotion and AIR channel lowering.
It changes allocation/copy placement, never the scalar computation region.
The analysis and transforms are aiec-opt passes (mlir/lib/Transform/AIECFixes/Residency_fix.cpp).
"""
import argparse
import json
from pathlib import Path

INIT = 'aie1.uniform_output_init'


def annotate_initializers(text):
    """`text` with aie1.uniform_output_init on Linalg operations whose output
    buffers hold a proven whole-buffer constant (aiec-annotate-uniform-init)."""
    from compiler.tools import aiec_opt
    return aiec_opt(text, "--aiec-annotate-uniform-init")


def _run(source, output, arguments, suffix):
    from compiler.tools import aiec_opt
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    report = output.with_suffix(suffix)
    output.write_text(aiec_opt(Path(source).read_text(), *(a.format(report=report) for a in arguments)))
    return json.loads(report.read_text())


def localize(source, output):
    """Fill promoted accumulators from proven uniform initial values
    (aiec-localize-accumulators); writes <output>.residency.json."""
    return _run(source, output, ['--aiec-localize-accumulators=report={report}', '--canonicalize', '--cse'],
                '.residency.json')


def split_readwrite_channels(source, output):
    """Give a herd buffer that is both received and sent distinct FIFO owners
    (aiec-split-readwrite-fifo); writes <output>.fifo.json."""
    return _run(source, output, ['--aiec-split-readwrite-fifo=report={report}'], '.fifo.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--split-readwrite', action='store_true')
    args = parser.parse_args()
    (split_readwrite_channels if args.split_readwrite else localize)(args.source, args.output)
