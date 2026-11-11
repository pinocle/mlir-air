"""Explain evidenced native resource failures without retrying smaller designs."""
from pathlib import Path


def explain_native_failure(directory):
    logs = sorted(Path(directory).rglob('*.stderr.log'), key=lambda p:p.stat().st_mtime_ns, reverse=True)
    for path in logs:
        if not path.stat().st_size:
            continue
        text = path.read_text(errors='replace')
        if ('number of input DMA channel exceeded' in text or
            'ObjectFIFO endpoint demand exceeds per-core DMA capacity' in text):
            return {'component':'ObjectFIFO dataflow / AIE1 DMA channel contract',
                'problem':'Independent FIFO inputs exceed per-core S2MM channels (AIE1: 2)',
                'evidence':str(path),
                'effect':'Compilation rejected without reducing cores or switching backend',
                'solution':'Keep the accumulator/initial value local, or explicitly schedule several logical inputs on shared DMA channels while preserving dependencies',
                'automatic_core_count_retry':False}
        if 'lock assigned invalid id' in text:
            return {'component':'native AIR-to-AIE resource allocation',
                'problem':'A native logical/physical tile exceeds its lock ID capacity',
                'evidence':str(path),
                'effect':'Device lowering rejected; no executable produced for this program',
                'solution':'Make channel/BD/lock allocation capacity-aware before emitting per-tile resources; consider ObjectFIFO lowering with a compatible host/PL binding',
                'automatic_core_count_retry':False}
        if 'Packet ID field can only hold' in text:
            return {'component':'AIR-to-AIE packet allocation / scheduling contract',
                'problem':'Generated packet identifiers exceed the target encoding',
                'evidence':str(path),
                'effect':'Device lowering rejected',
                'solution':'Account for the current packet namespace, then implement legal route-scoped ID reuse to expand it',
                'automatic_core_count_retry':False}
        if 'is assigned to the same lock' in text:
            return {'component':'AIR/AIE placement ownership contract',
                'problem':'Placement merged already-numbered lock namespaces',
                'evidence':str(path), 'effect':'Physical materialization rejected',
                'solution':'Preserve logical resource ownership or use a coordinated native resource allocator',
                'automatic_core_count_retry':False}
    return None
