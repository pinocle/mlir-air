"""Lower synchronous AIR host control to native LLVM/AArch64 callbacks.

The original mixed module supplies execution order, constants and views. Device
regions are projected separately; their host channel endpoints become runtime
bindings. No model arithmetic is interpreted or implemented on the CPU.
Run this module with the isolated AIR environment.
"""
import argparse
import hashlib
import json
import re
import struct
from math import prod
from pathlib import Path
import subprocess

from compiler.environment import aietools


from compiler.environment import ROOT
CLANG = aietools() / 'lnx64.o/tools/clang/bin/clang'


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def llvm18_compatible(text):
    """Losslessly spell LLVM24 FP32 constants for the installed LLVM18.

    LLVM24 uses f0x literals and shortest-roundtrip decimals/inf, while LLVM18
    requires exact double-width hex spellings for non-exact FP32 decimals.
    New GEP no-wrap promises are dropped (not arithmetic operations); the
    emitted host object preserves all values on defined source executions.
    """
    counts = {}
    from compiler.llvm_compat import fp32_double_hex, rewrite_fp32_hex

    def rewrite(pattern, replacement, name):
        nonlocal text
        text, counts[name] = re.subn(pattern, replacement, text)

    text, counts['fp32_hex'] = rewrite_fp32_hex(text)
    rewrite(r'\bfloat (-?\d+\.\d+e[+-]\d+)',
            lambda m: 'float ' + fp32_double_hex(struct.pack('>f', float(m[1]))), 'fp32_decimal')
    rewrite(r'\bfloat (-?inf)\b', lambda m: 'float ' + fp32_double_hex(struct.pack('>f', float(m[1]))), 'fp32_inf')
    rewrite(r'\bcaptures\(none\)', 'nocapture', 'nocapture')
    rewrite(r'(?<=getelementptr )(?:(?:inbounds|nuw|nusw) )+',
            lambda m: 'inbounds ' if 'inbounds' in m[0].split() else '', 'gep_flags')
    return text, counts


def lower_host(source, output, *, partition_manifest=None):
    """Generate verified host MLIR with the original source control flow."""
    from air import ir
    from air.dialects import arith, func, llvm, memref
    from air.passmanager import PassManager
    from compiler.ir_tools import walk

    source, output = Path(source).absolute(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(Path(partition_manifest).read_text()) if partition_manifest else None
    if manifest and manifest['source_sha256'] != _hash(source):
        raise ValueError('Partition source hash does not match host source')
    # The host side of an async schedule runs in program order (aiec-opt
    # pass); device segments keep their tokens. The source file is unchanged.
    from compiler.tools import aiec_opt
    synchronized = aiec_opt(source.read_text(), '--aiec-synchronize-host')
    with ir.Context() as context, ir.Location.unknown():
        context.load_all_available_dialects()
        module = ir.Module.parse(synchronized)
        module.operation.verify()
        entries = [o for o, _ in walk(module.operation) if o.name == 'func.func' and
                   o.regions[0].blocks and ('sym_visibility' not in o.attributes or
                   o.attributes['sym_visibility'].value != 'private')]
        if len(entries) != 1:
            raise ValueError('Exactly one public host entry is required')
        entry = entries[0]
        block = entry.regions[0].blocks[0]
        shapes = []
        for arg in block.arguments:
            typ = ir.MemRefType(arg.type)
            if str(typ.element_type) != 'f32' or any(d <= 0 for d in typ.shape):
                raise ValueError('Host entry requires static FP32 ranked buffers')
            shapes.append(list(typ.shape))
        if len(shapes) != 2:
            raise ValueError('Host entry requires one input and one output')
        # Host lowering below executes launches in source order and waits for
        # each one before continuing. Reuse only allocations whose complete
        # initialization and non-overlapping lifetimes are proven under that
        # existing synchronous contract; the source/device IR is unchanged.
        from compiler.model_analysis import _function
        host_memory = _function(ir, entry, optimize=True, synchronous_host=True)['reuse']
        entry.attributes['sym_name'] = ir.StringAttr.get('native_model')
        entry.attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()
        declarations = {o.attributes['sym_name'].value: [ir.IntegerAttr(v).value for v in ir.ArrayAttr(o.attributes['size'])]
                        for o, _ in walk(module.operation) if o.name == 'air.channel'}
        channel_ids = {name: i for i, name in enumerate(declarations)}
        launches = [o for o, _ in walk(module.operation) if o.name == 'air.launch']
        if not launches:
            raise ValueError('Host source has no AIR launches')
        if manifest and [p['source_order_id'] for p in manifest['programs']] != list(range(len(launches))):
            raise ValueError('Model requires a complete, ordered partition manifest')
        index, i64, ptr = ir.IndexType.get(), ir.IntegerType.get_signless(64), ir.Type.parse('!llvm.ptr')
        with ir.InsertionPoint(module.body):
            func.FuncOp('native_model_transfer', ([i64] * 5 + [ptr, i64, ptr, ptr], []), visibility='private')
            func.FuncOp('native_model_execute', ([i64], []), visibility='private')

        def const(n, typ=index):
            return arith.ConstantOp(typ, ir.IntegerAttr.get(typ, int(n))).result

        def cast(v):
            return arith.IndexCastOp(i64, v).result

        def pointer(v):
            return llvm.IntToPtrOp(ptr, cast(memref.ExtractAlignedPointerAsIndexOp(v).result)).result

        transfers = []
        for program_id, launch in enumerate(launches):
            sizes = list(ir.DenseI32ArrayAttr(launch.attributes['operandSegmentSizes']))
            if sizes[1] != 2:
                raise ValueError('Only two-dimensional launches are supported')
            launch_block = launch.regions[0].blocks[0]
            operands = list(launch.operands)[sizes[0]:]  # after async dependencies (complete tokens)
            dims = operands[:2]
            if any(v.owner.name != 'arith.constant' or ir.IntegerAttr(v.owner.attributes['value']).value != 1 for v in dims):
                raise ValueError('Launch dimensions must be static 1x1')
            with ir.InsertionPoint(launch):
                zero = const(0)
            replacements = [zero, zero, *dims, *operands[2:]]
            if len(replacements) != len(launch_block.arguments):
                raise ValueError('Unsupported launch argument layout')
            for arg, value in zip(launch_block.arguments, replacements):
                arg.replace_all_uses_with(value)
            segments = [o for o, _ in walk(launch) if o.name == 'air.segment']
            if len(segments) != 1:
                raise ValueError('Each launch must have one device segment')
            segment = segments[0]
            # Reject host-side arithmetic before erasing only the separately
            # compiled device segment. Constant initialization is allowed.
            # Integer/index arithmetic computes transfer addresses and extents
            # (including mask clamps such as min/max); any floating-point or
            # math operation would be model computation and is rejected.
            for op, parents in walk(launch):
                if 'air.segment' in parents:
                    continue
                if op.name == 'arith.constant':
                    continue
                address_only = op.name.startswith('arith.') and all(
                    isinstance(r.type, (ir.IndexType, ir.IntegerType)) for r in op.results)
                if op.name.startswith(('math.', 'arith.')) and not address_only:
                    raise ValueError('Model arithmetic outside device segment: ' + op.name)
            endpoints = [o for o, parents in walk(launch) if o.name in {'air.channel.put', 'air.channel.get'}
                         and 'air.segment' not in parents]
            for endpoint in endpoints:
                fields = list(ir.DenseI32ArrayAttr(endpoint.attributes['operandSegmentSizes']))
                if any(list(r.uses) for r in endpoint.results):
                    raise ValueError('Host channel endpoint token is still used')
                name = ir.FlatSymbolRefAttr(endpoint.attributes['chan_name']).value
                if name not in declarations:
                    raise ValueError('Undeclared channel ' + name)
                bundle_dims = declarations[name]
                # A single-bundle channel (e.g. one broadcast group) may be
                # addressed without indices.
                if fields[1] != len(bundle_dims) and not (fields[1] == 0 and prod(bundle_dims) == 1):
                    raise ValueError('Channel bundle index rank mismatch')
                operands = list(endpoint.operands)[fields[0]:]  # after async dependencies
                buffer = operands[fields[1]]
                typ = ir.MemRefType(buffer.type)
                rank = len(typ.shape)
                if not rank or rank > 8 or str(typ.element_type) != 'f32':
                    raise ValueError('Channel payload must be FP32 rank 1 through 8')
                if str(typ.memory_space) not in {'0 : i32', '0 : i64', '0', 'None'}:
                    raise ValueError('Host channel payload must reside in DDR memory space')
                direction = int(endpoint.name == 'air.channel.get')
                prefix = 'dst' if direction else 'src'
                # An explicit descriptor may view a buffer with a different
                # rank (e.g. a strided 3-D frame of a flat buffer); its own
                # offsets/sizes/strides then define the transfer rank.
                statics = {field: list(ir.DenseI64ArrayAttr(endpoint.attributes[f'static_{prefix}_{field}']))
                           for field in ('offsets', 'sizes', 'strides')}
                lengths = {len(v) for v in statics.values() if v}
                if len(lengths) > 1 or (lengths and not all(statics[f] for f in ('sizes', 'strides'))):
                    raise ValueError('Channel descriptor rank mismatch')
                descriptor_rank = lengths.pop() if lengths else rank
                if descriptor_rank > 8:
                    raise ValueError('Channel descriptor rank exceeds 8')
                # Allocas are entry-local, never inside frame loops; callbacks
                # copy descriptors synchronously before these arrays are reused.
                with ir.InsertionPoint(block.operations[0]):
                    size_array = memref.AllocaOp(ir.MemRefType.get([descriptor_rank], i64), [], []).result
                    stride_array = memref.AllocaOp(ir.MemRefType.get([descriptor_rank], i64), [], []).result
                with ir.InsertionPoint(endpoint):
                    metadata = memref.ExtractStridedMetadataOp(buffer)
                    base_offset = metadata.results[1]
                    logical_sizes = list(metadata.results[2:2 + rank])
                    logical_strides = list(metadata.results[2 + rank:])
                    cursor = fields[1] + 1
                    resolved = []
                    for field_index, field in enumerate(('offsets', 'sizes', 'strides'), start=3):
                        static = statics[field]
                        dynamic = iter(operands[cursor:cursor + fields[field_index]])
                        cursor += fields[field_index]
                        vals = [next(dynamic) if n == -(1 << 63) else const(n) for n in static]
                        resolved.append(vals)
                    offsets, frame_sizes, frame_strides = resolved
                    frame_sizes = frame_sizes or logical_sizes
                    frame_strides = frame_strides or logical_strides
                    offset = base_offset
                    for value, stride in zip(offsets, frame_strides):
                        offset = arith.AddIOp(offset, arith.MulIOp(value, stride).result).result
                    bundle = const(0)
                    for value, dim in zip(operands[:fields[1]], bundle_dims):
                        bundle = arith.AddIOp(arith.MulIOp(bundle, const(dim)).result, value).result
                    for j, (size, stride) in enumerate(zip(frame_sizes, frame_strides)):
                        memref.StoreOp(cast(size), size_array, [const(j)])
                        memref.StoreOp(cast(stride), stride_array, [const(j)])
                    args = [const(program_id, i64), const(channel_ids[name], i64), const(direction, i64),
                            cast(bundle), const(descriptor_rank, i64), pointer(buffer), cast(offset),
                            pointer(size_array), pointer(stride_array)]
                    func.CallOp([], 'native_model_transfer', args)
                endpoint.erase()
                transfers.append({'program_id': program_id, 'channel': name, 'channel_id': channel_ids[name],
                                  'direction': direction, 'rank': descriptor_rank, 'bundle_shape': bundle_dims})
            segment.erase()
            terminator = list(launch_block.operations)[-1].operation
            if terminator.name != 'air.launch_terminator':
                raise ValueError('Unexpected launch terminator')
            for child in list(launch_block.operations)[:-1]:
                child.operation.move_before(launch)
            with ir.InsertionPoint(launch):
                func.CallOp([], 'native_model_execute', [const(program_id, i64)])
            launch.erase()
        for op in list(module.body.operations):
            if op.operation.name == 'air.channel':
                op.operation.erase()
        # Only constant/buffer initialization linalg is legal on the host
        # (checked before canonicalization could fold host arithmetic away).
        for op, _ in walk(module.operation):
            if op.name.startswith('math.') or (op.name.startswith('arith.') and
                    any(str(v.type).startswith('f') for v in op.results) and op.name != 'arith.constant'):
                raise ValueError('Model arithmetic outside device: ' + op.name)
            if op.name == 'func.call' and ir.FlatSymbolRefAttr(op.attributes['callee']).value not in {
                    'native_model_transfer', 'native_model_execute'}:
                raise ValueError('Unknown host function call')
            if op.name.startswith('linalg.') and op.name != 'linalg.yield':
                payload = [o for o, _ in walk(op) if o is not op]
                if any(o.name not in {'linalg.yield', 'arith.constant'} for o in payload):
                    raise ValueError('Non-initialization host linalg: ' + op.name)
        # Token loop-carried values left by host synchronization fold away.
        PassManager.parse('builtin.module(canonicalize,cse)').run(module.operation)
        for op, _ in list(walk(module.operation)):
            if op.name == 'air.wait_all' and not op.operands and not list(op.results[0].uses):
                op.erase()
        for op, _ in walk(module.operation):
            if op.name.startswith('air.'):
                raise ValueError('Unlowered AIR host operation: ' + op.name)
        PassManager.parse('builtin.module(canonicalize,cse,symbol-dce)').run(module.operation)
        module.operation.verify()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(str(module) + '\n')
        ir.Module.parse(output.read_text()).operation.verify()
    return {'schema_version': 1, 'source': str(source), 'source_sha256': _hash(source),
            'host_mlir': str(output), 'host_sha256': _hash(output), 'program_count': len(launches),
            'entry': '_mlir_ciface_native_model', 'input_shape': shapes[0], 'output_shape': shapes[1],
            'channels': channel_ids, 'transfers': transfers, 'host_buffer_reuse': host_memory,
            'cpu_model_arithmetic': False,
            'board_execution_performed': False}


def compile_host(source, out_dir, *, partition_manifest=None, clang=CLANG, timeout=180):
    """Generate verified control MLIR, native LLVM IR and a real ARM object."""
    out_dir = Path(out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=False)
    report = lower_host(source, out_dir / 'host.mlir', partition_manifest=partition_manifest)
    tools = ROOT / '.venv/bin'
    pipeline = 'builtin.module(convert-linalg-to-loops,expand-strided-metadata,lower-affine,convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts)'
    commands = [[str(tools / 'air-opt'), str(out_dir / 'host.mlir'), '--pass-pipeline=' + pipeline,
                 '-o', str(out_dir / 'host-llvm.mlir')],
                [str(tools / 'air-translate'), '--mlir-to-llvmir', str(out_dir / 'host-llvm.mlir'),
                 '-o', str(out_dir / 'host.ll')],
                # Position independent: the object links into the model module (model.so).
                [str(clang), '--target=aarch64-linux-gnu', '-O1', '-fPIC', '-c', str(out_dir / 'host-llvm18.ll'),
                 '-o', str(out_dir / 'host.o')]]
    report['commands'] = commands
    for i, command in enumerate(commands):
        if i == 2:
            compatible, changes = llvm18_compatible((out_dir / 'host.ll').read_text())
            (out_dir / 'host-llvm18.ll').write_text(compatible)
            report['llvm18_compatibility'] = changes
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        (out_dir / f'stage-{i}.stdout.log').write_text(process.stdout)
        (out_dir / f'stage-{i}.stderr.log').write_text(process.stderr)
        if process.returncode:
            raise RuntimeError(f'Host native stage {i} failed: {process.stderr[-4000:]}')
    obj = out_dir / 'host.o'
    data = obj.read_bytes()
    if data[:4] != b'\x7fELF' or int.from_bytes(data[18:20], 'little') != 183:
        raise ValueError('Host compiler did not produce an AArch64 ELF object')
    report.update(arm_object=str(obj), arm_object_sha256=_hash(obj), target_triple='aarch64-linux-gnu',
                  llvm_ir=str(out_dir / 'host.ll'), compatible_llvm_ir=str(out_dir / 'host-llvm18.ll'),
                  runtime_schedule_generated=True, position_independent=True)
    (out_dir / 'host-control.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def _set_holds(text, values):
    """Evaluate an affine integer set over symbols only (broadcast groups)."""
    match = re.fullmatch(r'affine_set<\(\)\[([^\]]*)\] : \((.*)\)>', text)
    if not match:
        raise ValueError('Unsupported affine.if condition: ' + text)
    names = [x.strip() for x in match.group(1).split(',')]
    env = dict(zip(names, values))
    for constraint in match.group(2).split(','):
        found = re.fullmatch(r'\s*(.*) (>=|==) 0\s*', constraint)
        if not found or not re.fullmatch(r'[\ss0-9+\-*()]*', found.group(1)):
            raise ValueError('Unsupported affine.if constraint: ' + constraint)
        value = eval(re.sub(r'\b(s\d+)\b', lambda t: f'({env[t.group(1)]})', found.group(1)), {'__builtins__': {}})
        if (found.group(2) == '>=' and value < 0) or (found.group(2) == '==' and value != 0):
            return False
    return True


def bind_transport_frames(projection_air, transport_report, host_report, outjson):
    """Prove AIR channel identities using native Linalg operand roles.

    Shape alone is never an identity: the entire ordered Linalg payload and its
    operand positions establish source-local-buffer to native-buffer mapping.
    Transport frame ordering is taken from the proven native DMA BD cycle.
    """
    from air import ir
    from compiler.ir_tools import walk

    projection_air, transport_report, host_report, outjson = map(Path,
        (projection_air, transport_report, host_report, outjson))
    transport = json.loads(transport_report.read_text())
    host = json.loads(host_report.read_text())
    if _hash(host['source']) != host['source_sha256']:
        raise ValueError('Host source hash mismatch')
    device_path = Path(transport['source'])
    if _hash(device_path) != transport['source_sha256']:
        raise ValueError('Transport source hash mismatch')
    if outjson.exists():
        raise FileExistsError(outjson)
    generic = subprocess.run([str(ROOT / '.venv/bin/aie-opt'), str(device_path),
                              '--mlir-print-op-generic'], capture_output=True, text=True, timeout=30)
    if generic.returncode:
        raise ValueError('Native AIE device verification failed: ' + generic.stderr[-2000:])
    with ir.Context() as context:
        context.load_all_available_dialects()
        # AIR and AIE wheels use separate MLIR contexts. AIE is verified by its
        # owning tool; only its generic serialization is inspected here.
        context.allow_unregistered_dialects = True
        source = ir.Module.parse(projection_air.read_text())
        device = ir.Module.parse(generic.stdout)
        source.operation.verify()
        device.operation.verify()
        herds = [op for op, _ in walk(source.operation) if op.name == 'air.herd']
        herd_by_name = {ir.StringAttr(h.attributes['sym_name']).value:h for h in herds}
        if not herds or len(herd_by_name) != len(herds):
            raise ValueError('Projection needs uniquely named source herds')

        def fingerprint(operation):
            """Include payload SSA edges and constants, excluding memref IDs."""
            result = []
            ids = {}
            for region in operation.regions:
                for block in region.blocks:
                    for arg in block.arguments:
                        ids[ir.Value.__hash__(arg)] = ('arg', arg.arg_number)
            for op, parents in walk(operation):
                attrs = tuple((key, str(op.attributes[key])) for key in op.attributes
                              if key != 'linalg.memoized_indexing_maps')
                operands = []
                for value in op.operands:
                    key = ir.Value.__hash__(value)
                    if op is operation:
                        operands.append(('external', str(value.type)))
                    elif key in ids:
                        operands.append(ids[key])
                    elif hasattr(value.owner, 'name') and value.owner.name == 'arith.constant':
                        operands.append(('constant', str(value.owner.attributes['value'])))
                    else:
                        raise ValueError('Linalg payload captures unsupported external value')
                result.append((parents, op.name, attrs, tuple(operands), tuple(str(v.type) for v in op.results)))
                for i, value in enumerate(op.results):
                    ids[ir.Value.__hash__(value)] = ('result', len(result) - 1, i)
            return result

        cores = {}
        for op, _ in walk(device.operation):
            if op.name == 'aie.core':
                tile = op.operands[0].owner
                cores[tuple(ir.IntegerAttr(tile.attributes[k]).value for k in ('col', 'row'))] = op
        records_by_herd = {name:[] for name in herd_by_name}
        host_endpoints = {}
        for endpoint, parents in walk(source.operation):
            if endpoint.name in {'air.channel.put', 'air.channel.get'} and 'air.herd' not in parents:
                name = ir.FlatSymbolRefAttr(endpoint.attributes['chan_name']).value
                fields = list(ir.DenseI32ArrayAttr(endpoint.attributes['operandSegmentSizes']))
                if name in host_endpoints:
                    raise ValueError('Ambiguous host channel endpoint in projection')
                host_endpoints[name] = ir.MemRefType(endpoint.operands[fields[0] + fields[1]].type)
        for endpoint, parents in walk(source.operation):
            if endpoint.name not in {'air.channel.get', 'air.channel.put'} or 'air.herd' not in parents:
                continue
            name = ir.FlatSymbolRefAttr(endpoint.attributes['chan_name']).value
            if name not in host_endpoints:
                continue  # An internal FIFO has no ARM transfer callback.
            herd = endpoint.parent
            while herd.name != 'air.herd': herd = herd.parent
            herd_name = ir.StringAttr(herd.attributes['sym_name']).value
            fields = list(ir.DenseI32ArrayAttr(endpoint.attributes['operandSegmentSizes']))
            # Async dependencies order the core's transfers; air-to-aie
            # implements them with locks, so only the frame is checked here.
            if any(fields[3:]):
                raise ValueError('Source local endpoint must have a static full-buffer frame')
            direction = int(endpoint.name == 'air.channel.put')
            prefix = 'src' if direction else 'dst'
            if any(len(ir.DenseI64ArrayAttr(endpoint.attributes[f'static_{prefix}_{field}']))
                   for field in ('offsets', 'sizes', 'strides')):
                raise ValueError('Source local endpoint must transfer its whole buffer')
            payload = endpoint.operands[fields[0] + fields[1]]
            # Broadcast groups guard their endpoints with affine.if on herd
            # ids; a core executes the branch its ids satisfy.
            guards, child, parent = [], endpoint, endpoint.parent
            while parent != herd:
                if parent.name == 'affine.if':
                    then = any(child == op.operation for op in parent.regions[0].blocks[0].operations)
                    guards.append((str(parent.attributes['condition']), list(parent.operands), then))
                child, parent = parent, parent.parent
            records_by_herd[herd_name].append((endpoint, fields, direction, payload, guards))
        all_keys = []
        evidence = []
        fifo = transport.get('device_dataflow') == 'objectfifo'
        stream_counts = []
        for stream in transport['frames']:
            herd_name = stream.get('herd_name')
            herd = herd_by_name.get(herd_name)
            core = cores.get(tuple(stream['core_tile']))
            if core is None or herd is None:
                raise ValueError('Transport stream does not identify the source herd/core')
            source_math = [op for op, _ in walk(herd) if op.name.startswith('linalg.') and op.name != 'linalg.yield']
            source_fingerprints = [fingerprint(op) for op in source_math]
            endpoint_records = records_by_herd[herd_name]
            local_id = stream.get('herd_local_id')
            herd_size = stream.get('herd_size')
            if not local_id or not herd_size or len(local_id) != 2 or len(herd_size) != 2:
                raise ValueError('Missing native core herd coordinate metadata')
            for key, expected in [('air.herd_local_id', local_id), ('air.herd_size', herd_size)]:
                if list(ir.DenseI64ArrayAttr(core.attributes[key])) != expected:
                    raise ValueError('Transport/native core herd metadata mismatch')
            native_math = [op for op, _ in walk(core) if op.name.startswith('linalg.') and op.name != 'linalg.yield']
            if [fingerprint(op) for op in native_math] != source_fingerprints:
                raise ValueError('Source/native ordered Linalg computation differs')
            buffers = {}
            # Kernels may compute on views (reshape/sub-box) of a local
            # buffer; identity is the underlying allocation on both sides.
            views = {'memref.collapse_shape', 'memref.expand_shape', 'memref.subview',
                     'memref.reinterpret_cast'}

            def root(value):
                while getattr(value.owner, 'name', None) in views:
                    value = value.owner.operands[0]
                return value
            for before, after in zip(source_math, native_math):
                for old, new in zip(before.operands, after.operands):
                    if not isinstance(old.type, ir.MemRefType):
                        continue
                    if (getattr(old.owner, 'name', None) in views) != (getattr(new.owner, 'name', None) in views):
                        raise ValueError('Source/native buffer view structure differs')
                    old, new = root(old), root(new)
                    while (getattr(new.owner, 'name', None) in
                           {'builtin.unrealized_conversion_cast', 'memref.cast'}):
                        cast = new.owner
                        if len(cast.operands) != 1:
                            raise ValueError('Ambiguous native FIFO buffer cast')
                        old_type, new_type = ir.MemRefType(cast.operands[0].type), ir.MemRefType(new.type)
                        if (list(old_type.shape) != list(new_type.shape) or
                            old_type.element_type != new_type.element_type or old_type.layout != new_type.layout):
                            raise ValueError('FIFO cast changes logical buffer layout')
                        new = cast.operands[0]
                    if new.owner.name != 'aie.buffer':
                        raise ValueError('Native computation operand is not a local AIE buffer')
                    name = ir.StringAttr(new.owner.attributes['sym_name']).value
                    key = ir.Value.__hash__(old)
                    if key in buffers and buffers[key] != name:
                        raise ValueError('Source operand maps to multiple native buffers')
                    buffers[key] = name
            argument_values = {ir.Value.__hash__(arg): value for arg, value in
                               zip(herd.regions[0].blocks[0].arguments[:4], [*local_id, *herd_size])}

            def evaluate(value):
                key = ir.Value.__hash__(value)
                if key in argument_values:
                    return argument_values[key]
                owner = value.owner
                if not hasattr(owner, 'name'):
                    raise ValueError('Data-dependent channel bundle index')
                if owner.name == 'arith.constant':
                    return ir.IntegerAttr(owner.attributes['value']).value
                if owner.name in {'arith.addi', 'arith.muli', 'arith.subi', 'arith.divui', 'arith.remui'}:
                    a, b = map(evaluate, owner.operands)
                    return {'arith.addi': lambda: a + b, 'arith.muli': lambda: a * b,
                            'arith.subi': lambda: a - b, 'arith.divui': lambda: a // b,
                            'arith.remui': lambda: a % b}[owner.name]()
                if owner.name == 'arith.index_cast':
                    return evaluate(owner.operands[0])
                raise ValueError('Unsupported channel bundle index expression: ' + owner.name)

            keys = []
            counts = []
            def executes(guards):
                return all(_set_holds(text, [evaluate(v) for v in operands]) == then
                           for text, operands, then in guards)

            for frame in stream['cycle_frames']:
                matches = [(endpoint, fields, payload) for endpoint, fields, direction, payload, guards in endpoint_records
                           if direction == stream['core_dma_direction'] and
                           buffers.get(ir.Value.__hash__(payload)) == frame['local_buffer'] and executes(guards)]
                if len(matches) != 1:
                    raise ValueError('DMA frame must match exactly one source channel operand role')
                endpoint, fields, payload = matches[0]
                typ = ir.MemRefType(payload.type)
                if (str(typ.element_type) != 'f32' or list(typ.shape) != frame['local_shape'] or
                        prod(typ.shape) != frame['elements'] or frame['local_offset_elements'] != 0):
                    raise ValueError('Source channel/local DMA payload differs')
                name = ir.FlatSymbolRefAttr(endpoint.attributes['chan_name']).value
                if name not in host_endpoints or (not fifo and frame['external_shape'] is not None and
                                                  list(host_endpoints[name].shape) != frame['external_shape']):
                    raise ValueError('Source host channel/external DMA buffer shape differs')
                declarations = [op for op, _ in walk(source.operation) if op.name == 'air.channel' and
                                ir.StringAttr(op.attributes['sym_name']).value == name]
                if len(declarations) != 1 or name not in host['channels']:
                    raise ValueError('Source channel is absent from host ABI')
                host_records = [item for item in host['transfers'] if item['channel'] == name and
                                item['direction'] == stream['core_dma_direction']]
                if len(host_records) != 1:
                    raise ValueError('Source channel has no unique host callback identity')
                dimensions = [ir.IntegerAttr(v).value for v in ir.ArrayAttr(declarations[0].attributes['size'])]
                indices = [evaluate(v) for v in list(endpoint.operands)[fields[0]:fields[0] + fields[1]]]
                if 'broadcast_shape' in declarations[0].attributes:
                    # A broadcast channel's herd-side endpoints use herd ids;
                    # its bundle is the group along each broadcast dimension.
                    # A single-bundle channel (one group per channel, as AIR
                    # specializes them) is bundle 0.
                    shape = [ir.IntegerAttr(v).value for v in ir.ArrayAttr(declarations[0].attributes['broadcast_shape'])]
                    if prod(dimensions) == 1:
                        indices = [0] * len(dimensions)
                    elif len(shape) != len(indices) or any(b % n for b, n in zip(shape, dimensions)):
                        raise ValueError('Unsupported broadcast channel shape')
                    else:
                        indices = [i * n // b for i, n, b in zip(indices, dimensions, shape)]
                if len(indices) != len(dimensions) or any(i < 0 or i >= n for i, n in zip(indices, dimensions)):
                    raise ValueError('Source bundle index outside declaration shape')
                bundle = 0
                for index, size in zip(indices, dimensions):
                    bundle = bundle * size + index
                keys.append({'channel_id': host['channels'][name], 'bundle_index': bundle})
                visits = 1
                if fifo:
                    parent = endpoint.parent
                    while parent != herd:
                        if parent.name != 'scf.for':
                            raise ValueError('FIFO transport needs static serial endpoint loops inside each herd')
                        lower, upper, step = [evaluate(v) for v in list(parent.operands)[:3]]
                        if step <= 0 or upper <= lower or len(parent.results):
                            raise ValueError('FIFO endpoint loop has unsupported trip count/iter_args')
                        visits *= (upper-lower+step-1)//step
                        parent = parent.parent
                counts.append(visits)
                evidence.append({'channel': name, 'bundle_indices': indices, 'local_buffer': frame['local_buffer'],
                                 'packet_id': frame['packet_id'], 'core_tile': stream['core_tile'],
                                 'identity_proof': 'ordered_linalg_payload_and_operand_roles'})
            all_keys.append(keys)
            if fifo and (len(counts) != 1 or len(stream['cycle_frames']) != 1):
                raise ValueError('Multi-buffer FIFO rotation needs explicit per-slot channel binding')
            stream_counts.append(counts[0] if counts else 1)
    # A resident reduction consumes K input tiles for one output tile. Preserve
    # that ratio in an execution window; do not require one output per input BD.
    from math import gcd
    from functools import reduce
    common = reduce(gcd, stream_counts)
    repetitions = [n//common for n in stream_counts] if fifo else [1]*len(all_keys)
    report = {'schema_version': 1, 'projection': str(projection_air), 'projection_sha256': _hash(projection_air),
              'transport_report_sha256': _hash(transport_report), 'host_report_sha256': _hash(host_report),
              'frame_keys': all_keys, 'stream_repetitions': repetitions,
              'frame_count_source': 'source_air_endpoint_loop_trip_counts' if fifo else 'native_dma_cycle',
              'evidence': evidence, 'board_execution_performed': False}
    outjson.parent.mkdir(parents=True, exist_ok=True)
    outjson.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--partition-manifest', type=Path)
    args = parser.parse_args()
    result = compile_host(args.source, args.out_dir, partition_manifest=args.partition_manifest)
    print(f"Native ARM host control generated for {result['program_count']} device programs")


if __name__ == '__main__':
    main()
