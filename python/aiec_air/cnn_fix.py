"""Attach native Linalg filter markers and select per-operation AIE1 tiles."""

import hashlib
from math import prod
from pathlib import Path

from aiec_air.memory_fix import _divisors, plan_conv
from compiler.target_spec import CORE_STATIC_BYTES


MARKER = "__internal_linalg_transform__"


def _walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _structured_ops(op, inside_structured=False):
    """Payload index/yield belong to their parent, not separate tile jobs."""
    structured = op.name.startswith("linalg.") and op.name not in {"linalg.index", "linalg.yield"}
    if op.name in {"linalg.index", "linalg.yield"} and not inside_structured:
        raise ValueError(f"Unexpected Linalg payload outside a structured operation: {op.name}")
    if structured:
        if inside_structured:
            raise ValueError(f"Nested structured CNN operations require a separate policy: {op.name}")
        yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _structured_ops(child.operation, inside_structured or structured)


def _shape(ir, value):
    if not isinstance(value.type, ir.MemRefType):
        if str(value.type) == "f32":
            return None
        raise ValueError(f"CNN policy requires FP32 memrefs/scalars, got {value.type}")
    typ = ir.MemRefType(value.type)
    shape = list(typ.shape)
    if str(typ.element_type) != "f32" or any(x <= 0 for x in shape):
        raise ValueError(f"CNN policy requires static positive FP32 memrefs: {typ}")
    return shape


def _align(n, width=64):
    return (n + width - 1) // width * width


def _constant_init(ir, op):
    """Only scalar-constant fills, never arbitrary zero-input computation."""
    if op.name not in {"linalg.fill", "linalg.map", "linalg.generic"}:
        return False
    operands = list(op.operands)
    memrefs = [x for x in operands if isinstance(x.type, ir.MemRefType)]
    if len(memrefs) != 1:
        return False
    if op.name == "linalg.fill":
        return getattr(operands[0].owner, "name", None) == "arith.constant"
    if not op.regions or not op.regions[0].blocks:
        return False
    block = op.regions[0].blocks[0]
    body = [x.operation for x in block.operations]
    if any(x.name not in {"arith.constant", "linalg.yield"} for x in body):
        return False
    terminator = body[-1]
    if terminator.name != "linalg.yield" or len(terminator.operands) != 1:
        return False
    value = terminator.operands[0]
    if getattr(value.owner, "name", None) == "arith.constant":
        return True
    if isinstance(value, ir.BlockArgument):
        index = value.arg_number
        return (index < len(operands) and str(operands[index].type) == "f32"
                and getattr(operands[index].owner, "name", None) == "arith.constant")
    return False


def _affine(ir, expr, dims):
    """Evaluate monotone affine indexing extents; reject other maps explicitly."""
    if isinstance(expr, ir.AffineDimExpr):
        return dims[expr.position]
    if isinstance(expr, ir.AffineConstantExpr):
        if expr.value < 0:
            raise ValueError("Negative affine indexing needs a separate boundary policy")
        return expr.value
    if isinstance(expr, ir.AffineAddExpr):
        return _affine(ir, expr.lhs, dims) + _affine(ir, expr.rhs, dims)
    if isinstance(expr, ir.AffineMulExpr):
        return _affine(ir, expr.lhs, dims) * _affine(ir, expr.rhs, dims)
    raise ValueError(f"Unsupported CNN indexing expression: {expr}")


def _generic_plan(ir, op, *, l1_bytes=32768, stack_bytes=2048,
                  tiling_strategy='baseline', resources=None):
    for payload in _walk(op):
        if payload is op:
            continue
        # Region-captured buffers are not represented by indexing_maps and are
        # not promoted by the operand tile policy. Never budget only the output.
        if any(isinstance(value.type, ir.MemRefType) for value in payload.operands):
            raise ValueError("CNN structured payload captures or directly accesses a memref "
                             f"via {payload.name}; explicit indexed operands are required "
                             "before AIE1 tiling (nearest-upsample lowering is not yet supported)")
    maps = [ir.AffineMapAttr(x).value for x in op.attributes["indexing_maps"]]
    rank = maps[0].n_dims
    shapes = [_shape(ir, value) for value in op.operands]
    bounds = [None] * rank
    for mapping, shape in zip(maps, shapes):
        if mapping.n_symbols:
            raise ValueError("Symbolic CNN indexing maps are unsupported")
        if shape is None:
            continue
        if len(mapping.results) != len(shape):
            raise ValueError("Operand rank does not match indexing map")
        for expr, extent in zip(mapping.results, shape):
            if isinstance(expr, ir.AffineDimExpr):
                axis = expr.position
                if bounds[axis] is not None and bounds[axis] != extent:
                    raise ValueError("Inconsistent static loop extent across CNN operands")
                bounds[axis] = extent
    if not rank or any(x is None for x in bounds):
        raise ValueError("Cannot infer static CNN loop bounds from indexing maps")
    # Check the complete loop domain before selecting bounded subtiles.
    for mapping, shape in zip(maps, shapes):
        if shape is not None:
            for expr, extent in zip(mapping.results, shape):
                if _affine(ir, expr, [x - 1 for x in bounds]) >= extent:
                    raise ValueError("CNN indexing map exceeds operand bounds")
    tile = list(bounds)
    # '#linalg.iterator_type<parallel>' -> 'parallel'
    iterator_types = ['reduction' if 'reduction' in str(x) else 'parallel' for x in op.attributes["iterator_types"]]
    reduction = 'reduction' in iterator_types

    def memory_for(sizes):
        allocations = []
        for mapping, shape in zip(maps, shapes):
            if shape is None:
                continue
            footprint = [_affine(ir, expr, [x - 1 for x in sizes])
                         - _affine(ir, expr, [0] * rank) + 1 for expr in mapping.results]
            if footprint:
                footprint[-1] = _align(footprint[-1], 8)
            allocations.append(_align(prod(footprint) * 4))
        accumulator = allocations[-1] if reduction else 0
        # The tile's module also holds the core's static data (target_spec).
        return dict(operand_slots_bytes=allocations, slots=2, stack=stack_bytes,
                    scratch=256, accumulator=accumulator, static=CORE_STATIC_BYTES,
                    total=sum(allocations) * 2 + accumulator + stack_bytes + 256 + CORE_STATIC_BYTES,
                    l1_limit=l1_bytes, physical_allocation_verified=False)

    if tiling_strategy == 'automatic':
        from aiec_air.automatic_fix import select_schedule, vector_axis
        from aiec_air.residency_fix import INIT
        # AIR's indexing maps are the semantic source; the scalar region is
        # preserved. Output traffic includes reads only when the payload uses
        # the corresponding old-output argument (e.g. a reduction accumulator).
        block = op.regions[0].blocks[0]
        output_count = list(op.attributes['operandSegmentSizes'])[-1]
        def logical_bytes_for(sizes):
            traffic = []
            for i, (mapping, shape) in enumerate(zip(maps, shapes)):
                if shape is None:
                    continue
                footprint = [_affine(ir, expr, [x-1 for x in sizes])
                             - _affine(ir, expr, [0]*rank) + 1 for expr in mapping.results]
                uses_old_output = (i >= len(maps)-output_count and bool(list(block.arguments[i].uses)))
                traffic.append(prod(footprint)*4*(2 if uses_old_output else 1))
            return traffic
        kinds = iterator_types
        payload = sum(child.operation.name.startswith(('arith.', 'math.'))
                      and child.operation.name != 'arith.constant' for child in block.operations)
        if resources is None:
            raise ValueError('Automatic scheduling requires a native target/platform contract')
        initializers = list(op.attributes[INIT]) if INIT in op.attributes else [None]*output_count
        return select_schedule(bounds, kinds, memory_for, logical_bytes_for,
            resources=resources, vector_axis=vector_axis(ir, op, maps, kinds),
            payload_operations=max(1,payload),
            input_buffers=sum(s is not None for s in shapes[:-output_count]),
            output_buffers=output_count,
            read_output_buffers=sum(bool(list(a.uses)) and not isinstance(init, ir.FloatAttr)
                                    for a, init in zip(list(block.arguments)[-output_count:], initializers)))

    while True:
        memory = memory_for(tile)
        if memory["total"] <= l1_bytes:
            break
        candidates = []
        for axis, extent in enumerate(tile):
            if extent > 1:
                candidate = list(tile)
                candidate[axis] = max(1, extent // 2)
                # Retain a SIMD8 width when the shrunk extent permits it.
                if axis == rank - 1 and candidate[axis] >= 8:
                    candidate[axis] = candidate[axis] // 8 * 8
                # Native AIE1 promotion needs statically sized subviews. Choose
                # a factor of the original loop bound, not the current tile.
                candidate[axis] = max(x for x in _divisors(bounds[axis])
                                      if x <= candidate[axis])
                candidates.append((memory_for(candidate)["total"], axis, candidate))
        if not candidates:
            raise ValueError(f"No legal AIE1 CNN tile within {l1_bytes}-byte live buffer budget")
        tile = min(candidates, key=lambda x: (x[0], -x[1]))[2]
    valid = [{"extent": b, "tile": t, "last": (b - 1) % t + 1}
             for b, t in zip(bounds, tile)]
    return tile, dict(memory=memory, valid_extent=valid,
                      backend_constraints=dict(no_dynamic_tail=True),
                      loop_extents=bounds, iterator_types=iterator_types,
                      physical_allocation_verified=False)


def _epilogue_inputs(ir, conv) -> int:
    """Other inputs of the elementwise op aiec-fuse-conv-blocks (mlir/lib/Transform/AIECFixes/FuseConvBlocks_fix.cpp) would make
    the convolution's epilogue: its output's single all-parallel identity-map
    reader with buffers like the output. The resident plan reserves their
    tiles and streams."""
    out = conv.operands[2]
    readers = [use.owner for use in out.uses
               if use.owner != conv and not (use.owner.name == "memref.copy" and use.owner.operands[1] == out)]
    if len(readers) != 1 or readers[0].name != "linalg.generic":
        return 0
    reader = readers[0]
    count = list(ir.DenseI32ArrayAttr(reader.attributes["operandSegmentSizes"]))
    operands = list(reader.operands)
    maps = [ir.AffineMapAttr(m).value for m in reader.attributes["indexing_maps"]]
    rank = ir.MemRefType(out.type).rank
    if count[1] != 1 or any(v.type != out.type for v in operands) or \
            any("parallel" not in str(k) for k in reader.attributes["iterator_types"]) or \
            any(m.n_dims != rank or [str(r) for r in m.results] != [f"d{i}" for i in range(rank)] for m in maps):
        return 0
    return sum(1 for v in operands[:-1] if v != out)


def _transpose_channel_group(ir, conv):
    """Return the inner channel extent when a safe transpose can be fused.

    This recognizes the IR's actual buffer, view and consumer relationships;
    it does not use a model or shape allowlist. The later fusion pass still
    proves its generated tile offsets and all resource constraints.
    """
    out = conv.operands[2]
    if getattr(out.owner, 'name', None) != 'memref.alloc':
        return None
    views = []
    for use in out.uses:
        user = use.owner
        if user == conv:
            continue
        if (user.name == 'memref.copy' and user.operands[1] == out and
                user.operation.block == conv.block and user.operation.is_before_in_block(conv)):
            continue
        if user.name == 'memref.expand_shape':
            views.append(user)
            continue
        return None
    if len(views) != 1:
        return None
    expand = views[0]
    if expand.operation.block != conv.block:
        return None
    groups = [[ir.IntegerAttr(v).value for v in ir.ArrayAttr(a)]
              for a in ir.ArrayAttr(expand.attributes['reassociation'])]
    if groups != [[0], [1, 2], [3], [4]]:
        return None
    expanded = expand.results[0]
    shape = ir.MemRefType(expanded.type).shape
    if any(d <= 0 for d in shape):
        return None
    users = [use.owner for use in expanded.uses]
    if len(users) != 1 or users[0].name not in {'linalg.transpose', 'linalg.generic'}:
        return None
    transpose = users[0]
    if transpose.name == 'linalg.generic':
        counts = list(ir.DenseI32ArrayAttr(transpose.attributes['operandSegmentSizes']))
        if counts != [1, 1] or transpose.operands[0] != expanded:
            return None
        rank = ir.MemRefType(expanded.type).rank
        maps = [ir.AffineMapAttr(m).value for m in transpose.attributes['indexing_maps']]
        if (len(maps) != 2 or maps[1].n_dims != rank or
                [str(r) for r in maps[1].results] != [f'd{i}' for i in range(rank)] or
                any('parallel' not in str(x) for x in transpose.attributes['iterator_types']) or
                not all(isinstance(r, ir.AffineDimExpr) for r in maps[0].results) or
                sorted(r.position for r in maps[0].results) != list(range(rank))):
            return None
        out_shape = ir.MemRefType(transpose.operands[1].type).shape
        if len(out_shape) != rank or any(shape[i] != out_shape[maps[0].results[i].position] for i in range(rank)):
            return None
        block = transpose.regions[0].blocks[0]
        body = [o.operation for o in block.operations]
        if (len(body) != 1 or body[0].name != 'linalg.yield' or
                body[0].operands[0] != block.arguments[0] or list(block.arguments[-1].uses)):
            return None
    if transpose.operation.block != conv.block:
        return None
    dest = transpose.operands[1]
    if getattr(dest.owner, 'name', None) != 'memref.alloc' or dest.owner.operation.block != conv.block:
        return None
    if not dest.owner.operation.is_before_in_block(conv) and list(dest.owner.operands):
        return None
    for use in dest.uses:
        user = use.owner
        if user == transpose:
            continue
        op = user.operation
        while op is not None and op.block != conv.block:
            op = op.parent and op.parent.operation
        if op is None or op.is_before_in_block(transpose):
            return None
    return shape[2]


def prepare_cnn(source: Path, output: Path, *, max_cores=4, herd_axis_limits=(2, 2),
                l1_bytes=32768, stack_bytes=2048, tiling_strategy='baseline', resources=None,
                conv_distribution='n_oc', stream_budget=None, conv_dataflow='tiled', max_stages=1) -> dict:
    """Read native bufferized IR and annotate each supported Linalg operation.

    The output keeps named operations and their actual regions. A separate native
    generalization is used for indexing-map inspection and marker survival checks.
    """
    from air import ir
    from air.passmanager import PassManager
    from aiec_air.memory_fix import core_budget, plan_conv_resident, select_axes_herd, select_herd
    from compiler.target_spec import BUFFER_DESCRIPTORS, LOCKS_PER_TILE

    if tiling_strategy not in {'baseline', 'automatic'}:
        raise ValueError('Unknown structured tiling strategy')
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError(f"CNN policy output already exists: {output}")
    with ir.Context() as context:
        context.load_all_available_dialects()
        text = source.read_text()
        if tiling_strategy == 'automatic':
            from aiec_air.residency_fix import annotate_initializers
            text = annotate_initializers(text)
        module = ir.Module.parse(text)
        module.operation.verify()
        originals = {}
        host_initialization = []
        for op in _structured_ops(module.operation):
            if _constant_init(ir, op):
                host_initialization.append(dict(source_op=op.name,
                                                reason="constant buffer initialization retained in mixed IR"))
                continue
            supported = {"linalg.generic", "linalg.map", "linalg.fill", "linalg.broadcast",
                         "linalg.transpose", "linalg.copy", "linalg.conv_2d_nchw_fchw",
                         "linalg.depthwise_conv_2d_nchw_chw",
                         "linalg.matmul", "linalg.batch_matmul"}
            if op.name not in supported and not op.name.startswith("linalg.pooling_"):
                raise ValueError(f"CNN policy does not support {op.name}")
            marker = f"aie1_op_{len(originals)}"
            op.attributes[MARKER] = ir.StringAttr.get(marker)
            op.attributes["aie1.source_op"] = ir.StringAttr.get(op.name)
            originals[marker] = op
        if not originals and not host_initialization:
            raise ValueError("CNN policy input contains no supported Linalg operations")
        inspection = ir.Module.parse(str(module))
        PassManager.parse("builtin.module(linalg-generalize-named-ops)").run(inspection.operation)
        generalized = {ir.StringAttr(op.attributes[MARKER]).value: op
                       for op in _walk(inspection.operation) if MARKER in op.attributes}
        policies = []
        for marker, op in originals.items():
            if marker not in generalized or generalized[marker].name != "linalg.generic":
                raise ValueError(f"Native generalization did not preserve marker {marker}")
            if tiling_strategy == 'automatic':
                tile, plan = _generic_plan(ir, generalized[marker], l1_bytes=l1_bytes,
                    stack_bytes=stack_bytes, tiling_strategy=tiling_strategy, resources=resources)
                herd = plan['core_group']['herd_size']
            elif op.name == "linalg.conv_2d_nchw_fchw":
                shapes = [_shape(ir, x) for x in op.operands]
                if len(shapes) != 3 or any(x is None for x in shapes):
                    raise ValueError("Conv requires input, weight and output memrefs")
                stride = tuple(ir.DenseIntElementsAttr(op.attributes["strides"]))
                dilation = tuple(ir.DenseIntElementsAttr(op.attributes["dilations"]))
                if conv_dataflow == "resident":
                    options = dict(stride=stride, dilation=dilation, l1_bytes=l1_bytes, stack_bytes=stack_bytes,
                                   max_cores=max_cores, herd_axis_limits=herd_axis_limits, stream_budget=stream_budget,
                                   epilogue_inputs=_epilogue_inputs(ir, op), max_stages=max_stages)
                    plan = plan_conv_resident(shapes[0], shapes[1], **options)
                    group = _transpose_channel_group(ir, op)
                    if group and group % plan['tile']['oc']:
                        try:
                            aligned = plan_conv_resident(shapes[0], shapes[1], output_channel_group=group, **options)
                        except ValueError:
                            aligned = None
                        # The cost model's traffic terms use bytes/4. Bypassing
                        # transpose removes its read and write of the output.
                        saved = prod(shapes[2]) * 4 * 2 / 4
                        if aligned is not None and aligned['estimates']['score'] - plan['estimates']['score'] <= saved:
                            aligned['layout_fusion'] = {'channel_group': group,
                                'unaligned_score': plan['estimates']['score'],
                                'aligned_score': aligned['estimates']['score'],
                                'avoided_transfer_bytes': prod(shapes[2]) * 4 * 2}
                            plan = aligned
                else:
                    plan = plan_conv(shapes[0], shapes[1], stride=stride, dilation=dilation,
                                     mapping=conv_distribution, max_cores=max_cores, divisible_tiles=True,
                                     herd_axis_limits=herd_axis_limits, l1_bytes=l1_bytes, stack_bytes=stack_bytes,
                                     stream_budget=stream_budget)
                if plan["output_shape"] != shapes[2]:
                    raise ValueError("Conv output shape differs from static convolution contract")
                tile = [plan["tile"][x] for x in ("batch", "oc", "oh", "ow", "ic", "kh", "kw")]
                herd = plan["core_group"]["herd_size"]
            else:
                tile, plan = _generic_plan(ir, generalized[marker], l1_bytes=l1_bytes, stack_bytes=stack_bytes)
                bounds = plan["loop_extents"]
                extents = bounds[:2]+[1]*(2-len(bounds[:2]))
                tiles = tile[:2]+[1]*(2-len(tile[:2]))
                # Each core loads a tile of every operand its payload reads
                # (the output too, when it accumulates; not a pooling window,
                # which only gives its shape) and stores its output tile.
                body = generalized[marker].regions[0].blocks[0]
                inputs = sum(1 for arg in body.arguments if list(arg.uses))
                cores, limits = core_budget(inputs, 1, max_cores, herd_axis_limits, stream_budget)
                # A herd column's transfers share its shim tile, whose DMA has
                # BUFFER_DESCRIPTORS descriptors and LOCKS_PER_TILE locks, one of
                # each per external buffer (nothing is broadcast here).
                limits[1] = max(1, min(limits[1], min(BUFFER_DESCRIPTORS, LOCKS_PER_TILE) // (inputs + 1)))
                kinds = plan['iterator_types']
                if len(bounds) >= 2:
                    # Any two parallel loops, the larger group on columns; a
                    # native interchange puts them first (loop_permutation).
                    counts = [b // t if k == 'parallel' else 1 for b, t, k in zip(bounds, tile, kinds)]
                    herd, axes = select_axes_herd(counts, cores, limits)
                    order = list(axes) + [d for d in range(len(bounds)) if d not in axes]
                    plan['core_group'] = dict(cores=prod(herd), herd_size=herd, axes=list(axes),
                                              loop_permutation=order, placement_verified=False)
                else:
                    limits = [limit if i < len(kinds) and kinds[i] == 'parallel' else 1
                              for i, limit in enumerate(limits)]
                    herd = select_herd(extents, tiles, cores, limits)
            policies.append(dict(marker=marker, source_op=op.name, tile_sizes=tile,
                                 herd_size=herd, memory_plan=plan))
        module.operation.verify()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(str(module))
    return dict(operations=policies, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                host_initialization=host_initialization,
                native_generalization_markers_verified=True)
