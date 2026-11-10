"""FP32 Conv policy estimates for AIE1; this module does not execute a model."""

from itertools import product
from compiler.target_spec import CORE_STATIC_BYTES, DMA_CHANNELS_PER_DIRECTION
from math import ceil, isqrt, prod


def _ints(value, count, name, minimum=1):
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{name} must contain {count} static integers")
    if any(type(x) is not int or x < minimum for x in value):
        raise ValueError(f"{name} entries must be integers >= {minimum}")
    return tuple(value)


def _align(value, alignment=64):
    return ceil(value / alignment) * alignment


def _candidates(extent):
    values = {1, extent}
    power = 2
    while power < extent:
        values.add(power)
        power *= 2
    return sorted(values)


def _divisors(extent):
    """All positive factors, including useful non-power-of-two channel tiles."""
    values = set()
    for factor in range(1, isqrt(extent) + 1):
        if extent % factor == 0:
            values.update((factor, extent // factor))
    return sorted(values)


def select_herd(extents, tiles, max_cores=4, axis_limits=(2, 2)):
    """Equal division for AIR's existing first-two-parallel-loop mapping."""
    extents = _ints(extents, 2, 'herd extents')
    tiles = _ints(tiles, 2, 'herd tiles')
    limits = _ints(axis_limits, 2, 'herd axis limits')
    if type(max_cores) is not int or max_cores < 1:
        raise ValueError('max_cores must be positive')
    choices = [[x for x in _divisors(e) if x <= limit and ceil(e/t) % x == 0]
               for e, t, limit in zip(extents, tiles, limits)]
    return list(max(((a, b) for a in choices[0] for b in choices[1] if a*b <= max_cores),
                    key=lambda h: (prod(h), h[0], h[1])))


def select_axes_herd(counts, max_cores, axis_limits):
    """The two loops (of N, OC, OH, OW task counts) and the core group on
    them with the most cores: each group size divides its loop's task count
    (static subviews), within the herd axis limits and the core count. Ties
    keep the earlier loops (the generalized order). Returns (sizes, axes)."""
    limits = _ints(axis_limits, 2, 'herd axis limits')
    best = None
    for a in range(len(counts)):
        for b in range(len(counts)):
            if a == b:
                continue
            for ga in _divisors(counts[a]):
                for gb in _divisors(counts[b]):
                    if ga > limits[0] or gb > limits[1] or ga * gb > max_cores:
                        continue
                    key = (ga * gb, -min(a, b), -max(a, b), -a)
                    if best is None or key > best[0]:
                        best = (key, [ga, gb], (a, b))
    return best[1], best[2]


def plan_conv(input_shape, weight_shape, *, stride=(1, 1), padding=(0, 0),
              dilation=(1, 1), groups=1, l1_bytes=32768, stack_bytes=2048,
              double_buffer=True, max_cores=16, mapping="all", divisible_tiles=False,
              herd_axis_limits=(2, 2), stream_budget=None):
    """Choose named NCHW/OIHW tiles under a conservative per-core L1 budget.

    Shapes denote FP32 tensors. Returned JSON-compatible data is policy/report
    metadata, not execution IR or proof of physical bank/DMA legality.
    """
    n, ci, ih, iw = _ints(input_shape, 4, "input_shape")
    co, cig, kh, kw = _ints(weight_shape, 4, "weight_shape")
    sh, sw = _ints(stride, 2, "stride")
    ph, pw = _ints(padding, 2, "padding", 0)
    dh, dw = _ints(dilation, 2, "dilation")
    for name, value, minimum in (("groups", groups, 1), ("l1_bytes", l1_bytes, 1),
                                 ("stack_bytes", stack_bytes, 0), ("max_cores", max_cores, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if type(double_buffer) is not bool:
        raise ValueError("double_buffer must be boolean")
    if type(divisible_tiles) is not bool:
        raise ValueError("divisible_tiles must be boolean")
    if mapping not in {"all", "n_oc", "spatial"}:
        raise ValueError("mapping must be 'all', 'n_oc' or 'spatial'")
    _ints(herd_axis_limits, 2, 'herd axis limits')
    if l1_bytes > 32768:
        raise ValueError("AIE1 per-core L1 budget cannot exceed 32768 bytes")
    if ci % groups or co % groups or cig != ci // groups:
        raise ValueError("groups must divide input/output channels and match weight IC")
    oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("Conv output spatial dimensions must be positive")
    cog = co // groups
    slots = 2 if double_buffer else 1
    best = None
    legal_count = 0
    min_bytes = None
    candidates = _divisors if divisible_tiles else _candidates
    for toc, tic, th, tw in product(*map(candidates, (cog, cig, oh, ow))):
        halo_h = (th - 1) * sh + (kh - 1) * dh + 1
        halo_w = (tw - 1) * sw + (kw - 1) * dw + 1
        # Round W to SIMD8 storage, then each buffer to a 64-byte boundary.
        ib = _align(tic * halo_h * _align(halo_w, 8) * 4)
        wb = _align(toc * tic * kh * _align(kw, 8) * 4)
        ob = _align(toc * th * _align(tw, 8) * 4)
        chunks = ceil(cig / tic)
        ab = ob if chunks > 1 else 0
        memory = dict(input=ib * slots, weight=wb * slots, output=ob * slots,
                      accumulator=ab, stack=_align(stack_bytes), scratch=256,
                      l1_limit=l1_bytes, slots=slots)
        total = sum(memory[k] for k in
                    ("input", "weight", "output", "accumulator", "stack", "scratch"))
        min_bytes = total if min_bytes is None else min(min_bytes, total)
        if total > l1_bytes:
            continue
        legal_count += 1
        tasks = n * groups * ceil(cog / toc) * ceil(oh / th) * ceil(ow / tw)
        herd_size = axes = None
        if mapping == "n_oc":
            # Native air-linalg-codegen distributes the first two loops of
            # the generic (N, OC as generalized); spatial tiles and groups are
            # not counted as parallel workers.
            herd_size = select_herd((n, cog), (1, toc), max_cores, herd_axis_limits)
            cores = prod(herd_size)
        elif mapping == "spatial":
            # Any two of N, OC, OH, OW; a native interchange then puts them
            # first for air-linalg-codegen (loop_permutation below).
            # Each core streams X, W and C in (the accumulation reads C) and C
            # out; the program's transport bounds how many streams there are
            # (stream_budget: {"total": n} for packet IDs, {"inputs": n,
            # "outputs": n} for circuits).
            budget = max_cores
            if stream_budget:
                if "total" in stream_budget:
                    budget = min(budget, stream_budget["total"] // 4)
                budget = min(budget, stream_budget.get("inputs", budget * 3) // 3,
                             stream_budget.get("outputs", budget))
            herd_size, axes = select_axes_herd((n, ceil(cog / toc), ceil(oh / th), ceil(ow / tw)),
                                               max(1, budget), herd_axis_limits)
            cores = prod(herd_size)
        else:
            cores = min(tasks, max_cores, 400)
        # Unpadded logical estimates count halo re-fetch and no inter-task
        # weight residency. Boundary zero padding is conservatively counted.
        input_traffic = tasks * chunks * tic * halo_h * halo_w * 4
        weight_traffic = tasks * chunks * toc * tic * kh * kw * 4
        output_traffic = n * co * oh * ow * 4
        traffic = input_traffic + weight_traffic + output_traffic
        macs = n * co * oh * ow * cig * kh * kw
        work = tasks * chunks
        vector_util = tw / _align(tw, 8)
        # Heuristic work units, not cycles/bandwidth or a measured speedup.
        score = macs / (8 * cores * vector_util) + traffic / 4 + work * 64
        key = (score, total, -tw, -tic)
        if best is None or key < best[0]:
            tile = dict(batch=1, oc=toc, ic=tic, oh=th, ow=tw, kh=kh, kw=kw)
            extents = dict(batch=n, oc=cog, ic=cig, oh=oh, ow=ow, kh=kh, kw=kw)
            valid = {axis: {"extent": extent, "tile": tile[axis],
                            "last": (extent - 1) % tile[axis] + 1,
                            "count": ceil(extent / tile[axis])}
                     for axis, extent in extents.items()}
            memory["total"] = total
            best = (key, dict(tile=tile, valid_extent=valid,
                              output_shape=[n, co, oh, ow], groups=groups,
                              channel_chunks=chunks, memory=memory,
                              input_halo=[halo_h, halo_w],
                              logical_transfer_bytes=traffic,
                              estimates=dict(input_transfer_bytes=input_traffic,
                                             weight_transfer_bytes=weight_traffic,
                                             output_transfer_bytes=output_traffic,
                                             macs=macs, output_tasks=tasks,
                                             reduction_tasks=work, score=score,
                                             vector_width=8, width_utilization=vector_util),
                              core_group=dict(cores=cores, columns=min(50, cores),
                                              rows=ceil(cores / 50),
                                              mapping=mapping, herd_size=herd_size, axes=axes,
                                              loop_permutation=None if axes is None else
                                              list(axes) + [d for d in range(7) if d not in axes],
                                              placement_verified=False)))
    if best is None:
        raise ValueError(f"No legal AIE1 Conv tile: minimum candidate needs {min_bytes} "
                         f"bytes, L1 budget is {l1_bytes}")
    result = best[1]
    result["estimates"]["legal_candidates"] = legal_count
    result["logical_tensor_bytes"] = (prod(input_shape) + prod(weight_shape)
                                      + prod(result["output_shape"])) * 4
    result["dtype"] = "f32"
    result["backend_constraints"] = dict(no_dynamic_tail=divisible_tiles)
    result["physical_allocation_verified"] = False
    return result


def core_budget(inputs, outputs, max_cores, herd_axis_limits, stream_budget=None):
    """Cores and herd axis limits for cores that each receive `inputs` and
    send `outputs` kinds of transfer: the transport's streams (packet IDs, or
    PLIO channels for circuits) bound the cores, and a core with more input
    kinds than its tile DMA's S2MM channels takes the north neighbour's DMA
    and memory (aie1_tile.plan_channels), leaving only every other row."""
    budget = max_cores
    if stream_budget:
        if "total" in stream_budget:
            budget = min(budget, stream_budget["total"] // (inputs + outputs))
        budget = min(budget, stream_budget.get("inputs", budget * inputs) // max(1, inputs),
                     stream_budget.get("outputs", budget * outputs) // max(1, outputs))
    limits = list(herd_axis_limits)
    if inputs > DMA_CHANNELS_PER_DIRECTION:
        limits[1] = max(1, limits[1] // 2)
    return max(1, budget), limits


def plan_conv_resident(input_shape, weight_shape, *, stride=(1, 1), dilation=(1, 1), l1_bytes=32768,
                       stack_bytes=2048, max_cores=16, herd_axis_limits=(2, 2), stream_budget=None,
                       epilogue_inputs=0, max_stages=1, output_channel_group=None):
    """A convolution's per-core dataflow with the operands kept where they are
    reused (weight-stationary, output-resident; as Triton-XDNA promotes a
    matmul's accumulator outside its K loop and the GPU/TPU convolutions keep
    the accumulator in registers/VMEM across the reduction):

        forall cores (two of N, OC, OH, OW):
          for OC tiles:        W tile (all input channels) into L1, if it fits
            for OH, OW tiles:  C tile into L1, accumulated there
              for IC chunks:   X chunk into L1 (and W chunk, when W does not fit)

    Tiles divide their extents (static subviews). Each candidate's L1 use is
    its single buffers (with an output-sized tile per epilogue input) plus
    stack; the choice minimizes the same work and
    traffic estimate as plan_conv. ``output_channel_group`` optionally searches
    candidates whose output-channel tile divides that layout group; it does
    not change the default search. Returns the tile, the herd (sizes and axes),
    whether W is resident, and the estimates."""
    n, ci, ih, iw = _ints(input_shape, 4, "input_shape")
    co, cig, kh, kw = _ints(weight_shape, 4, "weight_shape")
    if output_channel_group is not None and (type(output_channel_group) is not int or
            output_channel_group <= 0 or co % output_channel_group):
        raise ValueError('Output channel group must be a positive divisor of output channels')
    sh, sw = _ints(stride, 2, "stride")
    dh, dw = _ints(dilation, 2, "dilation")
    if cig != ci:
        raise ValueError("Resident conv dataflow supports groups=1")
    oh = (ih - dh * (kh - 1) - 1) // sh + 1
    ow = (iw - dw * (kw - 1) - 1) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("Conv output spatial dimensions must be positive")
    # Each core receives three kinds of transfer (W per OC tile, the
    # accumulator's initial value per spatial tile, X per IC chunk), plus a
    # tile of each fused epilogue input, and sends one: its streams bound the cores the program's transport can feed
    # (packet IDs, or PLIO channels for circuits).
    inputs, outputs = 3 + epilogue_inputs, 1
    budget, herd_axis_limits = core_budget(inputs, outputs, max_cores, herd_axis_limits, stream_budget)
    best = None
    for toc, tic, th, tw in product(*map(_divisors, (co, ci, oh, ow))):
        if output_channel_group is not None and output_channel_group % toc:
            continue
        halo_h, halo_w = (th - 1) * sh + (kh - 1) * dh + 1, (tw - 1) * sw + (kw - 1) * dw + 1
        x = _align(tic * halo_h * halo_w * 4)
        c = _align(toc * th * tw * 4)
        w_full, w_chunk = _align(toc * ci * kh * kw * 4), _align(toc * tic * kh * kw * 4)
        # The tile's own module after the stack and the core's static data,
        # the budget the tile DMA channel planner places these buffers in.
        room = l1_bytes - CORE_STATIC_BYTES
        chunks = ci // tic
        for stages in range(1, min(max_stages, 2) + 1):
            # Two stages: the IC loop's buffers twice (ping-pong, as
            # air-label-scf-for-to-ping-pong duplicates them), on an even
            # number of chunks (its unroll by two).
            if stages == 2 and (chunks < 2 or chunks % 2):
                continue
            fixed = c * (1 + epilogue_inputs) + x * stages + _align(stack_bytes) + 256
            if fixed + w_full <= room:
                resident, w = True, w_full
            elif fixed + w_chunk * stages <= room:
                resident, w = False, w_chunk * stages
            else:
                continue
            counts = (n, co // toc, oh // th, ow // tw)
            herd, axes = select_axes_herd(counts, budget, herd_axis_limits)
            cores = prod(herd)
            sharing = cores // (herd[axes.index(1)] if 1 in axes else 1)  # cores that load the same W tile
            spatial = n * counts[2] * counts[3]
            x_traffic = counts[1] * spatial * chunks * tic * halo_h * halo_w * 4
            w_traffic = (counts[1] * toc * ci * kh * kw * 4 * (sharing if resident else spatial))
            c_traffic = 2 * n * co * oh * ow * 4
            macs = n * co * oh * ow * ci * kh * kw
            transfers = counts[1] * spatial * chunks
            compute = macs / (8 * cores * (tw / _align(tw, 8)))
            # Streamed chunks overlap the computation on the previous chunk
            # when double-buffered; otherwise each waits for its transfer.
            streamed = x_traffic + (0 if resident else w_traffic)
            other = x_traffic + w_traffic + c_traffic - streamed
            score = (max(compute, streamed / 4) if stages == 2 else compute + streamed / 4) + other / 4 + transfers * 64
            key = (score, -cores, -tw, stages)
            if best is not None and key >= best[0]:
                continue
            best = (key, dict(
                tile=dict(batch=1, oc=toc, ic=tic, oh=th, ow=tw, kh=kh, kw=kw), output_shape=[n, co, oh, ow],
                weights_resident=resident, input_halo=[halo_h, halo_w], stages=stages,
                memory=dict(input=x * stages, weight=w, output=c, epilogue=c * epilogue_inputs, stack=_align(stack_bytes), scratch=256,
                            static=CORE_STATIC_BYTES, total=fixed + w + CORE_STATIC_BYTES, l1_limit=l1_bytes, slots=stages),
                core_group=dict(cores=cores, herd_size=herd, axes=axes, mapping="resident"),
                estimates=dict(input_transfer_bytes=x_traffic, weight_transfer_bytes=w_traffic,
                               output_transfer_bytes=c_traffic, macs=macs, score=score, transfers=transfers),
                dataflow="resident", physical_allocation_verified=False))
    if best is None:
        raise ValueError(f"No conv tile fits {l1_bytes} bytes of L1 with single buffers")
    return best[1]
