"""Automatic structured scheduling from native target and platform contracts.

This module does not consume the legacy core-count/axis/anchor presets. It emits
options and native Transform IR, never replacement kernels or a model schedule.
The analytical ordering is not a calibrated latency model or an autotuner.
"""
from itertools import permutations, product
import hashlib
import json
from math import prod
from pathlib import Path

from aiec_air.memory_fix import _divisors
from compiler.tools import NativeCompileError

MANUAL_FIELDS = {'max_cores', 'herd_axis_limits', 'herd_first_dim', 'row_anchor',
                 'col_anchor', 'tile_sizes', 'l1_bytes', 'stack_bytes'}


def resolve(policy, target):
    """Resolve board availability before planning; hardware size is not usage."""
    from compiler.platform import gmio_endpoints
    from compiler.platform import BASE_ID, load_gmio_contract
    if MANUAL_FIELDS.intersection(policy):
        raise NativeCompileError('Automatic scheduling does not accept manual tile/core/axis presets: '
                                 + ', '.join(sorted(MANUAL_FIELDS.intersection(policy))))
    platform = policy.get('platform')
    if platform != BASE_ID or not policy.get('platform_contract'):
        raise NativeCompileError('Automatic scheduling requires the base GMIO platform contract')
    interface = Path(policy['platform_contract']).resolve()
    load_gmio_contract(interface)
    cores = [t for t in target['tiles'] if t['core']]
    if not cores:
        raise NativeCompileError('Native target has no available compute tiles')
    col = min(t['tile'][0] for t in cores)
    row = min(t['tile'][1] for t in cores)
    columns = max(t['tile'][0] for t in cores)-col+1
    rows = max(t['tile'][1] for t in cores)-row+1
    if len(cores) != columns*rows:
        raise NativeCompileError('Current native AIR placer needs a rectangular compute partition')
    endpoints = gmio_endpoints(interface)
    shims = [t for t in target['tiles'] if t.get('shim_noc')]
    if platform == BASE_ID:
        noc = {t['tile'][0] for t in shims}
        if any(p['column'] not in noc for ports in endpoints.values() for p in ports):
            raise NativeCompileError('GMIO endpoint is not a NoC shim in the native AIE1 target')
    resource = {
        'source': target['source'], 'platform': policy['platform'],
        'compute_tiles': len(cores), 'grid': [columns, rows], 'origin': [col, row],
        'local_memory_bytes': target['local_memory_bytes'],
        'stack_bytes': target['default_stack_bytes'],
        'core_dma_inputs': min(t['dma_destinations'] for t in cores),
        'core_dma_outputs': min(t['dma_sources'] for t in cores),
        'shim_dma_inputs': sum(t['shim_dma_sources'] for t in shims),
        'shim_dma_outputs': sum(t['shim_dma_destinations'] for t in shims),
        'pl_inputs': len(endpoints['input']), 'pl_outputs': len(endpoints['output']),
        'packet_ids': target['packet_ids'], 'packet_id_source': target['packet_id_source'],
        'interface_path': str(interface),
        'interface_sha256': hashlib.sha256(interface.read_bytes()).hexdigest(),
        'resource_scope': 'exclusive_full_partition',
        'transport': 'gmio_endpoint_budget',
        'device_dataflow': policy['device_dataflow'],
        'runtime_binding_verified': False,
    }
    if not all(resource[k] > 0 for k in ('core_dma_inputs', 'core_dma_outputs',
                                         'shim_dma_inputs', 'shim_dma_outputs')):
        raise NativeCompileError('Native DMA resource contract is incomplete')
    # These are derived backend arguments, not user-provided mapping policy.
    return {**policy, 'resources': resource, 'col_anchor': col, 'row_anchor': row,
            'l1_bytes': resource['local_memory_bytes'], 'stack_bytes': resource['stack_bytes'],
            'max_cores': resource['compute_tiles']}


def vector_axis(ir, op, maps, iterator_types):
    """Find an independent output axis supported by the current SIMD adapter.

    Only the actual scalar payload and unit-stride/broadcast indexing qualify.
    No operation-name substitution, reduction reassociation or packing is done.
    """
    from aiec_air.cnn_fix import _affine
    body = list(op.regions[0].blocks[0].operations)
    supported = {'arith.addf', 'arith.subf', 'arith.mulf', 'arith.constant', 'linalg.yield'}
    if any(child.operation.name not in supported for child in body):
        return None
    if not any(child.operation.name in {'arith.addf', 'arith.subf', 'arith.mulf'} for child in body):
        return None
    rank = len(iterator_types)
    outputs = list(op.attributes['operandSegmentSizes'])[-1]
    for axis in reversed(range(rank)):
        if iterator_types[axis] != 'parallel':
            continue
        unit = [int(i == axis) for i in range(rank)]
        legal = True
        for index, mapping in enumerate(maps):
            coefficients = [_affine(ir, expr, unit) - _affine(ir, expr, [0]*rank)
                            for expr in mapping.results]
            output = index >= len(maps) - outputs
            if not coefficients or any(coefficients[:-1]) or coefficients[-1] not in ({1} if output else {0, 1}):
                legal = False
                break
        if legal:
            return axis
    return None


def select_schedule(bounds, kinds, memory_for, traffic_for, *, resources,
                    vector_axis, payload_operations, input_buffers, output_buffers,
                    read_output_buffers=0,
                    candidate_limit=512, beam_width=24):
    """Jointly choose independent axes, tiles and a physical group shape.

    All independent axes can be considered, in either physical orientation.
    Reductions keep their relative order and are never distributed. Remaining
    logical tiles execute as native loop iterations, not extra physical cores.
    Ports bound the existing transport ABI, not the AIE architecture itself.
    """
    if (not bounds or len(bounds) != len(kinds) or
        any(type(n) is not int or n <= 0 for n in bounds) or
        any(k not in ('parallel', 'reduction') for k in kinds)):
        raise ValueError('Automatic scheduling requires positive static structured loop bounds')
    parallel = [i for i,k in enumerate(kinds) if k == 'parallel']
    if not parallel:
        raise ValueError('Automatic AIR scheduling needs an independent output axis')
    if vector_axis is not None and vector_axis not in parallel:
        raise ValueError('SIMD axis must be independent')
    # The connected native ObjectFIFO allocator assigns a DMA channel to each
    # endpoint. Do not assume the direct-DMA path's descriptor serialization.
    input_streams = input_buffers + read_output_buffers
    output_streams = output_buffers
    if input_streams > resources['core_dma_inputs'] or output_streams > resources['core_dma_outputs']:
        raise ValueError(f'ObjectFIFO endpoint demand exceeds per-core DMA capacity: '
                         f'inputs={input_streams}/{resources["core_dma_inputs"]}, '
                         f'outputs={output_streams}/{resources["core_dma_outputs"]}. '
                         'Proven constant initializers are localized; remaining logical inputs '
                         'require explicit DMA channel multiplexing or a stronger residency proof; '
                         'reducing the core count cannot solve this per-core constraint.')
    capacity = min(resources['compute_tiles'],
        min(resources['pl_inputs'], resources['shim_dma_inputs']) // max(1,input_streams),
        min(resources['pl_outputs'], resources['shim_dma_outputs']) // max(1,output_streams))
    # ObjectFIFO lowering uses native circuit endpoints. It does not inherit
    # the old direct-DMA lowering's global packet-identifier namespace.
    if capacity < 1:
        raise ValueError('Platform has insufficient transport resources for one worker')
    axes = list(permutations(parallel, 2)) if len(parallel) >= 2 else [(parallel[0], None)]
    # Rank-one operations may use columns or rows without naming a physical core.
    if len(parallel) == 1:
        axes.append((None, parallel[0]))
    factors = [list(_divisors(n)) for n in bounds]
    evaluated = {}

    def evaluate(tile):
        if tile in evaluated:
            return evaluated[tile]
        memory = memory_for(tile)
        if memory['total'] > memory['l1_limit']:
            result = {'tile': tile, 'legal': False,
                      'rank': (1, memory['total']/memory['l1_limit'], prod(tile))}
        else:
            counts = [b//t for b,t in zip(bounds,tile)]
            tasks = prod(counts)
            width = 8 if vector_axis is not None and tile[vector_axis] % 8 == 0 else 1
            traffic = sum(traffic_for(tile))*tasks
            work = prod(bounds)*max(1,payload_operations)/width
            groups = []
            for pair in axes:
                choices = [[1] if a is None else [n for n in _divisors(counts[a])
                           if n <= resources['grid'][d]] for d,a in enumerate(pair)]
                for group in product(*choices):
                    cores = prod(group)
                    if cores > capacity:
                        continue
                    # Shared DDR traffic is never divided by core count. These
                    # are normalized work units; no overlap/residency is assumed.
                    score = traffic/4 + (work + tasks*(input_buffers+output_buffers))/cores + cores
                    ordered = tuple(a for a in pair if a is not None)
                    groups.append(((score, cores, ordered, group), pair, group))
            key, pair, group = min(groups, key=lambda g:g[0])
            order = [a for a in pair if a is not None]
            order += [a for a in range(len(bounds)) if a not in order]
            # For a rank-one loop AIR first-dim selects its physical orientation.
            first_dim = int(pair[0] is None)
            result = {'tile': tile, 'legal': True, 'memory': memory, 'order': order,
                      'axes': pair, 'group': group, 'first_dim': first_dim,
                      'width': width, 'traffic': traffic, 'tasks': tasks,
                      'score': key[0], 'rank': (0, *key, memory['total'], tile)}
        evaluated[tile] = result
        return result

    minimum = evaluate(tuple(1 for _ in bounds))
    if not minimum['legal']:
        raise ValueError('Minimum live working set exceeds native local memory')
    frontier = [tuple(bounds)]
    while frontier and len(evaluated) < candidate_limit:
        neighbors = set()
        for tile in frontier:
            evaluate(tile)
            for a,size in enumerate(tile):
                smaller = factors[a][:factors[a].index(size)]
                for n in smaller[-1:] + [x for x in smaller if x <= size//2][-1:]:
                    candidate = tile[:a]+(n,)+tile[a+1:]
                    if candidate not in evaluated:
                        neighbors.add(candidate)
        records = []
        for candidate in sorted(neighbors):
            if len(evaluated) >= candidate_limit:
                break
            records.append(evaluate(candidate))
        frontier = [r['tile'] for r in sorted(records,key=lambda r:r['rank'])[:beam_width]]
    best = min((r for r in evaluated.values() if r['legal']),key=lambda r:r['rank'])
    tile, order = best['tile'], best['order']
    herd = list(best['group'])
    if best['first_dim']:
        herd.reverse()
    return [tile[a] for a in order], {
        'strategy': 'automatic', 'memory': best['memory'], 'loop_extents': list(bounds),
        'iterator_types': kinds, 'original_tile_sizes': list(tile), 'loop_permutation': order,
        'core_group': {'selected_axes': list(best['axes']), 'herd_size': herd,
                       'physical_shape': list(best['group']), 'cores': prod(herd),
                       'first_dim': best['first_dim'], 'placement_verified': False},
        'resource_contract': resources, 'transport_core_capacity': capacity,
        'transport_reservation': {'inputs_per_core':input_streams,'outputs_per_core':output_streams},
        'packet_reservation': {'required':False,'reason':'ObjectFIFO circuit dataflow'},
        'compute': {'vector_axis': vector_axis, 'vector_width': best['width'],
                    'reduction_order': 'preserved', 'backend_vectorization_verified': False},
        'estimates': {'score':best['score'],'units':'uncalibrated_work_units',
                      'logical_transfer_bytes':best['traffic'],'tile_invocations':best['tasks'],
                      'measured':False},
        'search': {'evaluated':len(evaluated),'candidate_limit':candidate_limit,'exhaustive':False},
        'buffering': {'reserved_slots':best['memory']['slots'],'overlap_scheduled':False},
        'physical_allocation_verified': False,
    }


def write_transform(plan, path):
    """Generate a native Transform program; native interchange owns SSA/maps."""
    lines = ['module attributes {transform.with_named_sequence} {',
             'transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {']
    for i,op in enumerate(plan['operations']):
        order = op['memory_plan']['loop_permutation']
        marker = json.dumps(op['marker'])
        lines += [f'%m{i} = transform.structured.match attributes {{__internal_linalg_transform__ = {marker}}} in %root : (!transform.any_op) -> !transform.any_op',
                  f'%g{i} = transform.structured.generalize %m{i} : (!transform.any_op) -> !transform.any_op',
                  f'%p{i} = transform.structured.interchange %g{i} iterator_interchange = [{", ".join(map(str,order))}] : (!transform.any_op) -> !transform.any_op']
    lines += ['transform.yield', '}', '}']
    Path(path).write_text('\n'.join(lines)+'\n')
