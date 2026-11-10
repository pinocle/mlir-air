"""AIR placement and AIE1 lowering arguments for the project pipeline."""

CORE_LOOP_PASSES = ('--air-lower-linalg-tensors',)


def air_lowering_arguments(policy, target):
    from compiler.backends.device import validate_policy
    validate_policy(policy, target)
    row, col = policy['row_anchor'], policy['col_anchor']
    fifo = ' use-objectfifo=true' if policy.get('device_dataflow') == 'objectfifo' else ''
    return ('--pass-pipeline=builtin.module('
        f'air-place-herds{{num-rows={target["rows"]-row} num-cols={target["columns"]-col} '
        f'row-anchor={row} col-anchor={col}}},'
        # aircc: same-channel async ops serialized after placement (FIFO order).
        + ('air-enforce-channel-fifo-order,func.func(air-renumber-dma),' if policy.get('air_schedule') == 'asynchronous' else '')
        + f'air-to-aie{{device={target["target"]} row-offset={row} col-offset={col} '
        f'emit-while-loop=false generate-shim-dma=true stack-size={policy["stack_bytes"]}{fifo}}},'
        # Host copies of herd bodies keep affine ops on herd ids; the AIE
        # bindings parse them without the AIR dialect, so lower them to scf.
        'func.func(lower-affine),canonicalize,cse)',)
