"""Real Linalg -> AIR -> AIE1 transformation and physical configuration passes."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

from compiler.tools import NativeCompileError, NativeStage, Toolchain, run_stage
from aiec_air.lowering_fix import CORE_LOOP_PASSES, air_lowering_arguments
from compiler.backends.device import target_for, validate_policy, record_resources, device_passes

from compiler.environment import ROOT
DEFAULT_TOOLCHAIN = ROOT / "toolchains/layered.json"

# Normalization of the common IR (Linalg on tensors), run by aiec-opt: the
# passes of hexagon-mlir 9ec15e8 LinalgToLLVMPass.cpp:204-334 that do not
# target Hexagon, in its order; the hexagon-mlir ones are copied into
# lib/Transforms. hexagon-lower-tm-tensor runs in the frontend
# (compiler/torch_mlir/compiler_utils.py), lower-quant-ops has no quant
# dialect to lower, and hexagon-fusion is upstream linalg-fuse-elementwise-ops
# (Torch-MLIR RefBackend's). remove-ml-program is left out: ml_program globals
# become memref.global in bufferization.
LINALG_TENSOR_NORMALIZE_PIPELINE = "builtin.module(" + ",".join([
    "func.func(reduce-contraction-rank)",
    "linalg-fold-unit-extent-dims", "canonicalize", "cse",
    "convert-elementwise-to-linalg", "cse",
    "func.func(linalg-generalize)", "canonicalize", "cse",
    "linalg-fuse-elementwise-ops",
    "erase-unused-linalg-operands", "canonicalize", "cse",
    "func.func(triton-decompose-tensor-concat)", "cse",
    "linalg-fold-unit-extent-dims", "canonicalize", "cse",
]) + ")"

# Bufferization, run by aiec-opt: hexagon-mlir LinalgToLLVMPass.cpp:367-409
# with mlir-air's function boundary (29e3586 python/air/compiler/util.py
# LINALG_TENSOR_TO_MEMREF_PIPELINE, python/air/backend/xrt.py
# compile_from_torch_mlir: identity-layout-map, buffer-results-to-out-params).
# Like hexagon-mlir and unlike mlir-air there is no copy-before-write, so
# buffers are reused in place. The deallocation pipeline is left out, as in
# mlir-air: AIR places device memory. aiec-read-only-arguments (not from a
# base) keeps function inputs unwritten, which copy-before-write ensured.
LINALG_TENSOR_TO_MEMREF_PIPELINE = "builtin.module(" + ",".join([
    "aiec-read-only-arguments",
    "resolve-shaped-type-result-dims",
    "eliminate-empty-tensors",
    "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    " allow-return-allocs-from-loops}",
    "cse", "canonicalize",
    "func.func(buffer-loop-hoisting)",
    "func.func(copy-canonicalization)", "canonicalize",
    "buffer-results-to-out-params{modify-public-functions=true hoist-static-allocs=true}",
    "convert-bufferization-to-memref", "canonicalize", "cse",
]) + ")"


def linalg_tensor_to_memref_pipeline(operation_names):
    """LINALG_TENSOR_TO_MEMREF_PIPELINE for a module with these operations.
    tm_tensor lowered in the frontend (tm-tensor-bufferize) leaves
    bufferization.to_tensor ops created before their buffer is written, which
    one-shot-bufferize's analysis (and eliminate-empty-tensors, which runs
    it) rejects; such a module bufferizes with copy-before-write and without
    eliminate-empty-tensors, as Torch-MLIR's RefBackend (refbackend.py) runs
    tm-tensor-bufferize."""
    if "bufferization.to_tensor" in operation_names:
        return (LINALG_TENSOR_TO_MEMREF_PIPELINE.replace("eliminate-empty-tensors,", "")
                .replace("one-shot-bufferize{", "one-shot-bufferize{copy-before-write "))
    return LINALG_TENSOR_TO_MEMREF_PIPELINE


def load_policy(path):
    policy = json.loads(Path(path).read_text())
    if policy.get("schema_version") != 1 or policy.get("target") != "xcvc1902":
        raise NativeCompileError("Policy must target VCK190 / xcvc1902, schema_version 1")
    from compiler.platform import BASE_ID, load_gmio_contract
    if policy.get("platform") not in (None, BASE_ID):
        raise NativeCompileError("Only the base GMIO platform is supported")
    if policy.get('platform') == BASE_ID:
        reference = policy.get('platform_contract')
        if not reference:
            raise NativeCompileError('Base GMIO policy requires platform_contract')
        contract_path = (Path(path).resolve().parent / reference).resolve()
        if not contract_path.is_file():
            raise NativeCompileError('Base GMIO platform is not prepared: ' + str(contract_path)
                                     + '; build/package and activate the platform contract first')
        load_gmio_contract(contract_path)
        policy['platform_contract'] = str(contract_path)
    tile = policy.get("tile_sizes")
    tiled = policy.get('tiling_strategy') != 'automatic'
    if tiled and (not isinstance(tile, list) or not tile or any(type(x) is not int or x <= 0 for x in tile)):
        raise NativeCompileError("Policy tile_sizes must be positive static integers")
    if policy.get("transport") not in {"circuit", "packet"}:
        raise NativeCompileError("Policy transport must be circuit or packet")
    if policy.get("operator_pipeline", "standard") not in {"standard", "cnn"}:
        raise NativeCompileError("Unknown operator pipeline")
    if policy.get('core_optimization', 'none') not in {'none', 'provided-simd'}:
        raise NativeCompileError('Unknown AIE1 core optimization')
    strategy = policy.get('tiling_strategy', 'baseline')
    if strategy not in {'baseline', 'automatic'}:
        raise NativeCompileError('Unknown structured tiling strategy')
    if strategy == 'automatic' and (policy.get('operator_pipeline') != 'cnn' or policy.get('core_optimization') != 'provided-simd'):
        raise NativeCompileError('Automatic tiling requires structured CNN lowering and the provided SIMD backend')
    if strategy == 'automatic':
        from aiec_air.automatic_fix import MANUAL_FIELDS
        if MANUAL_FIELDS.intersection(policy):
            raise NativeCompileError('Automatic scheduling rejects manual tile/core/axis presets')
        if policy.get('device_dataflow') != 'objectfifo' or policy['transport'] != 'circuit':
            raise NativeCompileError('Automatic scheduling requires native ObjectFIFO circuit lowering')
        return policy
    if type(policy.get('herd_first_dim', 0)) is not int or policy.get('herd_first_dim', 0) not in (0, 1):
        raise NativeCompileError('AIR herd_first_dim must be 0 or 1')
    if policy.get('conv_distribution', 'n_oc') not in ('n_oc', 'spatial'):
        raise NativeCompileError('Conv distribution must be n_oc or spatial')
    if policy.get('conv_dataflow', 'tiled') not in ('tiled', 'resident'):
        raise NativeCompileError('Conv dataflow must be tiled or resident')
    if policy.get('air_schedule', 'synchronous') not in ('synchronous', 'asynchronous'):
        raise NativeCompileError('AIR schedule must be synchronous or asynchronous')
    cores = policy.get('max_cores', 4)
    limits = policy.get('herd_axis_limits', [2, 2])
    if type(cores) is not int or cores < 1 or not isinstance(limits, list) or len(limits) != 2 or any(type(n) is not int or n < 1 for n in limits):
        raise NativeCompileError('Core policy requires a positive core budget and two positive axis limits')
    for name in ("row_anchor", "col_anchor"):
        value = policy.get(name)
        if type(value) is not int or value < 0:
            raise NativeCompileError(f"Invalid AIE1 {name}")
    # Physical limits come from the selected native target, checked on entry.
    if any(type(policy.get(k)) is not int or policy[k] <= 0 for k in ('l1_bytes','stack_bytes')):
        raise NativeCompileError('L1 and stack budgets must be positive integers')
    return policy


def _native_python(mode, source, output, tool_config, log_dir, dialect="air", extra=()):
    # A venv interpreter is often a symlink. Resolving it loses pyvenv.cfg.
    python = Path(tool_config["python"]).absolute()
    command = [str(python), "-m", "compiler.ir_tools", mode, str(source),
               "--out", str(output), "--dialect", dialect, *map(str, extra)]
    started = time.monotonic()
    with (log_dir / (output.stem + ".python.stdout.log")).open("xb") as stdout, \
         (log_dir / (output.stem + ".python.stderr.log")).open("xb") as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=120, check=False)
    if result.returncode or not output.is_file() or output.stat().st_size == 0:
        raise NativeCompileError(f"Native {mode} failed; see {log_dir / (output.stem + '.python.stderr.log')}")
    return {"command": command, "elapsed_seconds": time.monotonic() - started,
            "output": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def _resident_conv_transform(operations):
    """A native Transform program for the resident conv dataflow
    (memory.plan_conv_resident): per marked conv, a forall over the cores'
    blocks, then OC tiles (weights promoted to L1 there when resident), OH/OW
    tiles (the accumulator promoted there), IC chunks (the input promoted
    there, and the weights when not resident). Loops are (n, oc, oh, ow, ic,
    kh, kw), as the CNN generalization leaves them."""
    lines = ['module attributes {transform.with_named_sequence} {',
             'transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {']
    any_op = '!transform.any_op'

    def tile(i, step, target, sizes, kind='for'):
        loops = sum(1 for x in sizes if x)
        results = ', '.join([any_op] * (loops + 1)) if kind == 'for' else f'{any_op}, {any_op}'
        op = 'tile_using_for' if kind == 'for' else 'tile_using_forall'
        lines.append(f'%{step}{i}, %{step}{i}_loops' + (f':{loops}' if kind == 'for' and loops > 1 else '') +
                     f' = transform.structured.{op} %{target} tile_sizes [{", ".join(map(str, sizes))}] : ({any_op}) -> ({results})')

    def promote(i, step, target, operands):
        lines.append(f'%{step}{i} = transform.air.linalg_promote %{target} <{{operands_to_promote = [{", ".join(map(str, operands))}], memory_space = "L1"}}> : ({any_op}) -> {any_op}')

    for i, op in enumerate(operations):
        plan = op["memory_plan"]
        n, oc, oh, ow = plan["output_shape"]
        t = plan["tile"]
        herd, axes = plan["core_group"]["herd_size"], plan["core_group"]["axes"]
        extents = [n, oc, oh, ow]
        block = list(extents)
        for g, a in zip(herd, axes):
            block[a] = extents[a] // g
        lines.append(f'%m{i} = transform.structured.match attributes {{__internal_linalg_transform__ = {json.dumps(op["marker"])}}} in %root : ({any_op}) -> {any_op}')
        tile(i, 'core', f'm{i}', [block[a] if a in axes else 0 for a in range(4)] + [0, 0, 0], kind='forall')
        tile(i, 'oc', f'core{i}', [0, t["oc"], 0, 0, 0, 0, 0])
        current = f'oc{i}'
        if plan["weights_resident"]:
            promote(i, 'w', current, [1])
            current = f'w{i}'
        tile(i, 'sp', current, [t["batch"] if n > 1 else 0, 0, t["oh"], t["ow"], 0, 0, 0])
        promote(i, 'c', f'sp{i}', [2])
        tile(i, 'ic', f'c{i}', [0, 0, 0, 0, t["ic"], 0, 0])
        promote(i, 'x', f'ic{i}', [0] if plan["weights_resident"] else [0, 1])
    lines += ['transform.yield', '}', '}']
    return '\n'.join(lines) + '\n'


def _prepare_array(current, out, config, logs, report):
    """AIE1 cascade flows lowered, then the device checked against the array
    contract (aie1_array prepare); returns the device to continue with."""
    destination = Path(out) / (Path(current).stem + '-array.mlir')
    command = [str(Path(config['python']).absolute()), '-m', 'compiler.backends.aie1_array', 'prepare', str(current),
               '--out', str(destination)]
    with (logs / 'array.stdout.log').open('xb') as stdout, (logs / 'array.stderr.log').open('xb') as stderr:
        done = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=120)
    if done.returncode:
        raise NativeCompileError(f'AIE array preparation failed; see {logs / "array.stderr.log"}')
    report['array'] = json.loads(destination.with_suffix('.json').read_text())
    return destination


def _assign_dma_channels(current, out, config, logs, report):
    """Split a tile's transfer tasks over its two DMA channels (aie1_tile.plan_channels)."""
    destination = Path(out) / (Path(current).stem + '-channels.mlir')
    command = [str(Path(config['python']).absolute()), '-m', 'compiler.backends.aie1_tile', 'channels', str(current),
               '--out', str(destination)]
    with (logs / 'dma-channels.stdout.log').open('xb') as stdout, (logs / 'dma-channels.stderr.log').open('xb') as stderr:
        done = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=120)
    if done.returncode:
        raise NativeCompileError(f'Tile DMA channel assignment failed; see {logs / "dma-channels.stderr.log"}')
    report['dma_channels'] = json.loads(destination.with_suffix('.json').read_text())
    return destination


def _map_core_matmuls(current, out, config, logs, policy, report):
    """Core-local matmuls -> AIE API aie::mmul adapter calls (provided SIMD)."""
    if policy.get('core_optimization') != 'provided-simd':
        return current
    destination = Path(out) / (Path(current).stem + '-mmul.mlir')
    command = [str(Path(config['python']).absolute()), '-m', 'compiler.backends.aie1_core', 'map', str(current),
               '--out', str(destination)]
    with (logs / 'mmul.stdout.log').open('xb') as stdout, (logs / 'mmul.stderr.log').open('xb') as stderr:
        done = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=120)
    if done.returncode:
        raise NativeCompileError(f'Core matmul mapping failed; see {logs / "mmul.stderr.log"}')
    report['core_matmul'] = json.loads(destination.with_suffix('.json').read_text())
    return destination


def _compile_partitioned_aie(source, out, policy, toolchain, config, report, *, stop_after="aie", budget_seconds=600):
    """Compile independent projections; source order is never a runtime schedule."""
    started = time.monotonic()
    deadline = started + budget_seconds
    out = Path(out)
    logs = out / "logs"
    projections = out / "air-projections"
    command = [str(Path(config["python"]).absolute()), "-m", "aiec_air.partition_fix", str(source), str(projections)]
    with (logs / "partition.stdout.log").open("xb") as stdout, (logs / "partition.stderr.log").open("xb") as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=min(120, budget_seconds), check=False)
    if result.returncode:
        raise NativeCompileError(f"Native AIR partition failed; see {logs / 'partition.stderr.log'}")
    manifest_path = projections / "manifest.json"
    partition = json.loads(manifest_path.read_text())
    report["air_partition"] = {"manifest": str(manifest_path), "command": command,
        "source_launch_count": partition["source_launch_count"], "projection_only": True}
    report.update({"programs": [], "runtime_schedule_generated": False,
                   "execution_model": "independent_device_programs_without_runtime_schedule",
                   "scheduled_mixed_ir": str(source)})
    physical = {"native_verifier": "not_completed", "devices": [], "cores": [], "buffers": [],
                "simultaneous_placement_verified": False}
    report["inspections"]["physical"] = physical
    backend = {"programs": [], "cores": [], "core_machine_code_generated": False,
               "device_configuration_generated": False, "elf_generated": False,
               "runtime_schedule_generated": False, "executable_image_generated": False,
               "platform_runtime_verified": False, "board_execution_performed": False}
    listing = out / "device-programs.json"
    if stop_after == "elf":
        report["backend"] = backend
    try:
        for item in partition["programs"]:
            program_policy = dict(policy)
            if item.get('device_dataflow'):
                program_policy['device_dataflow'] = item['device_dataflow']
                if item['device_dataflow'] == 'packet':
                    program_policy['transport'] = 'packet'
            lower = air_lowering_arguments(program_policy, report['device_target'])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NativeCompileError("Partitioned AIE compilation exceeded its total time budget")
            program_id = f"program-{item['source_order_id']:04d}"
            folder = out / "programs" / program_id
            folder.mkdir(parents=True)
            program_logs = folder / "logs"
            program_logs.mkdir()
            entry = {"program_id": program_id, "source_order_id": item["source_order_id"],
                "segment": item["segment"], "herd": item["herd"], "projection": item["path"],
                "projection_only": True, "status": "compiling", "stages": [], "inspections": {},
                "runtime_schedule_generated": False, "channel_mapping": item["channel_mapping"]}
            entry['source_launches'] = item.get('source_launches',[item['source_order_id']])
            entry['herds'] = item['herds']
            entry['device_dataflow'] = program_policy.get('device_dataflow','packet')
            report["programs"].append(entry)
            current = Path(item["path"])

            def program_stage(name, tool, arguments):
                nonlocal current
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise NativeCompileError("Partitioned AIE compilation exceeded its total time budget")
                destination = folder / (name + ".mlir")
                record = run_stage(NativeStage(name, tool, tuple(arguments)), current, destination,
                                   program_logs, toolchain, timeout=min(120, remaining))
                entry["stages"].append(record)
                current = destination
                entry["last_successful_output"] = str(current)

            def program_inspect(label):
                destination = folder / (label + ".json")
                _native_python("inspect", current, destination, config, program_logs, "aie")
                data = json.loads(destination.read_text())
                entry["inspections"][label] = data
                return data

            try:
                # AIE1 rejects the repeat counts air-to-aie emits for tasks of
                # one channel; verification follows the channel assignment.
                program_stage("04-aie-host", "air-opt", ["--verify-each=false", "--mlir-print-op-generic", *lower])
                current = _assign_dma_channels(current, folder, config, program_logs, entry)
                entry["mixed_host_device_ir"] = str(current)
                program_stage("05-aie-generic", "air-opt", ["--mlir-print-op-generic"])
                device = folder / "06-device.mlir"
                entry["device_extraction"] = _native_python("extract", current, device, config, program_logs, "aie")
                current = device
                current = _prepare_array(current, folder, config, program_logs, entry)
                program_stage("07-physical", "aie-opt", device_passes(program_policy))
                inspection = program_inspect("physical")
                if not inspection["cores"] or not inspection["devices"]:
                    raise NativeCompileError("Native AIE projection generated no cores/devices")
                for name in ("devices", "cores", "buffers"):
                    physical[name].extend({**value, "program_id": program_id} for value in inspection[name])
                current = _map_core_matmuls(current, folder, config, program_logs, policy, entry)
                program_stage("08-core-loops", "air-opt", CORE_LOOP_PASSES)
                program_inspect("lowered")
                entry['device_resources'] = record_resources(config['python'], current, folder/'device-resources.json')
                entry["device_ir"] = str(current)
                entry["completed_stage"] = "aie"
                if stop_after == "elf":
                    from compiler.backends.aie1 import compile_cores
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise NativeCompileError("Partitioned ELF compilation exceeded its total time budget")
                    child = compile_cores(current, folder / "aie1-backend", timeout=min(120, remaining),
                                          core_optimization=policy.get('core_optimization', 'none'))
                    entry["backend"] = child
                    backend["programs"].append({"program_id": program_id, **child})
                    backend["cores"].extend({**core, "program_id": program_id} for core in child["cores"])
                    entry["completed_stage"] = "elf"
                entry["status"] = "compiled_ir"
            except BaseException as exc:
                entry["status"] = "failed"
                entry["error"] = str(exc)
                raise
            finally:
                entry["last_successful_output"] = str(current)
                (folder / "compilation.json").write_text(json.dumps(entry, indent=2) + "\n")
        physical["native_verifier"] = "passed"
        report["inspections"]["lowered"] = {"native_verifier": "passed", "program_count": len(report["programs"])}
        if stop_after == "elf":
            for key in ("core_machine_code_generated", "device_configuration_generated", "elf_generated"):
                backend[key] = bool(backend["programs"]) and all(program[key] for program in backend["programs"])
            report["core_machine_code_generated"] = backend["core_machine_code_generated"]
            report["device_configuration_generated"] = backend["device_configuration_generated"]
        report["completed_stage"] = stop_after
        report["device_programs_manifest"] = str(listing)
        return listing
    finally:
        listing.write_text(json.dumps({"schema_version": 1, "source_mixed_ir": str(source),
            "projection_only": True, "runtime_schedule_generated": False,
            "simultaneous_placement_verified": False, "programs": report["programs"]}, indent=2) + "\n")


def _aircc_optimization(channels, device):
    # AMD aircc buildOptimizationPipeline (tools/aircc/aircc.cpp) for a device
    # without L2 memory: no air-split-l2-memref; the ping-pong labeller
    # declines loops whose doubled L1 buffers would not fit the device's tile.
    return ["air-dependency", "air-annotate-packet-ids{assign=true}", "air-hoist-dma-in-accum-pattern",
            "air-broadcast-detection", "air-specialize-dma-broadcast", channels, "canonicalize", "cse",
            "air-dependency-canonicalize", "canonicalize", "cse",
            "air-isolate-async-dma-loop-nests{scope=launch}", "canonicalize", "cse",
            "air-fuse-channels", "canonicalize", "cse", "func.func(air-fuse-alloc-dealloc)",
            "func.func(air-shrink-memref-sizes-by-access)", f"air-label-scf-for-to-ping-pong{{device={device}}}",
            "air-ping-pong-transform", "canonicalize", "cse"]


def _lower_air(current, out, policy, toolchain, config, report, stage, inspect, *, stop_after, partition_programs):
    """Shared AIR -> channels -> per-launch AIE1 lowering for every frontend."""
    logs = out / "logs"
    air = inspect("air", source=current)
    if not air["operation_counts"].get("air.herd"):
        raise NativeCompileError("Native codegen produced no AIR herd; this operation/tiling is not supported")
    if air["unmapped_computation"]:
        raise NativeCompileError("Computation remained outside AIR herds: " + ", ".join(sorted(set(air["unmapped_computation"]))))
    report["host_initialization"] = air["host_initialization"]
    channels = "air-dma-to-channel{force-shim-packet-flow=true}" if policy["transport"] == "packet" else "air-dma-to-channel"
    # Nested CNN reductions currently fail native AIE1 async wait_all
    # legalization. A synchronous channel schedule preserves order and
    # exposes that limitation instead of falling back to CPU arithmetic.
    # Explicit async policy below retains the native device dependency graph;
    # grouping normalizes only the host side to the sequential runtime ABI.
    dependency = "" if policy.get("operator_pipeline") == "cnn" else "air-dependency,"
    if policy.get("air_schedule") == "asynchronous":
        # AMD aircc's optimization pipeline (tools/aircc/aircc.cpp,
        # buildOptimizationPipeline): dependency graph, broadcast detection,
        # channels, isolated DMA loop nests and ping-pong buffering.
        current = stage("03-optimized", "air-opt", ["--pass-pipeline=builtin.module(" + ",".join(_aircc_optimization(channels, report['device_target']['target'])) + ")"], source=current)
        # Token-only segment loops air-to-aie cannot legalize (aiec-opt pass).
        current = stage("03-scheduled", "aiec-opt", ["--aiec-prune-token-loops"], source=current)
    else:
        current = stage("03-scheduled", "air-opt", ["--pass-pipeline=builtin.module(" + dependency + channels + ",canonicalize,cse)"], source=current)
    report["async_dependency_analysis"] = bool(dependency) or policy.get("air_schedule") == "asynchronous"
    if policy.get('device_dataflow') == 'objectfifo':
        # A herd buffer both received and sent gets distinct FIFO owners (aiec-opt pass).
        fifo_report = out / '03-fifo-ownership.fifo.json'
        current = stage('03-fifo-ownership', 'aiec-opt', [f'--aiec-split-readwrite-fifo=report={fifo_report}'],
                        source=current)
        report['fifo_ownership'] = json.loads(fifo_report.read_text())
    if policy.get('program_grouping', 'auto') not in {'auto', 'off'}:
        raise NativeCompileError('program_grouping must be auto or off')
    if policy.get('program_grouping', 'auto') == 'auto' and policy.get('platform_contract'):
        from compiler.platform import load_gmio_contract
        ports = load_gmio_contract(policy['platform_contract'])['endpoints']
        grouped = out / '03-grouped.mlir'
        command = [config['python'], '-m', 'aiec_air.grouping_fix', str(current), str(grouped),
                   '--columns', str(report['device_target']['columns']-policy['col_anchor']),
                   '--rows', str(report['device_target']['rows']-policy['row_anchor']),
                   '--local-bytes',str(policy['l1_bytes']),'--stack-bytes',str(policy['stack_bytes']),
                   '--input-ports', str(len(ports['input'])), '--output-ports', str(len(ports['output']))]
        with (logs/'grouping.stdout.log').open('xb') as stdout, (logs/'grouping.stderr.log').open('xb') as stderr:
            result = subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=120)
        if result.returncode:
            raise NativeCompileError('Spatial program grouping failed; see '+str(logs/'grouping.stderr.log'))
        report['program_grouping'] = json.loads(grouped.with_suffix('.json').read_text())
        current = grouped
    analysis_path = out / "model-analysis.json"
    optimized = out / "03-model-memory.mlir"
    command = [str(Path(config["python"]).absolute()), "-m", "compiler.model_analysis",
               str(current), str(analysis_path), "--optimized", str(optimized)]
    with (logs / "model-analysis.stdout.log").open("xb") as stdout, \
         (logs / "model-analysis.stderr.log").open("xb") as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=120, check=False)
    if result.returncode or not analysis_path.is_file():
        raise NativeCompileError(f"Native model analysis failed; see {logs / 'model-analysis.stderr.log'}")
    analysis = json.loads(analysis_path.read_text())
    if analysis["buffer_reuse_applied"]:
        if not optimized.is_file():
            raise NativeCompileError("Native model analysis did not produce optimized AIR")
        current = optimized
    report["model_analysis"] = {"path": str(analysis_path),
        "sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "source_sha256": analysis["source_sha256"], "buffer_reuse_applied": analysis["buffer_reuse_applied"],
        "reuse_count": sum(f["reuse"]["applied"] for f in analysis["functions"]),
        "uncertain_functions": [f["name"] for f in analysis["functions"] if f["ordering_or_effects_uncertain"]]}
    scheduled = inspect("schedule", source=current)
    report["completed_stage"] = "air"
    if stop_after == "air":
        report["completed_stage"] = "air"
    else:
        groups = report.get('program_grouping',{}).get('groups',[])
        overrides = [g['device_dataflow'] for g in groups if g.get('device_dataflow')]
        if partition_programs or ((policy.get("operator_pipeline") == "cnn" or overrides) and scheduled["operation_counts"].get("air.launch", 0) > 1):
            current = _compile_partitioned_aie(current, out, policy, toolchain, config, report, stop_after=stop_after)
        else:
            device_policy = dict(policy)
            if overrides:
                device_policy.update(device_dataflow=overrides[0],transport=overrides[0])
            report['device_dataflow'] = device_policy.get('device_dataflow','packet')
            current = stage("04-aie-host", "air-opt", ["--verify-each=false", "--mlir-print-op-generic", *air_lowering_arguments(device_policy, report['device_target'])], source=current)
            current = _assign_dma_channels(current, out, config, logs, report)
            report["mixed_host_device_ir"] = str(current)
            current = stage("05-aie-generic", "air-opt", ["--mlir-print-op-generic"], source=current)
            device = out / "06-device.mlir"
            report["device_extraction"] = _native_python("extract", current, device, config, logs, "aie")
            current = device
            current = _prepare_array(current, out, config, logs, report)
            current = stage("07-physical", "aie-opt", device_passes(device_policy), source=current)
            physical = inspect("physical", "aie", source=current)
            if not physical["cores"] or not physical["devices"]:
                raise NativeCompileError("Native AIE lowering generated no cores/devices")
            current = _map_core_matmuls(current, out, config, logs, policy, report)
            current = stage("08-core-loops", "air-opt", CORE_LOOP_PASSES, source=current)
            inspect("lowered", "aie", source=current)
            report['device_resources'] = record_resources(config['python'], current, out/'device-resources.json')
            report["device_ir"] = str(current)
            report["completed_stage"] = "aie"
            if stop_after == "elf":
                from compiler.backends.aie1 import compile_cores
                backend = compile_cores(current, out / "aie1-backend",
                                        core_optimization=policy.get('core_optimization', 'none'))
                report["backend"] = backend
                report["core_machine_code_generated"] = backend["core_machine_code_generated"]
                report["device_configuration_generated"] = backend["device_configuration_generated"]
                report["completed_stage"] = "elf"
    return current


def compile_mlir(source, out_dir, policy_path, *, toolchain_path=DEFAULT_TOOLCHAIN,
                 frontend_manifest=None, stop_after="aie", partition_programs=False, entry_stage="linalg"):
    """Emit native IR and optionally core ELFs; model deployment is separate.

    ``entry_stage="air"`` accepts AIR whose launches/herds were already formed by
    a frontend (the Inductor/Triton path) and continues with the shared lowering.
    """
    if entry_stage not in {"linalg", "air"}:
        raise NativeCompileError("Supported entry stages: linalg, air")
    if entry_stage == "air" and stop_after == "common":
        raise NativeCompileError("AIR input has no common-IR stage")
    source = Path(source).resolve()
    out = Path(out_dir).absolute()
    if stop_after not in {"common", "air", "aie", "elf"}:
        raise NativeCompileError("Supported stop stages: common, air, aie, elf")
    if out.exists() or out.is_symlink():
        raise NativeCompileError(f"Output already exists: {out}")
    if not source.is_file():
        raise NativeCompileError(f"MLIR source not found: {source}")
    policy = load_policy(policy_path)
    toolchain = Toolchain.from_json(toolchain_path)
    config = json.loads(Path(toolchain_path).read_text())
    config["python"] = str((Path(toolchain_path).resolve().parent / config["python"]).absolute())
    out.mkdir(parents=True, exist_ok=False)
    logs = out / "logs"
    logs.mkdir()
    report = {"schema_version": 1, "status": "compiling", "target": "xcvc1902", "policy": policy,
              "transform_policy": {"path": str(ROOT / "policies/common.mlir"),
                  "sha256": hashlib.sha256((ROOT / "policies/common.mlir").read_bytes()).hexdigest()},
              "source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
              "stages": [], "inspections": {}, "toolchain": config.get("versions", {}),
              "frontend": frontend_manifest,
              "core_machine_code_generated": False, "device_configuration_generated": False,
              "runtime_schedule_generated": False, "executable_image_generated": False,
              "platform_runtime_verified": False, "board_execution_performed": False,
              "numerical_validation": "not_performed", "cpu_fallback": False}
    current = source

    def stage(name, tool, arguments, source=None):
        nonlocal current
        dest = out / (name + ".mlir")
        record = run_stage(NativeStage(name, tool, tuple(arguments)), source or current, dest, logs, toolchain)
        report["stages"].append(record)
        current = dest
        return dest

    def inspect(label, dialect="air", source=None):
        dest = out / (label + ".json")
        _native_python("inspect", source or current, dest, config, logs, dialect)
        data = json.loads(dest.read_text())
        report["inspections"][label] = data
        return data

    try:
        report['device_target'] = target_for(config['python'])
        if policy.get('tiling_strategy') == 'automatic':
            from aiec_air.automatic_fix import resolve
            policy = resolve(policy, report['device_target'])
            report['resolved_policy'] = policy
            (out/'resource-contract.json').write_text(json.dumps(policy['resources'], indent=2)+'\n')
        validate_policy(policy, report['device_target'])
        report['device_pipeline'] = list(device_passes(policy))
        if entry_stage == "air":
            current = _lower_air(current, out, policy, toolchain, config, report, stage, inspect,
                                 stop_after=stop_after, partition_programs=partition_programs)
        else:
            # Text is the only boundary between independently versioned MLIR tools.
            stage("00-common", "air-opt", ["--pass-pipeline=builtin.module(inline,symbol-dce,canonicalize,cse)"])
            library = json.dumps(str(ROOT / "policies/common.mlir"))
            stage("00-transform", "air-opt", ["--pass-pipeline=builtin.module(transform-preload-library{transform-library-paths=" + library + "},transform-interpreter)"])
            common = inspect("common-input")
            if not common["functions"]:
                raise NativeCompileError("Common IR needs a defined function")
            # Fusing tensor elementwise producers before bufferization also keeps
            # SiLU's sigmoid result from becoming an inter-program buffer.
            stage("00-normalized", "aiec-opt", ["--pass-pipeline=" + LINALG_TENSOR_NORMALIZE_PIPELINE])
            before = common["operation_counts"]
            common = inspect("common")
            report["normalization"] = {
                "pipeline": LINALG_TENSOR_NORMALIZE_PIPELINE,
                "before_generic_ops": before.get("linalg.generic", 0),
                "after_generic_ops": common["operation_counts"].get("linalg.generic", 0),
                "performance_measured": False,
            }
            report["completed_stage"] = "common"
            if stop_after == "common":
                report["completed_stage"] = "common"
            else:
                entries = [item for item in common["functions"] if item["defined"] and item["visibility"] == "public"]
                if len(entries) != 1 or len(common["functions"]) != 1:
                    raise NativeCompileError("AIE compilation requires one defined entry; external calls and multiple public entries are unsupported")
                memref_pipeline = linalg_tensor_to_memref_pipeline(common["operation_counts"])
                report["bufferization"] = {"pipeline": memref_pipeline}
                stage("01-memref", "aiec-opt", ["--pass-pipeline=" + memref_pipeline])
                # Every element type bufferizes; the AIE backend after it takes FP32.
                if not entries[0]["static_fp32_abi"]:
                    raise NativeCompileError("First AIE backend requires a static ranked FP32 tensor/buffer ABI")
                if policy.get("operator_pipeline") == "cnn":
                    annotated = out / "01-cnn-policy.mlir"
                    automatic = policy.get('tiling_strategy') == 'automatic'
                    extra = ['--l1-bytes', policy['l1_bytes'], '--stack-bytes', policy['stack_bytes'],
                             '--tiling-strategy', policy.get('tiling_strategy', 'baseline')]
                    if automatic:
                        extra += ['--resource-contract', out/'resource-contract.json']
                    else:
                        extra += ['--max-cores', policy.get('max_cores',4), '--herd-axis-limits', *policy.get('herd_axis_limits',[2,2]),
                                  '--conv-distribution', policy.get('conv_distribution', 'n_oc')]
                        extra += ['--conv-dataflow', policy.get('conv_dataflow', 'tiled')]
                        # The async schedule's ping-pong transform double-buffers loops whose
                        # buffers fit twice; the plan leaves room for it.
                        extra += ['--max-stages', 2 if policy.get('air_schedule') == 'asynchronous' else 1]
                        if policy.get('conv_dataflow') == 'resident':
                            # Herd sizes from the array the herds are placed in (columns, core
                            # rows from the anchor), not preset counts; an explicit policy
                            # value only caps them.
                            target = report['device_target']
                            columns = target['columns'] - policy['col_anchor']
                            rows = target['rows'] - policy['row_anchor']
                            limits = [min(columns, policy.get('herd_axis_limits', [columns, rows])[0]),
                                      min(rows, policy.get('herd_axis_limits', [columns, rows])[1])]
                            cores = min(columns * rows, policy.get('max_cores', columns * rows))
                            extra[extra.index('--max-cores') + 1] = cores
                            at = extra.index('--herd-axis-limits')
                            extra[at + 1:at + 3] = limits
                        if policy.get('conv_distribution') == 'spatial' or policy.get('conv_dataflow') == 'resident':
                            # Packet IDs bound native packet routing; dedicated
                            # circuits use the generated platform endpoint budget.
                            from compiler.platform import gmio_endpoints
                            endpoints = gmio_endpoints(policy['platform_contract'])
                            budget = ({'total': report['device_target']['packet_ids']} if policy.get('transport') == 'packet'
                                      else {'inputs': len(endpoints['input']), 'outputs': len(endpoints['output'])})
                            extra += ['--stream-budget', json.dumps(budget)]
                            stream_budget = budget
                    report["cnn_policy_preparation"] = _native_python("prepare-cnn", current, annotated, config, logs,
                        extra=extra)
                    current = annotated
                    cnn_plan = json.loads(annotated.with_suffix(".policy.json").read_text())
                    report["cnn_policy"] = cnn_plan
                    transform_path = ROOT / 'policies/cnn.mlir'
                    if automatic:
                        from aiec_air.automatic_fix import write_transform
                        transform_path = out/'01-automatic-transform.mlir'
                        write_transform(cnn_plan, transform_path)
                    transform = json.dumps(str(transform_path))
                    report["cnn_transform_policy"] = {
                        "path": str(transform_path),
                        "sha256": hashlib.sha256(transform_path.read_bytes()).hexdigest(),
                    }
                    stage("01-cnn-generalized", "air-opt", ["--pass-pipeline=builtin.module(transform-preload-library{transform-library-paths=" + transform + "},transform-interpreter)"])
                    # Operations distributed over other loops than the first two:
                    # a native interchange puts those loops first, where
                    # air-linalg-codegen's herd tiling applies.
                    resident = [op for op in cnn_plan["operations"] if op["memory_plan"].get("dataflow") == "resident"]
                    if resident:
                        script = out / "01-cnn-resident.transform.mlir"
                        script.write_text(_resident_conv_transform(resident))
                        report["cnn_resident_conv"] = {"path": str(script), "operations": len(resident)}
                        stage("01-cnn-resident", "air-opt", ["--pass-pipeline=builtin.module(transform-preload-library{transform-library-paths=" + json.dumps(str(script)) + "},transform-interpreter,canonicalize,cse)"])
                        # Bias initialization and elementwise epilogue into the conv cores.
                        from compiler.target_spec import CORE_STATIC_BYTES, DMA_CHANNELS_PER_DIRECTION, MODULE_BYTES
                        fusion_report = out / "01-cnn-fused-blocks.fusion.json"
                        options = (f"module-bytes={MODULE_BYTES} stack-bytes={policy['stack_bytes']} "
                                   f"static-bytes={CORE_STATIC_BYTES} channels-per-direction={DMA_CHANNELS_PER_DIRECTION} "
                                   f"stream-total={stream_budget.get('total', 0)} stream-inputs={stream_budget.get('inputs', 0)} "
                                   f"stream-outputs={stream_budget.get('outputs', 0)} report={fusion_report}")
                        stage("01-cnn-fused-blocks", "aiec-opt", ["--aiec-fuse-conv-blocks=" + options])
                        fusion = json.loads(fusion_report.read_text())
                        report["conv_block_fusion"] = fusion
                        cnn_plan["operations"] = [op for op in cnn_plan["operations"]
                                                  if op["marker"] not in set(fusion["fused_markers"])]
                    permuted = [op for op in cnn_plan["operations"] if not automatic and
                                op["memory_plan"].get("dataflow") != "resident" and
                                op["memory_plan"].get("core_group", {}).get("loop_permutation") not in
                                (None, list(range(len(op["tile_sizes"]))))]
                    if permuted:
                        interchange = out / "01-cnn-interchange.mlir"
                        lines = ['module attributes {transform.with_named_sequence} {',
                                 'transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {']
                        for i, op in enumerate(permuted):
                            order = op["memory_plan"]["core_group"]["loop_permutation"]
                            lines += [f'%m{i} = transform.structured.match attributes {{__internal_linalg_transform__ = {json.dumps(op["marker"])}}} in %root : (!transform.any_op) -> !transform.any_op',
                                      f'%p{i} = transform.structured.interchange %m{i} iterator_interchange = [{", ".join(map(str, order))}] : (!transform.any_op) -> !transform.any_op']
                        lines += ['transform.yield', '}', '}']
                        interchange.write_text("\n".join(lines) + "\n")
                        report["cnn_interchange"] = {"path": str(interchange), "operations": len(permuted)}
                        stage("01-cnn-interchanged", "air-opt", ["--pass-pipeline=builtin.module(transform-preload-library{transform-library-paths=" + json.dumps(str(interchange)) + "},transform-interpreter)"])
                    transforms = []
                    for operation in cnn_plan["operations"]:
                        if operation["memory_plan"].get("dataflow") == "resident":
                            continue  # tiled and promoted by 01-cnn-resident
                        order = operation["memory_plan"].get("core_group", {}).get("loop_permutation") if not automatic else None
                        sizes = operation["tile_sizes"] if order is None else [operation["tile_sizes"][d] for d in order]
                        tile = ",".join(str(x) for x in sizes)
                        herd = ",".join(str(x) for x in operation["herd_size"])
                        transforms.append("air-linalg-codegen{input-filter=" + operation["marker"] + " l1-promote=true l2-promote=false l1-size=" + str(policy['l1_bytes']-policy['stack_bytes']) + " herd-size=" + herd + " l1-tile-size=" + tile + "}")
                        if automatic:
                            transforms.append('air-par-to-herd{depth=-1 first-dim='+str(operation['memory_plan']['core_group']['first_dim'])+'}')
                    if not automatic:
                        if resident:
                            transforms.append("func.func(scf-forall-to-parallel)")
                        transforms.append("air-par-to-herd{depth=-1 first-dim=" + str(policy.get('herd_first_dim', 0)) + "}")
                    if automatic:
                        stage('02-promoted', 'air-opt', ['--pass-pipeline=builtin.module('+','.join(transforms)+',canonicalize,cse)'])
                        # Proven uniform initial values fill promoted accumulators in L1 (aiec-opt pass).
                        residency_report = out / '02-resident.residency.json'
                        stage('02-resident', 'aiec-opt', [f'--aiec-localize-accumulators=report={residency_report}',
                                                          '--canonicalize', '--cse'])
                        report['accumulator_residency'] = json.loads(residency_report.read_text())
                        transforms = []
                    transforms.extend(["air-copy-to-dma", "air-insert-launch-around-herd{insert-segment=true}", "canonicalize", "cse"])
                    stage("02-air", "air-opt", ["--pass-pipeline=builtin.module(" + ",".join(transforms) + ")"])
                else:
                    tile = ",".join(str(x) for x in policy["tile_sizes"])
                    stage("02-air", "air-opt", ["--pass-pipeline=builtin.module(air-linalg-codegen{l1-promote=true l2-promote=false l1-size=" + str(policy['l1_bytes']-policy['stack_bytes']) + " l1-tile-size=" + tile + "},air-par-to-herd{depth=-1},air-copy-to-dma,air-insert-launch-around-herd{insert-segment=true},canonicalize,cse)"])
                current = _lower_air(current, out, policy, toolchain, config, report, stage, inspect,
                                     stop_after=stop_after, partition_programs=partition_programs)
        report["status"] = "compiled_ir"
        report["output"] = str(current)
    except BaseException as exc:
        from compiler.diagnostics import explain_native_failure
        diagnostic = explain_native_failure(out)
        if diagnostic:
            report['backend_blocker'] = diagnostic
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report["last_successful_output"] = str(current)
        (out / "compilation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
