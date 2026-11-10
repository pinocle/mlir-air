"""Native synchronous AIR compile projections; not a model execution schedule.

Run with the AIR environment. Each projection retains the host function/ABI,
and isolates one launch so AIRToAIE's module-local packet allocator starts fresh.
The original mixed module is the authoritative program, not any projection.
"""

import argparse
import hashlib
import json
from pathlib import Path


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def partition_air(source, out_dir, *, selected_ids=None):
    """Return a manifest of verified per-launch MLIR files in a new directory.

    Reject hierarchy results other than an async token, channel edges
    shared by launches, and arithmetic outside herds. No numeric packet IDs are
    rewritten. Native AIRToAIE must be invoked independently on each file.
    selected_ids is for bounded compilation/probing, never changes source IDs.
    """
    from air import ir
    import air.dialects.air  # noqa: F401 -- native dialect registration
    from compiler.ir_tools import inspect_file, walk

    source, out_dir = Path(source).absolute(), Path(out_dir).absolute()
    if out_dir.exists():
        raise FileExistsError(f"Partition output already exists: {out_dir}")
    inspection = inspect_file(source, "air")
    for function in inspection["functions"]:
        if function["defined"] and function["visibility"] == "public" and not function["static_fp32_abi"]:
            raise ValueError("Public AIR entry must have a static ranked FP32 buffer ABI: " + function["name"])
    if inspection["unmapped_computation"]:
        raise ValueError("Arithmetic outside AIR herd is unsupported: " +
                         ", ".join(inspection["unmapped_computation"][:8]))
    data = source.read_bytes()
    with ir.Context() as context:
        context.load_all_available_dialects()
        module = ir.Module.parse(data.decode())
        if not module.operation.verify():
            raise ValueError("Source AIR native verification failed")
        launches = [op for op, _ in walk(module.operation) if op.name == "air.launch"]
        if not launches:
            raise ValueError("No AIR launches to partition")
        declarations = {op.attributes["sym_name"].value: op
                        for op, _ in walk(module.operation) if op.name == "air.channel"}
        owners, descriptions = {}, []
        for source_id, launch in enumerate(launches):
            ops = list(walk(launch))
            hierarchy = [op for op, _ in ops if op.name in {"air.launch", "air.segment", "air.herd"}]
            # Async hierarchy (aircc's dependency schedule) yields only a
            # completion token; the source order of launches is a valid
            # topological order of those tokens.
            if any(len(op.results) > 1 or any("async.token" not in str(r.type) for r in op.results)
                   for op in hierarchy):
                raise ValueError("AIR hierarchy may only yield an async completion token")
            segments = [op for op, _ in ops if op.name == "air.segment"]
            herds = [op for op, _ in ops if op.name == "air.herd"]
            if len(segments) != 1 or not herds or any(op.name == "air.launch" for op, _ in ops[1:]):
                raise ValueError("Each launch must contain one segment and at least one herd")
            channels = sorted({ir.FlatSymbolRefAttr(op.attributes["chan_name"]).value
                               for op, _ in ops if op.name in {"air.channel.put", "air.channel.get"}})
            for name in channels:
                if name not in declarations:
                    raise ValueError(f"Missing channel declaration: {name}")
                if name in owners:
                    raise ValueError(f"Cross-launch channel {name} joins source launches {owners[name]} and {source_id}")
                owners[name] = source_id
            descriptions.append({"source_order_id": source_id,
                "segment": segments[0].attributes["sym_name"].value,
                "herd": herds[0].attributes["sym_name"].value,
                "herds": [h.attributes['sym_name'].value for h in herds],
                "source_launches":list(ir.DenseI64ArrayAttr(launch.attributes['aie1.source_launches'])) if 'aie1.source_launches' in launch.attributes else [source_id],
                "device_dataflow":ir.StringAttr(launch.attributes['aie1.group_transport']).value if 'aie1.group_transport' in launch.attributes else None,
                "channels": channels})
        for op, parents in walk(module.operation):
            if op.name in {"air.channel.put", "air.channel.get"} and "air.launch" not in parents:
                raise ValueError("Channel endpoint outside a launch is unsupported")
        selected = list(range(len(launches))) if selected_ids is None else list(selected_ids)
        if not selected or len(set(selected)) != len(selected) or any(
                type(i) is not int or i < 0 or i >= len(launches) for i in selected):
            raise ValueError("selected_ids must contain unique valid source launch IDs")
        # Source order is provenance, not an inferred execution schedule.
        selected.sort()
        out_dir.mkdir(parents=True)
        original = out_dir / "source-mixed.mlir"
        original.write_bytes(data)
        manifest = {"schema_version": 1, "source": str(source), "source_sha256": _sha(data),
            "preserved_source": str(original), "projection_only": True,
            "runtime_schedule_generated": False, "executable_image_generated": False,
            "platform_runtime_verified": False, "board_execution_performed": False,
            "source_launch_count": len(launches), "programs": [],
            "packet_id_scope": "independent_native_air_to_aie_invocation_per_program",
            "host_initialization": inspection["host_initialization"],
            "functions": inspection["functions"]}
        from compiler.tools import aiec_opt
        for source_id in selected:
            kept_channels = set(descriptions[source_id]["channels"])
            # The projection itself is a native pass (aiec-project-launch);
            # only semantics-preserving cleanup removes unused values and
            # globals. Host side effects and source constant initialization stay.
            text = aiec_opt(data.decode(), f"--aiec-project-launch=index={source_id}", "--canonicalize", "--cse",
                            "--symbol-dce")
            # Reparse serialization, including dense_resource constants.
            reparsed = ir.Module.parse(text)
            if not reparsed.operation.verify():
                raise ValueError(f"Projection {source_id} failed native verification")
            projected = [op for op, _ in walk(reparsed.operation) if op.name == "air.launch"]
            channels = {op.attributes["sym_name"].value for op, _ in walk(reparsed.operation) if op.name == "air.channel"}
            if len(projected) != 1 or channels != kept_channels:
                raise ValueError(f"Projection {source_id} does not hold exactly its launch and channels")
            destination = out_dir / f"program-{source_id:04d}.mlir"
            destination.write_text(text)
            program = dict(descriptions[source_id], path=str(destination),
                sha256=_sha(text.encode()), native_verifier="passed", projection_only=True,
                channel_mapping={name: name for name in sorted(kept_channels)})
            manifest["programs"].append(program)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--select", nargs="+", type=int)
    args = parser.parse_args()
    result = partition_air(args.source, args.out_dir, selected_ids=args.select)
    print(f"Verified {len(result['programs'])} native AIR compile projections; runtime schedule not generated")


if __name__ == "__main__":
    main()
