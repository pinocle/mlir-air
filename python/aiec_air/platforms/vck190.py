"""Platform transport capabilities, distinct from the AIE1 silicon limits.

GMIO endpoints come from Vitis metadata, not the Harness PLIO constraint file.
An AIE-compiled contract permits IR preparation; it is not a bootable platform.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from aiec_air.toolchain.environment import ROOT

BASE_NAME = "xilinx_vck190_base_202520_1"
BASE_ID = "vck190-base-gmio-2025.2"
GMIO_TRANSPORT = "gmio_dedicated_circuit"
DEFAULT_CONTRACT = ROOT / "artifacts/vck190-base-gmio-2025.2/transport-contract.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_base(xpfm):
    xpfm = Path(xpfm).resolve(strict=True)
    if xpfm.stem != BASE_NAME:
        raise ValueError("Expected the VCK190 2025.2 base platform")
    xsa = xpfm.parent / "hw/hw.xsa"
    with zipfile.ZipFile(xsa) as archive:
        tree = ET.fromstring(archive.read("xsa.xml"))
    board = next(e for e in tree.iter() if e.tag.split('}')[-1] == "Board")
    build = next(e for e in tree.iter() if e.tag.split('}')[-1] == "Build")
    if not board.attrib.get("Part", "").startswith("xcvc1902-"):
        raise ValueError("Base XSA does not target VC1902")
    return {"name": BASE_NAME, "xpfm": str(xpfm), "xpfm_sha256": digest(xpfm),
            "xsa": str(xsa), "xsa_sha256": digest(xsa), "part": board.attrib["Part"],
            "platform_state": build.attrib.get("PlatformState"),
            "requires_system_link": True}


def gmio_contract(metadata, xpfm):
    metadata = Path(metadata).resolve(strict=True)
    data = json.loads(metadata.read_text())["aie_metadata"]
    driver = data["driver_config"]
    if (driver.get("hw_gen"), driver.get("num_columns"), driver.get("num_rows")) != (1, 50, 9):
        raise ValueError("GMIO metadata must describe the VCK190 AIE1 array")
    entries = data["GMIOs"]
    entries = entries.values() if isinstance(entries, dict) else entries
    endpoints = {"input": [], "output": []}
    physical, names = set(), set()
    for port in entries:
        direction = {0: "input", 1: "output"}.get(port["type"])
        col, channel = port["shim_column"], port["channel_number"]
        # Column legality is checked against the native target when routing.
        if direction is None or type(col) is not int or not 0 <= col < 50 or type(channel) is not int or channel not in (0, 1):
            raise ValueError("Invalid AIE1 GMIO direction/column/DMA channel")
        key = direction, col, channel
        if key in physical or port["name"] in names:
            raise ValueError("Duplicate physical GMIO endpoint or logical name")
        physical.add(key); names.add(port["name"])
        endpoints[direction].append({"name": port["name"], "column": col, "channel": channel,
                                    "bundle": "DMA", "burst_bytes": port["burst_length_in_16byte"] * 16})
    for direction, ports in endpoints.items():
        if not ports or len(ports) > 32:
            raise ValueError("Expected 1..32 GMIO endpoints in each direction")
        ports.sort(key=lambda p: (p["column"], p["channel"]))
        for index, port in enumerate(ports, 1):
            port["id"] = index
            if port["burst_bytes"] not in (64, 128, 256):
                raise ValueError("Unsupported AIE1 GMIO burst length")
    return {"schema_version": 1, "platform": BASE_ID, "target": "xcvc1902",
            "transport": GMIO_TRANSPORT, "state": "aie_compiled",
            "base": inspect_base(xpfm), "metadata": str(metadata), "metadata_sha256": digest(metadata),
            "endpoints": endpoints, "alignment_bytes": 4,
            "runtime_owner": "libxaie", "system_linked": False, "board_verified": False}


def load_gmio_contract(path):
    contract = json.loads(Path(path).read_text())
    if contract.get("platform") != BASE_ID or contract.get("transport") != GMIO_TRANSPORT:
        raise ValueError("Unknown GMIO platform contract")
    if contract.get("schema_version") != 1 or contract.get("target") != "xcvc1902":
        raise ValueError("Invalid GMIO platform target/schema")
    original = gmio_contract(contract["metadata"], contract["base"]["xpfm"])
    for key in ("base", "metadata_sha256", "endpoints", "alignment_bytes", "runtime_owner"):
        if contract.get(key) != original[key]:
            raise ValueError("GMIO platform contract/source mismatch: " + key)
    return contract


def gmio_endpoints(path):
    contract = load_gmio_contract(path)
    # `plio` remains the legacy host ABI's stream identity, not a PL interface.
    # The bundle determines which native flow is emitted; no PLIF is enabled.
    return {direction: [dict(p, plio=p["id"], plio_index=p["id"] - 1,
                            direction="slave" if direction == "input" else "master")
                        for p in ports] for direction, ports in contract["endpoints"].items()}


def require_packaged(contract):
    if not contract.get('system_linked') or contract.get('state') != 'packaged':
        raise ValueError('Base GMIO hardware/Linux packaging is not complete; cannot deploy to the Harness image')
    for key in ('linked_xsa', 'runtime_xclbin', 'boot_image'):
        if not contract.get(key) or not Path(contract[key]).is_file() or digest(contract[key]) != contract.get(key+'_sha256'):
            raise ValueError('Missing or modified base platform deployment artifact: '+key)
    return contract


def activate_contract(path, destination=DEFAULT_CONTRACT):
    """Publish a verified local build for the default policies, without booting it."""
    contract = require_packaged(load_gmio_contract(path))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.json.new')
    temporary.write_text(json.dumps(contract, indent=2)+'\n')
    temporary.replace(destination)
    return destination
