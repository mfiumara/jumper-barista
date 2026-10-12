"""Bounded local native prerequisite build; SDK authorization must precede use.

Run from the repository root. Never fetch the Rockchip runtime. Source acquisition
and licence review are documented in NATIVE_PREREQUISITES.md, not automated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out/toolchain"
RECEIPTS = BASE / "receipts"
PREFIX = BASE / "cyclonedds-install"
SOURCE = BASE / "cyclonedds-source"
COMMIT = "e54e991f75a3e67f8e628da3171122e36ea5b872"
HEADER_HASH = "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"


def run(name: str, args: list[str], env: dict[str, str] | None = None) -> None:
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    log = RECEIPTS / f"{name}.log"
    log.write_text(result.stdout + result.stderr)
    with (RECEIPTS / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"name": name, "argv": args, "cwd": str(ROOT),
                                 "exit": result.returncode, "log": str(log)}) + "\n")
    print(f"{name}: exit {result.returncode}; {log}", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["build", "verify", "test"])
    parser.add_argument("--python", help="Existing Python 3.10–3.13 with pytest, read-only")
    args = parser.parse_args()
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PATH=f"{PREFIX / 'bin'}:{env['PATH']}", CYCLONEDDS_HOME=str(PREFIX),
               CARGO_TARGET_DIR=str(BASE / "cargo-target"), CARGO_BUILD_JOBS="1",
               CARGO_NET_OFFLINE="true", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=f"{ROOT}:{ROOT / 'rl'}", TMPDIR=str(BASE / "tmp"))
    (BASE / "tmp").mkdir(exist_ok=True)
    if args.phase == "build":
        identity = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
        if identity != COMMIT:
            raise SystemExit(f"Unexpected CycloneDDS commit: {identity}")
        run("configure", ["cmake", "-S", str(SOURCE), "-B", str(BASE / "cyclonedds-build"),
                          "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={PREFIX}",
                          "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DBUILD_TESTING=OFF",
                          "-DBUILD_EXAMPLES=OFF", "-DBUILD_IDLC=ON"], env)
        run("build", ["cmake", "--build", str(BASE / "cyclonedds-build"), "--parallel", "2"], env)
        run("install", ["cmake", "--install", str(BASE / "cyclonedds-build")], env)
    elif args.phase == "verify":
        header = ROOT / "deploy/fsm/vendor/rknpu2/include/rknn_api.h"
        if hashlib.sha256(header.read_bytes()).hexdigest() != HEADER_HASH:
            raise SystemExit("Rockchip header hash mismatch")
        run("idlc-version", [str(PREFIX / "bin/idlc"), "-v"], env)
        paths = [PREFIX / "bin/idlc", *sorted((PREFIX / "lib").glob("*.dylib"))]
        run("architecture", ["file", *map(str, paths)], env)
        run("dependencies", ["otool", "-L", *map(str, paths)], env)
        run("probe-compile", ["clang", "-arch", "arm64", str(ROOT / "barista/native_dds_probe.c"),
                              f"-I{PREFIX / 'include'}", f"-L{PREFIX / 'lib'}", "-lddsc",
                              f"-Wl,-rpath,{PREFIX / 'lib'}", "-o", str(BASE / "native-dds-probe")], env)
        run("probe-run", [str(BASE / "native-dds-probe")], env)
        run("probe-dependencies", ["otool", "-L", str(BASE / "native-dds-probe")], env)
        paths += [header, *sorted((PREFIX / "include").rglob("*.h"))]
        (RECEIPTS / "installed-sha256.txt").write_text("".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n" for p in paths))
    else:
        if not args.python:
            parser.error("test requires --python")
        (RECEIPTS / "test-environment.json").write_text(json.dumps(
            {k: env[k] for k in ["PATH", "CYCLONEDDS_HOME", "CARGO_TARGET_DIR",
                                 "CARGO_BUILD_JOBS", "CARGO_NET_OFFLINE", "PYTHONPATH", "TMPDIR"]}, indent=2))
        run("provenance-test", [args.python, "-m", "pytest",
                               "tests/test_deploy_sources.py::test_the_provenance_dates_everything_the_controller_is_built_from",
                               "-q", "-p", "no:cacheprovider", f"--basetemp={BASE / 'pytest-temp'}"], env)


if __name__ == "__main__":
    main()
