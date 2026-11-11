"""Build and run the internal planner-contract test using an existing build.

Reuses the compile flags recorded for form_device_program.cc in
build/compile_commands.json, so the test sees exactly the same headers,
defines and include paths as the production translation unit.

Usage: python testing/cpp/run_tenstorrent_planner_contract.py [build-directory]
"""

import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
build = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "build"
entries = json.loads((build / "compile_commands.json").read_text())
entry = next(
    item for item in entries
    if item["file"].endswith("/tenstorrent/transform/form_device_program.cc")
)
arguments = entry.get("arguments") or shlex.split(entry["command"])
flags = []
skip = False
for argument in arguments:
    if skip:
        skip = False
        continue
    if argument == "-o":
        skip = True
        continue
    if argument == "-c" or argument == entry["file"]:
        continue
    flags.append(argument)
output = build / "test_tenstorrent_planner_contract"
subprocess.run(
    flags + [
        "-O0", "-ffunction-sections", "-fdata-sections",
        str(root / "testing/cpp/test_tenstorrent_planner_contract.cc"),
        "-Wl,--gc-sections", f"-L{build / 'lib'}",
        f"-Wl,-rpath,{build / 'lib'}", "-ltilelang", "-ltvm_compiler",
        "-ltvm_runtime", "-ltvm_ffi", "-o", str(output),
    ],
    cwd=entry["directory"], check=True,
)
subprocess.run([str(output)], cwd=root, check=True)
