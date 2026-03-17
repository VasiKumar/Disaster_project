from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement


def normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")


def parse_requirements(path: Path) -> list[str]:
    requirements: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


def find_missing_or_incompatible(req_lines: list[str]) -> list[str]:
    to_install: list[str] = []
    for req_line in req_lines:
        req = Requirement(req_line)
        pkg_name = normalize_name(req.name)
        try:
            installed = version(req.name)
        except PackageNotFoundError:
            to_install.append(req_line)
            print(f"MISSING   : {req_line}")
            continue

        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            to_install.append(req_line)
            print(f"MISMATCH  : {req_line} (installed {installed})")
        else:
            print(f"OK        : {pkg_name}=={installed}")

    return to_install


def main() -> None:
    parser = argparse.ArgumentParser(description="Install only missing/incompatible Python packages")
    parser.add_argument("--requirements", default="requirements.txt")
    args = parser.parse_args()

    req_path = Path(args.requirements)
    req_lines = parse_requirements(req_path)
    to_install = find_missing_or_incompatible(req_lines)

    if not to_install:
        print("\nAll requirements already satisfied. No install needed.")
        return

    print("\nInstalling only missing/incompatible packages:")
    for req in to_install:
        print(f"  - {req}")

    cmd = [sys.executable, "-m", "pip", "install", *to_install]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
