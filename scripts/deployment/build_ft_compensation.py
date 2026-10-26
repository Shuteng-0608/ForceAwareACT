"""Build the vendored mathematical compensation core; no device access."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def build(output, *, dependency_prefix=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output exists; use a new build path")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT / "third_party/tianji_ft"
    prefix = Path(dependency_prefix).resolve() if dependency_prefix else Path('/usr')
    library_dirs = [prefix/'lib/x86_64-linux-gnu', prefix/'lib']
    library_flags = [flag for directory in library_dirs if directory.is_dir()
                     for flag in ('-L'+str(directory), '-Wl,-rpath,'+str(directory))] if dependency_prefix else []
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-shared", "-fPIC", "-I", str(source / "include"),
                    "-I", str(prefix/"include/eigen3"), "-I", str(prefix/"include"), str(source / "src/ft_compensation.cpp"),
                    str(source / "compensation_c_api.cpp"), *library_flags, "-lyaml-cpp", "-o", str(output)], check=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/ft_compensation.so")
    parser.add_argument("--dependency-prefix", type=Path, help="local Eigen/yaml-cpp prefix; no system install needed")
    args=parser.parse_args()
    print(build(args.output, dependency_prefix=args.dependency_prefix))
