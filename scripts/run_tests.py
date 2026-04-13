#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestCase:
    name: str
    cuda_exe: str
    cuda_verify_args: list[str]
    cuda_bench_args: list[str]
    py_script: str
    py_bench_args: list[str]


ROOT = Path(__file__).resolve().parents[1]

TEST_CASES = [
    TestCase("test_relu", "tests/test_relu", ["4096"], ["1048576"], "tests/cuda_core/activation/test_relu.py", ["1048576"]),
    TestCase("test_gelu", "tests/test_gelu", ["4096"], ["1048576"], "tests/cuda_core/activation/test_gelu.py", ["1048576"]),
    TestCase("test_rms_norm", "tests/test_rms_norm", ["16", "128"], ["128", "128"], "tests/cuda_core/normalization/test_rms_norm.py", ["128", "128"]),
    TestCase("test_sgemv", "tests/test_sgemv", ["256", "128"], ["1024", "128"], "tests/cuda_core/blas/test_sgemv.py", ["1024", "128"]),
    TestCase("test_softmax", "tests/test_softmax", ["16", "256"], ["128", "256"], "tests/cuda_core/normalization/test_softmax.py", ["128", "256"]),
    TestCase("test_cumsum", "tests/test_cumsum", ["1048576"], ["1048576"], "tests/cuda_core/scan/test_cumsum.py", ["1048576"]),
    TestCase("test_sgemm", "tests/test_sgemm", ["256", "256", "256"], ["512", "512", "512"], "tests/cuda_core/blas/test_sgemm.py", ["512", "512", "512"]),
]

TEST_CASE_BY_NAME = {case.name: case for case in TEST_CASES}
TEST_CASE_BY_STEM = {}
for case in TEST_CASES:
    TEST_CASE_BY_STEM[Path(case.cuda_exe).stem] = case
    TEST_CASE_BY_STEM[Path(case.py_script).stem] = case

CUDA_BENCH_RE = re.compile(
    r"^BENCH\s+(?P<name>\S+)\s+gpu_ms=(?P<gpu>[0-9.eE+-]+)$",
    re.MULTILINE,
)
PY_BENCH_RE = re.compile(
    r"^PYBENCH\s+(?P<name>\S+)\s+eager_ms=(?P<eager>[0-9.eE+-]+)\s+inductor_ms=(?P<inductor>[0-9.eE+-]+)$",
    re.MULTILINE,
)


def run_command(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode, proc.stdout


def resolve_case(token: str) -> TestCase | None:
    if token == "all":
        return None
    key = Path(token).stem if ("/" in token or token.endswith(".cu") or token.endswith(".py")) else token
    return TEST_CASE_BY_NAME.get(key) or TEST_CASE_BY_STEM.get(key)


def select_cases(targets: str) -> list[TestCase]:
    tokens = [item.strip() for item in targets.split(",") if item.strip()]
    if not tokens:
        raise SystemExit("no targets selected")
    if tokens == ["all"]:
        return TEST_CASES

    cases: list[TestCase] = []
    seen: set[str] = set()
    for token in tokens:
        case = resolve_case(token)
        if case is None:
            raise SystemExit(f"unknown target: {token}")
        if case.name not in seen:
            cases.append(case)
            seen.add(case.name)
    return cases


def parse_cuda_bench(stdout: str) -> float:
    match = CUDA_BENCH_RE.search(stdout)
    if not match:
        raise SystemExit("could not parse CUDA benchmark output")
    return float(match.group("gpu"))


def parse_py_bench(stdout: str) -> tuple[float, float]:
    match = PY_BENCH_RE.search(stdout)
    if not match:
        raise SystemExit("could not parse Python benchmark output")
    return float(match.group("eager")), float(match.group("inductor"))


def print_summary_table(rows: list[tuple[str, float | None, float | None, float | None]]) -> None:
    headers = ("case", "cuda_ms", "eager_ms", "inductor_ms")
    table_rows: list[tuple[str, str, str, str]] = []
    for name, cuda_ms, eager_ms, inductor_ms in rows:
        cuda_str = "-" if cuda_ms is None else f"{cuda_ms:.4f}"
        eager_str = "failed" if eager_ms is None else f"{eager_ms:.4f}"
        inductor_str = "failed" if inductor_ms is None else f"{inductor_ms:.4f}"
        table_rows.append((name, cuda_str, eager_str, inductor_str))

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: tuple[str, str, str, str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    header_line = fmt_row(headers)
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    print(header_line)
    print(sep_line)
    for row in table_rows:
        print(fmt_row(row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenKernels tests in correctness-then-performance order.")
    parser.add_argument("--build-dir", default="build", help="CMake build directory.")
    parser.add_argument("--target", default="all", help="all, test name, or path to a single .cu/.py file.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for benchmark scripts.")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations for CUDA and Python.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations for Python benchmarks.")
    parser.add_argument("--device", default="cuda", help="Device for Python benchmark scripts.")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    if not build_dir.exists():
        print(f"build dir not found: {build_dir}", file=sys.stderr)
        return 2

    try:
        cases = select_cases(args.target)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cuda_failures: list[str] = []
    py_failures: list[str] = []
    rows: list[tuple[str, float | None, float | None, float | None]] = []

    for idx, case in enumerate(cases, start=1):
        cuda_exe = build_dir / case.cuda_exe
        py_script = ROOT / case.py_script
        if not cuda_exe.exists():
            print(f"[{idx}/{len(cases)}] missing executable: {cuda_exe}", file=sys.stderr)
            cuda_failures.append(case.name)
            continue
        if not py_script.exists():
            print(f"[{idx}/{len(cases)}] missing python script: {py_script}", file=sys.stderr)
            py_failures.append(case.name)
            continue

        print(f"[{idx}/{len(cases)}] verify {case.name}")
        verify_cmd = [str(cuda_exe), "--verify", *case.cuda_verify_args]
        rc, _ = run_command(verify_cmd)
        if rc != 0:
            cuda_failures.append(case.name)
            continue

        print(f"[{idx}/{len(cases)}] bench {case.name} (CUDA)")
        cuda_cmd = [str(cuda_exe), "--bench", "--iters", str(args.iters), *case.cuda_bench_args]
        rc, stdout = run_command(cuda_cmd)
        if rc != 0:
            cuda_failures.append(case.name)
            continue
        cuda_ms = parse_cuda_bench(stdout)

        print(f"[{idx}/{len(cases)}] bench {case.name} (PyTorch)")
        py_cmd = [
            args.python,
            str(py_script),
            *case.py_bench_args,
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--device",
            args.device,
        ]
        rc, stdout = run_command(py_cmd)
        if rc != 0:
            py_failures.append(case.name)
            rows.append((case.name, cuda_ms, None, None))
            continue

        eager_ms, inductor_ms = parse_py_bench(stdout)
        rows.append((case.name, cuda_ms, eager_ms, inductor_ms))

    print("\nSummary")
    print_summary_table(rows)

    if cuda_failures:
        print(f"cuda failed: {', '.join(cuda_failures)}", file=sys.stderr)
    if py_failures:
        print(f"python failed: {', '.join(py_failures)}", file=sys.stderr)

    if cuda_failures or py_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
