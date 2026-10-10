"""Local measurement of revised CSR entry points against the published primitive."""

import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import quadrants as qd

ROOT = Path(__file__).resolve().parent
original_path = ROOT / "original-source/python/quadrants/algorithms/_csr.py"
spec = importlib.util.spec_from_file_location("published_csr_baseline", original_path)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


@qd.kernel
def old_apply(
    row_ptr: qd.types.NDArray,
    col_idx: qd.types.NDArray,
    values: qd.types.NDArray,
    rhs: qd.types.NDArray,
    out: qd.types.NDArray,
):
    original.csr_spmm(row_ptr, col_idx, values, rhs, out)


@qd.kernel
def new_apply(matrix: qd.linalg.CSRMatrix, rhs: qd.types.NDArray, out: qd.types.NDArray):
    (matrix @ rhs).write_to(out)


@qd.kernel
def dense_apply(weights: qd.types.NDArray, rhs: qd.types.NDArray, out: qd.types.NDArray):
    for row in range(out.shape[0]):
        trial = row // 512
        receiver = row % 512
        accumulator = qd.f32(0.0)
        for source in range(512):
            accumulator += weights[trial, source, receiver] * rhs[trial * 512 + source, 0]
        out[row, 0] = accumulator


def upload(data):
    array = qd.ndarray(qd.i32 if data.dtype == np.int32 else qd.f32, shape=data.shape)
    array.from_numpy(data)
    return array


def main():
    qd.init(arch=qd.metal, enable_fallback=False, fast_math=False, debug=False, offline_cache=False)
    rng = np.random.default_rng(7)
    trials, n, degree, steps = 32, 512, 16, 256
    rows = trials * n
    pointers = np.arange(rows + 1, dtype=np.int32) * degree
    indices = np.empty((rows, degree), np.int32)
    values = rng.uniform(-0.05, 0.05, (rows, degree)).astype(np.float32)
    dense = np.zeros((trials, n, n), np.float32)
    rhs_host = rng.uniform(-1, 1, (rows, 1)).astype(np.float32)
    reference = np.empty((rows, 1), np.float64)
    for row in range(rows):
        trial, receiver = divmod(row, n)
        local = np.sort(rng.choice(n, degree, replace=False))
        indices[row] = local + trial * n
        dense[trial, local, receiver] = values[row]
        reference[row, 0] = np.sum(values[row].astype(np.float64) * rhs_host[indices[row], 0].astype(np.float64))
    rp, ci, v, rhs, weights = [upload(data) for data in (pointers, indices.reshape(-1), values.reshape(-1), rhs_host, dense)]
    matrix = qd.linalg.SparseMatrix.from_csr(rp, ci, v, shape=(rows, rows))
    outputs = {name: qd.ndarray(qd.f32, shape=(rows, 1)) for name in ("published_csr", "typed_csr", "host_csr", "dense")}
    product = matrix @ rhs
    paths = {
        "published_csr": lambda: old_apply(rp, ci, v, rhs, outputs["published_csr"]),
        "typed_csr": lambda: new_apply(matrix, rhs, outputs["typed_csr"]),
        "host_csr": lambda: product.write_to(outputs["host_csr"]),
        "dense": lambda: dense_apply(weights, rhs, outputs["dense"]),
    }
    for operation in paths.values():
        operation()
    qualified = {}
    for name, out in outputs.items():
        data = out.to_numpy()
        np.testing.assert_allclose(data, reference, rtol=2e-5, atol=2e-6)
        qualified[name] = data
    for data in qualified.values():
        np.testing.assert_array_equal(data, qualified["published_csr"])

    def sample(operation):
        qd.sync()
        start = time.perf_counter_ns()
        for _ in range(steps):
            operation()
        qd.sync()
        return (time.perf_counter_ns() - start) / 1e6

    for _ in range(2):
        for operation in paths.values():
            sample(operation)
    timings = {name: [] for name in paths}
    names = list(paths)
    for round_id in range(5):
        order = names[round_id % len(names):] + names[:round_id % len(names)]
        if round_id % 2:
            order.reverse()
        for name in order:
            timings[name].append(sample(paths[name]))
            np.testing.assert_array_equal(outputs[name].to_numpy(), qualified[name])
    medians = {name: statistics.median(samples) for name, samples in timings.items()}
    receipt = {
        "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "quadrants": qd.__version__,
        "source": str(Path(qd.__file__).resolve()),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "baseline_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "revised_sources": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (
            "python/quadrants/linalg/_csr.py", "python/quadrants/lang/ast/ast_transformer.py")},
        "command": "SKBUILD_EDITABLE_SKIP=/Users/bengaskin/dev/quadrants/build/cp310-cp310-macosx_26_0_arm64 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python outputs/csr-spmm/pr-revision/20261010/api-smoke.py",
        "case": {"seed": 7, "N": n, "degree": degree, "B": trials, "K": 1, "steps": steps},
        "protocol": "2 warmup blocks; 5 rotating/reversed paired rounds; resident inputs; synchronous wall time for 256 launches; no fallback; fast_math=False; debug=False; offline_cache=False",
        "environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "load_before_receipt": os.getloadavg(),
        "samples_ms": timings,
        "median_ms": medians,
        "paired_ratio_medians": {
            "typed_over_published": statistics.median(a / b for a, b in zip(timings["typed_csr"], timings["published_csr"])),
            "host_over_published": statistics.median(a / b for a, b in zip(timings["host_csr"], timings["published_csr"])),
            "dense_over_typed": statistics.median(a / b for a, b in zip(timings["dense"], timings["typed_csr"])),
        },
        "checks": "all four outputs match independent float64 reference within atol=2e-6, rtol=2e-5; qualified outputs byte-identical; every timed output equals its qualified output",
        "limits": "Short multiplication-only smoke measurement on interactive desktop; one seed and one K=1 configuration; excludes preparation and scientific workflow; host path reuses a deferred product.",
    }
    (ROOT / "api-smoke.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"median_ms": medians, "ratios": receipt["paired_ratio_medians"], "checks": receipt["checks"]}, indent=2))


if __name__ == "__main__":
    main()
