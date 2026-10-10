# Buffer-backed CSR portability evidence

`SparseMatrix.from_csr(...)` and `(A @ X).write_to(Y)` use the same interface on CPU, CUDA and Metal.

Final checks: local CPU 19 passes; Metal 15 passes and four backend-specific skips; real T4 CUDA 16 passes and three CPU-only skips; native Linux x86 CPU CSR/API 38 passes and five optional-dependency collection skips. Pre-commit and scoped library typing pass; Sphinx HTML builds with 62 warnings.

Reproduce with the revised checkout installed:

```sh
python tests/run_tests.py csr_spmm --arch cuda -t 1 -v
python tests/run_tests.py --arch cpu -t 1 -k 'csr_spmm or test_api' -m 'not needs_torch'
```

The receipt identifies the CI wheel, unchanged native core, current revised sources, hardware and exact test commands. CUDA fallback was disabled. The wheel's public Python sources were overlaid with this revision. A rebuilt PR package remains the package-integration check.

The CUDA tests caught an assertion ordering bug in the CSR helper. A debug-only assertion now runs after the guarded loop publishes and consumes its safe extent. Tests cover invalid/valid call reuse and debug output preservation. Numerical tolerances and multiplication arithmetic are unchanged.

The successful CUDA phase and successful Linux CPU phase came from separate invocations: the first invocation's later CPU collection needed a missing public tool checker, which was added for the CPU-only rerun. The detailed receipt preserves that distinction.

This supplement contains the current revision patch and raw remote test logs/JUnit. It records correctness and API portability; the earlier M5 performance measurements retain their original source identity.

## Kernel error contract

Host calls raise on incompatible metadata. Matching kernel shapes are caller preconditions: incompatible runtime shapes silently leave the previous output in place in release mode. Debug reports an assertion where supported; Metal and Linux ARM64 lack runtime assertions. This safety fallback does not produce a valid multiplication result. The documentation now states this behavior explicitly.

`revision-at-portability-test.patch` preserves the snapshot used by the remote checks; `revision-from-base.patch` includes the later documentation clarification. Executable Python sources and tests are unchanged.
