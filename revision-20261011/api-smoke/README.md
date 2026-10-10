# Revised CSR API smoke measurement

Apple M5, N=512, degree=16, 32 independent matrices, K=1, seed=7, 256 resident multiplies. Two warmup blocks precede five rotating/reversed paired rounds, with fast_math=False, debug=False, no backend fallback and resident inputs.

| Entry point | Median ms |
| --- | ---: |
| Original direct-call CSR implementation from unmerged PR #967 at 40e7289 | 6.906417 |
| Revised typed CSR kernel | 7.376500 |
| Reused host product | 7.486000 |
| Matched source-major Quadrants dense loop | 70.708833 |

Median paired ratios: typed/original 1.060051; host/original 1.088182; dense/typed 9.581484. The receipt's legacy key `published_csr` identifies that original unmerged direct-call implementation, and `typed_over_published`/`host_over_published` use the same baseline.

[receipt.json](receipt.json) preserves the exact command, raw samples, source hashes, version, environment and independent numerical checks. All compared outputs are byte-identical; every timed output equals its qualified output. Float64 reference bounds are atol=2e-6, rtol=2e-5.

This is a one-seed multiplication-only measurement on an interactive desktop. It excludes preparation and the ESN workflow. It predates the later metadata-assertion correction; multiplication arithmetic and binding are unchanged. It establishes the measured execution paths, rather than timings for the final correction or CUDA performance.

## Reproduction and source identity

[api-smoke.py](api-smoke.py) is the original measured harness, with its baseline preserved under [original-source/](original-source/). Its relative path lookup expects that directory beside the script. From a Quadrants checkout with the revised CSR API installed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python /path/to/api-smoke/api-smoke.py
```

The script writes a new `api-smoke.json` beside itself; the frozen original receipt here is named `receipt.json`. For exact original source, [revision-at-measurement.patch](revision-at-measurement.patch) applies on PR base 82467d548. The [measured CSR module](measured-sources/python/quadrants/linalg/_csr.py) and [measured AST transformer](measured-ast-transformer.py) match the hashes in the receipt. The frozen handoff manifest identifies other revision files; its `measured_vs_final_ast_equal` records the intervening import-format/lint-suppression adjustment. The current assertion correction is identified separately in the portability receipt.
