# Benchmark results

This directory contains the raw benchmark results of the repeated modular
multiplications studied in the submitted paper.

The results are stored as JSON files with filenames in the format of
`repeated_multiplications-<machine>-<compiler>-<date>-<sequence>.json`.
The `compiler` and `date` fields are fixed to `clang++` (the Clang compiler) and
`20240730` (the date of the benchmark), respectively, and the `sequence` field
is set to integers between 1 and 5 to distinguish multiple runs.
The meaning of the `machine` field is as follows:

| Machine  | System                         | Processor (microarchitecture)               | SIMD instruction set                           |
| -------- | ------------------------------ | ------------------------------------------- | ---------------------------------------------- |
| `r7i`    | AWS EC2 r7i.large instance     | Intel Xeon Platinum 8488C (Sapphire Rapids) | AVX-512 (including F, DQ, and IFMA extensions) |
| `r7a`    | AWS EC2 r7a.large instance     | AMD EPYC 9R14 (Zen 4)                       | AVX-512 (including F, DQ, and IFMA extensions) |
| `r8g`    | AWS EC2 r8g.large instance     | AWS Graviton 4 (Neoverse V2)                | SVE2                                           |
| `fx1000` | Fujitsu PRIMEHPC FX1000 system | Fujitsu A64FX                               | SVE                                            |

As described in the paper, the number of vectors processed concurrently is set
to a power of two between 1 and 8192, and the result with the highest
performance is adopted for each algorithm.
The number of iterations of each algorithm is set so that the execution time
exceeds one second for each run.
The execution times are averaged over five runs.
Refer to the paper for the details.
