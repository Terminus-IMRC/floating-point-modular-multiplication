# [floating-point-modular-multiplication](https://github.com/Terminus-IMRC/floating-point-modular-multiplication)

This repository offers the source code for the paper <cite>Improved Modular
Multiplication Algorithms Using Solely IEEE 754 Binary Floating-Point
Operations</cite> by Yukimasa Sugizaki and Daisuke Takahashi (IEEE Transactions
on Emerging Topics on Computing, 2025,
doi:[10.1109/TETC.2025.3582551](https://doi.org/10.1109/TETC.2025.3582551)).
In particular, it contains the source code for the conventional and the proposed
modular multiplication algorithms using integer and floating-point operations.
It also provides the source code for the benchmark that compares the performance
of the algorithms by repeated modular multiplications with fixed modulus and
multiplicand.
Refer to the paper for the details.


## Data availability of our paper

For the benchmark results in the submitted paper, we use the source codes tagged
as
[`ieee-tetc-submission-20240803-1`](https://github.com/Terminus-IMRC/floating-point-modular-multiplication/tree/ieee-tetc-submission-20240803-1).
The raw results are available as JSON files in the
[`data/`](https://github.com/Terminus-IMRC/floating-point-modular-multiplication/tree/ieee-tetc-submission-20240803-1/data)
subdirectory.
The command-line arguments for the program are shown in
`context.command_line_arguments` field in each JSON file.


## Building and running the benchmark

The source code is written in the C++ language and is compiled using the
[CMake](https://cmake.org/) build system.
It uses the [Benchmark](https://github.com/google/benchmark) library for
performance measurement.
On Ubuntu, you can install the required packages with the following commands:

```bash
$ sudo apt update
$ sudo apt install build-essential cmake libbenchmark-dev
```

To build and run the benchmark, clone or download this repository and run the
following commands:

```bash
$ cmake .
$ cmake --build .
$ ./bench_repeated_multiplications
```

By default, the executable is built and tuned for the Sapphire Rapids and
Neoverse V2 microarchitectures on the x86-64 and Arm instruction set
architectures, respectively.
To target other microarchitectures, specify the options below for the CMake
configuration (e.g., `cmake . -D TARGET_UARCH=a64fx -D SVE_VECTOR_BITS=512` for
the Fujitsu A64FX processor):

| CMake option | Description |
| --- | --- |
| `TARGET_UARCH` | Target microarchitecture. Examples: <ul> <li>`skylake-avx512`: Intel Skylake (server)</li> <li>`cascadelake`: Intel Cascade Lake</li> <li>`icelake-server`: Intel Ice Lake (server)</li> <li>`sapphirerapids`: Intel Sapphire Rapids</li> <li>`znver4`: AMD Zen 4</li> <li>`neoverse-v1`: Arm Neoverse V1 (AWS Graviton3/3E)</li> <li>`neoverse-v2`: Arm Neoverse V2 (NVIDIA Grace, AWS Graviton4, Azure Cobalt 100, Google Axion)</li> <li>`a64fx`: Fujitsu A64FX</li> </ul> |
| `SVE_VECTOR_BITS` (Arm only) | Vector width (in bits) for the SVE instruction set. In the examples above, specify `128` for Neoverse V2 and V3, `256` for Neoverse V1, and `512` for A64FX. |

For statically-linked and cross-architecture builds, refer to the examples in
the GitHub Actions workflow file:
[.github/workflows/test.yml](.github/workflows/test.yml).


## License and contribution

For license and copyright notices, see the SPDX file tags in each file.
Unless otherwise noted, files in this project are licensed under the Apache
License, Version 2.0 (SPDX short-form identifier: Apache-2.0) and copyrighted by
the contributors.

Everyone is encouraged to contribute to this project.
See the [CONTRIBUTING.md](CONTRIBUTING.md) file for instructions.
