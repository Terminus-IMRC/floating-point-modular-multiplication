// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2024 Yukimasa Sugizaki

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

#include <cxxabi.h>

#include "common.h"
#include "modmul.hpp"
#include "modulus.hpp"
#include "rounding_mode.hpp"

#include <benchmark/benchmark.h>

template <class T> static std::string demangle(void) {
  int status;
  const std::unique_ptr<char, decltype(&std::free)> name{
      abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status),
      &std::free};
  if (status != 0) {
    throw std::runtime_error("abi::__cxa_demangle: " + std::to_string(status));
  }
  return std::string{name.get()};
}

template <class modmul_type_, std::uint64_t num_regs>
class BenchmarkRepeatedMultiplications {

public:
  using modmul_type = modmul_type_;

  using modulus_type = typename modmul_type::modulus_type;
  static inline constexpr std::uint64_t modulus{modulus_type::get_modulus()};

  using vector_register_type = typename modmul_type::vector_register_type;
  static inline constexpr std::uint64_t num_elements{
      vector_register_type::get_num_elements()};

  std::uint64_t a[num_regs][num_elements], b[num_regs][num_elements];
  alignas(64) vector_register_type va[num_regs], vb[num_regs], vbp[num_regs],
      vc[num_regs];

  template <class generator_type>
  BenchmarkRepeatedMultiplications(generator_type &generator) {
    reset(generator);
  }

  static constexpr std::uint64_t get_num_regs(void) { return num_regs; }

  template <class generator_type> void reset(generator_type &generator) {
    std::uniform_int_distribution<std::uint64_t> distribution{0, modulus - 1};

    const typename modmul_type::modulus_inverse_type modulus_inverse{
        modmul_type::get_modulus_inverse()};

    for (std::uint64_t i{0}; i < num_regs; ++i) {
      std::uint64_t t[num_elements];

      for (std::uint64_t j{0}; j < num_elements; ++j) {
        a[i][j] = distribution(generator);
        t[j] = modmul_type::to_operand(a[i][j]);
      }
      va[i].load(t);

      for (std::uint64_t j{0}; j < num_elements; ++j) {
        b[i][j] = distribution(generator);
        t[j] = modmul_type::to_multiplicand(b[i][j]);
      }
      vb[i].load(t);

      for (std::uint64_t j{0}; j < num_elements; ++j) {
        t[j] = modmul_type::precompute(t[j], modulus_inverse);
      }
      vbp[i].load(t);
    }
  }

  void benchmark(std::uint64_t num_loops) {
    for (std::uint64_t i{0}; i < num_regs; ++i) {
      vc[i] = va[i];
    }

    for (; num_loops; --num_loops) {
      for (std::uint64_t i{0}; i < num_regs; ++i) {
        vc[i] = modmul_type::multiply(vc[i], vb[i], vbp[i]);
      }
    }
  }

  void verify(const std::uint64_t num_loops) const {
    for (std::uint64_t i{0}; i < num_regs; ++i) {
      std::uint64_t c[num_elements];
      vc[i].store(c);
      for (std::uint64_t j{0}; j < num_elements; ++j) {
        const std::uint64_t correct{modulus_type::multiply(
            a[i][j], modulus_type::power(b[i][j], num_loops))};
        c[j] = modmul_type::from_operand(c[j]);
        if (c[j] != correct) {
          throw std::runtime_error{
              (std::ostringstream{}
               << "Mismatch at i = " << i << ", j = " << j << ": " << c[j]
               << " != " << correct << " = " << a[i][j] << " * pow(" << b[i][j]
               << ", " << num_loops
               << ", N) % N where N = " << modulus_type::get_modulus())
                  .str()};
        }
      }
    }
  }
};

template <class benchmark_type, class generator_type>
static void run_benchmark(benchmark::State &state, generator_type &generator) {
  const std::uint64_t num_loops(state.range(0));
  benchmark_type benchmark{generator};

  for (auto _ : state) {
    benchmark.benchmark(num_loops);
    ::benchmark::DoNotOptimize(benchmark);
  }

  state.SetItemsProcessed(state.iterations() * num_loops *
                          benchmark_type::get_num_regs());
  state.counters["num_loops"] = num_loops;
  state.counters["num_regs"] = benchmark_type::get_num_regs();

  benchmark.verify(num_loops);
}

template <class benchmark_type, class generator_type>
static auto register_benchmark(
    generator_type &generator,
    const std::float_round_style style = std::round_indeterminate) {
  const std::string name{demangle<benchmark_type>()};
  if (style == std::round_indeterminate) {
    return benchmark::RegisterBenchmark(
        name, run_benchmark<benchmark_type, generator_type>,
        std::ref(generator));
  } else {
    return benchmark::RegisterBenchmark(
        name,
        [style](benchmark::State &state, generator_type &generator) {
          const RoundingModePush push{style};
          run_benchmark<benchmark_type>(state, generator);
        },
        std::ref(generator));
  }
}

int main(int argc, char *argv[]) {
  const std::string command_line_arguments{
      argc <= 0 ? ""
                : std::accumulate(&argv[1], &argv[argc], std::string{argv[0]},
                                  [](const std::string &a, const char b[]) {
                                    return a + ' ' + b;
                                  })};

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }

  benchmark::AddCustomContext("compiler_version", __VERSION__);
  benchmark::AddCustomContext("command_line_arguments", command_line_arguments);

  using generator_type = std::default_random_engine;
  generator_type generator{42};

  using num_regs_list =
      std::integer_sequence<std::uint64_t, 1, 2, 4, 8, 16, 32, 64, 128, 256,
                            512, 1024, 2048, 4096>;

  for_each_sequence<num_regs_list>([&]<std::uint64_t num_regs> {
    constexpr long num_loops{(std::uint64_t{1} << 20) / num_regs};

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedInt64ByInt32Shoup<Modulus<UINT64_C(0x7fff'ffff'ffff'ffff)>>,
        num_regs>>(generator)
        ->Arg(num_loops);

    register_benchmark<
        BenchmarkRepeatedMultiplications<UnsignedInt64ByInt32Montgomery<Modulus<
                                             UINT64_C(0x7fff'ffff'ffff'ffff)>>,
                                         num_regs>>(generator)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedInt52ByBinary64Shoup<Modulus<UINT64_C(0x0007'ffff'ffff'ffff)>>,
        num_regs>>(generator, std::round_toward_zero)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedInt52ByBinary64Montgomery<
            Modulus<UINT64_C(0x000f'ffff'ffff'ffff)>>,
        num_regs>>(generator, std::round_toward_zero)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedBinary64ShoupRadixWise<
            Modulus<UINT64_C(0x0000'ffff'ffff'ffff)>>,
        num_regs>>(generator, std::round_toward_zero)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedBinary64MontgomeryRadixWise<
            Modulus<UINT64_C(0x0000'ffff'ffff'ffff)>>,
        num_regs>>(generator, std::round_toward_zero)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedBinary64ShoupExplicitR2I<
            Modulus<UINT64_C(0x0010'0000'0000'0000)>>,
        num_regs>>(generator, std::round_toward_zero)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedBinary64ShoupImplicitR2I<
            Modulus<UINT64_C(0x0010'0000'0000'0000)>>,
        num_regs>>(generator, std::round_toward_zero)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedBinary64MontgomeryExplicitR2I<
            Modulus<UINT64_C(0x001f'ffff'ffff'ffff)>>,
        num_regs>>(generator, std::round_toward_zero)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedBinary64MontgomeryImplicitR2I<
            Modulus<UINT64_C(0x0010'0000'0000'0001)>>,
        num_regs>>(generator, std::round_toward_zero)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        SignedBinary64ShoupExplicitR2I<
            Modulus<UINT64_C(0x0013'60ad'1185'67cd)>>,
        num_regs>>(generator, std::round_to_nearest)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        SignedBinary64ShoupImplicitR2I<
            Modulus<UINT64_C(0x0013'60ad'1185'67cd)>>,
        num_regs>>(generator, std::round_to_nearest)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        SignedBinary64MontgomeryExplicitR2I<
            Modulus<UINT64_C(0x001f'ffff'ffff'ffff)>>,
        num_regs>>(generator, std::round_toward_neg_infinity)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        SignedBinary64MontgomeryImplicitR2I<
            Modulus<UINT64_C(0x000f'ffff'ffff'ffff)>>,
        num_regs>>(generator, std::round_to_nearest)
        ->Arg(num_loops);

#if defined(__AVX512IFMA__)

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedInt52Shoup<Modulus<UINT64_C(0x0007'ffff'ffff'ffff)>>,
        num_regs>>(generator)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedInt52Montgomery<Modulus<UINT64_C(0x0007'ffff'ffff'ffff)>>,
        num_regs>>(generator)
        ->Arg(num_loops);

#endif /* defined(__AVX512IFMA__) */

#if defined(__ARM_FEATURE_SVE)

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedInt64Shoup<Modulus<UINT64_C(0x7fff'ffff'ffff'ffff)>>,
        num_regs>>(generator)
        ->Arg(num_loops);

    register_benchmark<BenchmarkRepeatedMultiplications<
        UnsignedInt64Montgomery<Modulus<UINT64_C(0x7fff'ffff'ffff'ffff)>>,
        num_regs>>(generator)
        ->Arg(num_loops);

#endif /* defined(__ARM_FEATURE_SVE) */
  });

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
