// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2024 Yukimasa Sugizaki

#ifndef FPMODMUL_VECTOR_REGISTER_HPP_INCLUDED
#define FPMODMUL_VECTOR_REGISTER_HPP_INCLUDED

#include <cstdint>

#include "intrinsics.hpp"

class VectorRegister {

#if defined(__AVX512F__)
  using value_type = __m512i;
  using value_internal_type = value_type;
#elif defined(__ARM_FEATURE_SVE)
  /*
   * Without this distinction, Clang of some versions will fail with internal
   * errors.
   */
  using value_type = svuint64_t;
  using value_internal_type =
      svuint64_t __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
#endif

  value_internal_type value;

public:
  VectorRegister(void) = default;

  VectorRegister(const value_type value) : value(value) {}

  operator value_type() const { return value; }

  static constexpr std::uint64_t get_num_elements(void) {
    return sizeof(value) / 8;
  }

#if defined(__AVX512F__)

  void broadcast(const std::uint64_t a) { value = _mm512_set1_epi64(a); }

  void load(const std::uint64_t *const p) { value = _mm512_loadu_epi64(p); }

  void store(std::uint64_t *const p) const { _mm512_storeu_epi64(p, value); }

#elif defined(__ARM_FEATURE_SVE)

  void broadcast(const std::uint64_t a) { value = svdup_u64(a); }

  void load(const std::uint64_t *const p) { value = svld1(svptrue_b64(), p); }

  void store(std::uint64_t *const p) const { svst1(svptrue_b64(), p, value); }

#endif
};

#endif /* FPMODMUL_VECTOR_REGISTER_HPP_INCLUDED */
