// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2024 Yukimasa Sugizaki

#ifndef FPMODMUL_INTRINSICS_HPP_INCLUDED
#define FPMODMUL_INTRINSICS_HPP_INCLUDED

#include <cstdint>

#if defined(__AVX512F__)

#include <immintrin.h>

static inline __m512i _mm512_mulhi_epu64(const __m512i a, const __m512i b) {
  const __m512i a1{_mm512_srli_epi64(a, 32)};
  const __m512i b1{_mm512_srli_epi64(b, 32)};
  const __m512i a0b0{_mm512_mul_epu32(a, b)};
  const __m512i a0b1{_mm512_mul_epu32(a, b1)};
  const __m512i a1b0{_mm512_mul_epu32(a1, b)};
  const __m512i a1b1{_mm512_mul_epu32(a1, b1)};
  const __m512i t0{_mm512_add_epi64(_mm512_srli_epi64(a0b0, 32), a0b1)};
  const __m512i t1{_mm512_add_epi64(
      _mm512_maskz_mov_epi32(_cvtu32_mask16(0x5555), t0), a1b0)};
  return _mm512_add_epi64(_mm512_add_epi64(_mm512_srli_epi64(t0, 32), a1b1),
                          _mm512_srli_epi64(t1, 32));
}

static inline __m512i _mm512_mul64hi_epu63(const __m512i a, const __m512i b) {
  const __m512i a1{_mm512_srli_epi64(a, 32)};
  const __m512i b1{_mm512_srli_epi64(b, 32)};
  const __m512i a0b0{_mm512_mul_epu32(a, b)};
  const __m512i a0b1{_mm512_mul_epu32(a, b1)};
  const __m512i a1b0{_mm512_mul_epu32(a1, b)};
  const __m512i a1b1{_mm512_mul_epu32(a1, b1)};
  const __m512i t{_mm512_add_epi64(_mm512_srli_epi64(a0b0, 32),
                                   _mm512_add_epi64(a0b1, a1b0))};
  return _mm512_add_epi64(_mm512_srli_epi64(t, 32), a1b1);
}

#elif defined(__ARM_FEATURE_SVE)

#if !defined(__ARM_FEATURE_SVE_BITS) || __ARM_FEATURE_SVE_BITS == 0
#error we require SVE vector length to be fixed at compile time
#endif

#include <arm_sve.h>

/*
 * For SVE (version 1), we prefer revw + sel to trn[12] since it achieves
 * better throughput on A64FX.
 *
 * | Instruction | Throughput (instruction/cycle) | Latency (cycle) |
 * | ----------- |------------------------------- | --------------- |
 * | revw        |                              2 |               4 |
 * | sel         |                              2 |               4 |
 * | trn[12]     |                              1 |               6 |
 */

[[maybe_unused]]
static svuint32_t svtrn1_u32_implicit(const svuint32_t a, const svuint32_t b) {
#if defined(__ARM_FEATURE_SVE2)
  return svtrn1_u32(a, b);
#else
  return svsel(
      svptrue_b64(), a,
      svreinterpret_u32(svrevw_x(svptrue_b64(), svreinterpret_u64(b))));
#endif
}

[[maybe_unused]]
static svuint32_t svtrn2_u32_implicit(const svuint32_t a, const svuint32_t b) {
#if defined(__ARM_FEATURE_SVE2)
  return svtrn2_u32(a, b);
#else
  return svsel(svptrue_b64(),
               svreinterpret_u32(svrevw_x(svptrue_b64(), svreinterpret_u64(a))),
               b);
#endif
}

static svuint64_t svmul_u64_by_u32(const svuint64_t a64, const svuint64_t b64) {
  const svuint32_t a32{svreinterpret_u32(a64)};
  const svuint32_t b32{svreinterpret_u32(b64)};
  const svuint32_t b32swap{svreinterpret_u32(svrevw_x(svptrue_b64(), b64))};
#if defined(__ARM_FEATURE_SVE2)
  const svuint32_t zero{svdup_u32(0)};
  svuint32_t t;
  t = svmul_x(svptrue_b32(), a32, b32swap);
  t = svaddp_x(svptrue_b32(), zero, t);
  return svmlalb(svreinterpret_u64(t), a32, b32);
#else
  const svbool_t even32{svptrue_b64()};
  const svbool_t odd32{svnot_z(svptrue_b32(), even32)};
  const svuint32_t a0b0h{svmulh_z(even32, a32, b32)};
  svuint32_t t;
  t = svmla_x(svptrue_b32(), a0b0h, a32, b32swap);
  t = svadd_z(odd32, t,
              svreinterpret_u32(svrevw_x(svptrue_b64(), svreinterpret_u64(t))));
  t = svmla_m(even32, t, a32, b32);
  return svreinterpret_u64(t);
#endif
}

static svuint64_t svmulh_u63_by_u32(const svuint64_t a64,
                                    const svuint64_t b64) {
  const svbool_t even32{svptrue_b64()};
  const svuint32_t a32{svreinterpret_u32(a64)};
  const svuint32_t b32{svreinterpret_u32(b64)};
  const svuint32_t b32swap{svreinterpret_u32(svrevw_x(svptrue_b64(), b64))};
#if defined(__ARM_FEATURE_SVE2)
  const svuint32_t a0b0h{svmulh_z(even32, a32, b32)};
  svuint64_t w;
  w = svmlalb(svreinterpret_u64(a0b0h), a32, b32swap);
  w = svmlalt(w, a32, b32swap);
  w = svlsr_x(svptrue_b64(), w, 32);
  w = svmlalt(w, a32, b32);
  return w;
#else
  const svbool_t odd32{svnot_z(svptrue_b32(), even32)};
  const svuint32_t zero{svdup_u32(0)};
  const svuint32_t a0b0h_a1b1h{svmulh_x(svptrue_b32(), a32, b32)};
  const svuint32_t a0b1l_a1b0l{svmul_x(svptrue_b32(), a32, b32swap)};
  const svuint32_t a0b1h_a1b0h{svmulh_x(svptrue_b32(), a32, b32swap)};
  const svuint64_t a0b0h{svreinterpret_u64(svsel(even32, a0b0h_a1b1h, zero))};
  const svuint64_t a0b1{
      svreinterpret_u64(svtrn1_u32_implicit(a0b1l_a1b0l, a0b1h_a1b0h))};
  const svuint64_t a1b0{
      svreinterpret_u64(svtrn2_u32_implicit(a0b1l_a1b0l, a0b1h_a1b0h))};
  const svuint64_t a1b1{svreinterpret_u64(
      svtrn2_u32_implicit(svmul_x(odd32, a32, b32), a0b0h_a1b1h))};
  svuint64_t w;
  w = svadd_x(svptrue_b64(), a0b0h, a0b1);
  w = svadd_x(svptrue_b64(), w, a1b0);
  w = svlsr_x(svptrue_b64(), w, 32);
  w = svadd_x(svptrue_b64(), w, a1b1);
  return w;
#endif
}

static svuint64_t svmulh_u63_u64_by_u32(const svuint64_t a64,
                                        const svuint64_t b64) {
  const svuint32_t a32{svreinterpret_u32(a64)};
  const svuint32_t b32{svreinterpret_u32(b64)};
  const svuint32_t b32swap{svreinterpret_u32(svrevw_x(svptrue_b64(), b64))};
#if defined(__ARM_FEATURE_SVE2)
  const svuint32_t a0b0h_a1b1h{svmulh_x(svptrue_b32(), a32, b32)};
  const svuint64_t a0b1{svmullb(a32, b32swap)};
  svuint64_t w;
  w = svreinterpret_u64(svaddlb(svreinterpret_u32(a0b1), a0b0h_a1b1h));
  w = svmlalt(w, a32, b32swap);
  w = svreinterpret_u64(svaddlt(svreinterpret_u32(w), svreinterpret_u32(a0b1)));
  w = svmlalt(w, a32, b32);
  return w;
#else
  const svbool_t even32{svptrue_b64()};
  const svbool_t odd32{svnot_z(svptrue_b32(), even32)};
  const svuint32_t zero{svdup_u32(0)};
  const svuint32_t a0b0h_a1b1h{svmulh_x(svptrue_b32(), a32, b32)};
  const svuint32_t a0b1l_a1b0l{svmul_x(svptrue_b32(), a32, b32swap)};
  const svuint32_t a0b1h_a1b0h{svmulh_x(svptrue_b32(), a32, b32swap)};
  const svuint64_t a0b0h{svreinterpret_u64(svsel(even32, a0b0h_a1b1h, zero))};
  const svuint64_t a0b1l{svreinterpret_u64(svsel(even32, a0b1l_a1b0l, zero))};
  const svuint64_t a0b1h{svreinterpret_u64(svsel(even32, a0b1h_a1b0h, zero))};
  const svuint64_t a1b0{
      svreinterpret_u64(svtrn2_u32_implicit(a0b1l_a1b0l, a0b1h_a1b0h))};
  const svuint64_t a1b1{svreinterpret_u64(
      svtrn2_u32_implicit(svmul_x(odd32, a32, b32), a0b0h_a1b1h))};
  svuint64_t w;
  w = svadd_x(svptrue_b64(), a0b0h, a1b0);
  w = svadd_x(svptrue_b64(), w, a0b1l);
  w = svlsr_x(svptrue_b64(), w, 32);
  w = svadd_x(svptrue_b64(), w, svadd_x(svptrue_b64(), a0b1h, a1b1));
  return w;
#endif
}

#else

#error unsupported vector type

#endif

#endif /* FPMODMUL_INTRINSICS_HPP_INCLUDED */
