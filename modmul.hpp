// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2024 Yukimasa Sugizaki

#ifndef FPMODMUL_MODMUL_HPP_INCLUDED
#define FPMODMUL_MODMUL_HPP_INCLUDED

#include <bit>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <string>

#include "intrinsics.hpp"
#include "vector_register.hpp"

/*
 * Shoup multiplication for positive 64-bit integers by positive 32-bit integer
 * operations.
 */
template <class modulus_type_> struct UnsignedInt64ByInt32Shoup {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    std::uint64_t Ninv_lo, Ninv_hi;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    /*
     * Precomputation without 256-bit integer operations based on
     * https://github.com/libntl/ntl/blob/main/include/NTL/sp_arith.h#L1142
     */
    static_assert(N >= 1);
    const unsigned __int128 Ninv{std::has_single_bit(N)
                                     ? (unsigned __int128){1}
                                           << (128 - std::bit_width(N))
                                     : ~(unsigned __int128){0} / N};
    return {
        .Ninv_lo{static_cast<std::uint64_t>(Ninv)},
        .Ninv_hi{static_cast<std::uint64_t>(Ninv >> 64)},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) { return a; }

  static std::uint64_t from_operand(const std::uint64_t a) {
    assert(a <= modulus_type::get_modulus() - 1);
    return a;
  }

  static std::uint64_t to_multiplicand(const std::uint64_t b) { return b; }

  static std::uint64_t precompute(const std::uint64_t b,
                                  const modulus_inverse_type modulus_inverse) {
    constexpr std::uint64_t modulus{modulus_type::get_modulus()};
    std::uint64_t bp;
    bp = (unsigned __int128){modulus_inverse.Ninv_lo} * b >> 64;
    bp += modulus_inverse.Ninv_hi * b;
    /*
     * Correction of precomputed value based on
     * https://github.com/Mysticial/ProtoNTT/blob/master/source/Internals/ModularIntrinsics.h#L104-L109
     */
    if (-bp * modulus >= modulus) {
      ++bp;
    }
    return bp;
  }

  static vector_register_type multiply(const vector_register_type a,
                                       const vector_register_type b,
                                       const vector_register_type bp) {
#if defined(__AVX512F__)
    const __m512i N{_mm512_set1_epi64(modulus_type::get_modulus())};
    const __m512i ab{_mm512_mullo_epi64(a, b)};
    const __m512i q{_mm512_mulhi_epu64(a, bp)};
    const __m512i qN{_mm512_mullo_epi64(q, N)};
    __m512i c;
    c = _mm512_sub_epi64(ab, qN);
    /*
     * Correction without comparison operation [HLQ16], [Takahashi18].
     */
    c = _mm512_min_epu64(c, _mm512_sub_epi64(c, N));
    return c;
#elif defined(__ARM_FEATURE_SVE)
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svuint64_t q{svmulh_u63_u64_by_u32(a, bp)};
    const svuint64_t ab{svmul_u64_by_u32(a, b)};
    svuint64_t c;
    c = svmul_u64_by_u32(q, svdup_u64(N));
    c = svsub_x(svptrue_b64(), ab, c);
    c = svmin_x(svptrue_b64(), c, svsub_x(svptrue_b64(), c, N));
    return c;
#endif
  }
};

/*
 * Montgomery multiplication for positive 64-bit integers by positive 32-bit
 * integer operations.
 */
template <class modulus_type_> struct UnsignedInt64ByInt32Montgomery {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    std::uint64_t Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    static_assert(N % 2 == 1);
    /*
     * Newton-Raphson iterations for modulus inverse [BMSZ14].
     */
    std::uint64_t Ninv{(N * 3) ^ 2}; /* 5 */
    Ninv *= 2 - N * Ninv;            /* 10 */
    Ninv *= 2 - N * Ninv;            /* 20 */
    Ninv *= 2 - N * Ninv;            /* 40 */
    Ninv *= 2 - N * Ninv;            /* 64 */
    return {
        .Ninv{Ninv},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) { return a; }

  static std::uint64_t from_operand(const std::uint64_t a) {
    assert(a <= modulus_type::get_modulus() - 1);
    return a;
  }

  static std::uint64_t to_multiplicand(const std::uint64_t b) {
    return modulus_type::multiply(b, -modulus_type::get_modulus());
  }

  static std::uint64_t precompute(const std::uint64_t b,
                                  const modulus_inverse_type modulus_inverse) {
    return b * modulus_inverse.Ninv;
  }

  static vector_register_type multiply(const vector_register_type a,
                                       const vector_register_type b,
                                       const vector_register_type bp) {
#if defined(__AVX512F__)
    const __m512i N{_mm512_set1_epi64(modulus_type::get_modulus())};
    const __m512i ab_hi{_mm512_mulhi_epu64(a, b)};
    const __m512i q{_mm512_mullo_epi64(a, bp)};
    const __m512i qN_hi{_mm512_mulhi_epu64(q, N)};
    __m512i c;
    c = _mm512_sub_epi64(ab_hi, qN_hi);
    c = _mm512_min_epu64(c, _mm512_add_epi64(c, N));
    return c;
#elif defined(__ARM_FEATURE_SVE)
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svuint64_t q{svmul_u64_by_u32(a, bp)};
    const svuint64_t ab1{svmulh_u63_by_u32(a, b)};
    const svuint64_t qN1{svmulh_u63_u64_by_u32(svdup_u64(N), q)};
    svuint64_t c;
    c = svsub_x(svptrue_b64(), ab1, qN1);
    c = svmin_x(svptrue_b64(), c, svadd_x(svptrue_b64(), c, N));
    return c;
#endif
  }
};

template <class modulus_type_> struct UnsignedInt52ByBinary64Shoup {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    std::uint64_t Ninv_lo, Ninv_hi;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t mask52{(std::uint64_t{1} << 52) - 1};
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const unsigned __int128 Ninv{((unsigned __int128){1} << 104) / N};
    return {
        .Ninv_lo{static_cast<std::uint64_t>(Ninv) & mask52},
        .Ninv_hi{static_cast<std::uint64_t>(Ninv >> 52)},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) { return a; }

  static std::uint64_t from_operand(const std::uint64_t a) {
    assert(a <= modulus_type::get_modulus() - 1);
    return a;
  }

  static std::uint64_t to_multiplicand(const std::uint64_t bu) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(bu));
  }

  static std::uint64_t precompute(std::uint64_t b,
                                  const modulus_inverse_type modulus_inverse) {
    constexpr std::uint64_t mask52{(std::uint64_t{1} << 52) - 1};
    constexpr std::uint64_t modulus{modulus_type::get_modulus()};
    b = static_cast<std::uint64_t>(std::bit_cast<double>(b));
    std::uint64_t bp;
    bp = (unsigned __int128){modulus_inverse.Ninv_lo} * b >> 52;
    bp += modulus_inverse.Ninv_hi * b;
    if (((-bp * modulus) & mask52) >= modulus) {
      ++bp;
    }
    return std::bit_cast<std::uint64_t>(static_cast<double>(bp));
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512i mask52{_mm512_set1_epi64((std::uint64_t{1} << 52) - 1)};
    const __m512d two104{_mm512_set1_pd(0x1p104)},
        two104_two52{_mm512_set1_pd(0x1p104 + 0x1p52)};
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d a{_mm512_cvtepu64_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d q{_mm512_cvtepu64_pd(_mm512_and_epi64(
        _mm512_castpd_si512(_mm512_fmadd_pd(a, bp, two104)), mask52))};
    const __m512d ab0{_mm512_fmadd_pd(
        a, b, _mm512_sub_pd(two104_two52, _mm512_fmadd_pd(a, b, two104)))};
    const __m512d qN0{_mm512_fmadd_pd(
        q, N, _mm512_sub_pd(two104_two52, _mm512_fmadd_pd(q, N, two104)))};
    __m512i c{
        _mm512_sub_epi64(_mm512_castpd_si512(ab0), _mm512_castpd_si512(qN0))};
    c = _mm512_and_epi64(c, mask52);
    c = _mm512_min_epu64(
        c, _mm512_sub_epi64(c, _mm512_set1_epi64(modulus_type::get_modulus())));
    return c;
#elif defined(__ARM_FEATURE_SVE)
    constexpr std::uint64_t mask52{(std::uint64_t{1} << 52) - 1};
    constexpr double two104{0x1p104}, two104_two52{0x1p104 + 0x1p52};
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svcvt_f64_x(svptrue_b64(), ai)},
        b{svreinterpret_f64(bi)}, bp{svreinterpret_f64(bpi)};
    const svfloat64_t q{svcvt_f64_x(
        svptrue_b64(),
        svand_x(svptrue_b64(),
                svreinterpret_u64(svmad_x(svptrue_b64(), a, bp, two104)),
                mask52))};
    const svfloat64_t ab0{
        svmla_x(svptrue_b64(),
                svsubr_x(svptrue_b64(), svmad_x(svptrue_b64(), a, b, two104),
                         two104_two52),
                a, b)};
    const svfloat64_t qN0{svmla_x(
        svptrue_b64(),
        svsubr_x(svptrue_b64(), svmad_x(svptrue_b64(), q, svdup_f64(N), two104),
                 two104_two52),
        q, N)};
    svuint64_t c;
    c = svsub_x(svptrue_b64(), svreinterpret_u64(ab0), svreinterpret_u64(qN0));
    c = svand_x(svptrue_b64(), c, mask52);
    c = svmin_x(svptrue_b64(), c,
                svsub_x(svptrue_b64(), c, modulus_type::get_modulus()));
    return c;
#endif
  }
};

template <class modulus_type_> struct UnsignedInt52ByBinary64Montgomery {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    double Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t mask52{(std::uint64_t{1} << 52) - 1};
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    std::uint64_t Ninv{(N * 3) ^ 2}; /* 5 */
    Ninv *= 2 - N * Ninv;            /* 10 */
    Ninv *= 2 - N * Ninv;            /* 20 */
    Ninv *= 2 - N * Ninv;            /* 40 */
    Ninv *= 2 - N * Ninv;            /* 64 */
    Ninv &= mask52;
    return {
        .Ninv{
            static_cast<double>(Ninv),
        },
    };
  }

  static std::uint64_t to_operand(const std::uint64_t ai) { return ai; }

  static std::uint64_t from_operand(const std::uint64_t a) {
    assert(a <= modulus_type::get_modulus() - 1);
    return a;
  }

  static std::uint64_t to_multiplicand(std::uint64_t b) {
    b = modulus_type::multiply(b, std::uint64_t{1} << 52);
    return std::bit_cast<std::uint64_t>(static_cast<double>(b));
  }

  static std::uint64_t precompute(const std::uint64_t bu,
                                  const modulus_inverse_type modulus_inverse) {
#if defined(__AVX512F__)
    const __m128d mask52{
        _mm_set_sd(std::bit_cast<double>((std::uint64_t{1} << 52) - 1))};
    const __m128d two104{_mm_set_sd(0x1p104)},
        two104_two52{_mm_set_sd(0x1p104 + 0x1p52)};
    const __m128d Ninv{_mm_set_sd(modulus_inverse.Ninv)};
    const __m128d b{_mm_set_sd(std::bit_cast<double>(bu))};
    const __m128d bp{_mm_cvtepu64_pd(_mm_castpd_si128(_mm_and_pd(
        _mm_fmadd_sd(b, Ninv,
                     _mm_sub_sd(two104_two52, _mm_fmadd_sd(b, Ninv, two104))),
        mask52)))};
    return std::bit_cast<std::uint64_t>(_mm_cvtsd_f64(bp));
#elif defined(__ARM_FEATURE_SVE)
    constexpr std::uint64_t mask52{(std::uint64_t{1} << 52) - 1};
    constexpr double two104{0x1p104}, two104_two52{0x1p104 + 0x1p52};
    const svbool_t first{svptrue_pat_b64(SV_VL1)};
    const svfloat64_t Ninv{svdup_f64_x(first, modulus_inverse.Ninv)};
    const svfloat64_t b{svdup_f64_x(first, std::bit_cast<double>(bu))};
    const svfloat64_t bp{svcvt_f64_x(
        first, svand_x(first,
                       svreinterpret_u64(svmad_x(
                           first, b, Ninv,
                           svsubr_x(first, svmad_x(first, b, Ninv, two104),
                                    two104_two52))),
                       mask52))};
    return std::bit_cast<std::uint64_t>(svlastb(first, bp));
#endif
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d mask52{
        _mm512_set1_pd(std::bit_cast<double>((std::uint64_t{1} << 52) - 1))};
    const __m512d two104{_mm512_set1_pd(0x1p104)},
        two104_two52{_mm512_set1_pd(0x1p104 + 0x1p52)};
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d a{_mm512_cvtepu64_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d q{_mm512_cvtepu64_pd(_mm512_castpd_si512(_mm512_and_pd(
        _mm512_fmadd_pd(
            a, bp, _mm512_sub_pd(two104_two52, _mm512_fmadd_pd(a, bp, two104))),
        mask52)))};
    const __m512d ab1{_mm512_fmadd_pd(a, b, two104)};
    const __m512d qN1{_mm512_fmadd_pd(q, N, two104)};
    __m512i c{
        _mm512_sub_epi64(_mm512_castpd_si512(ab1), _mm512_castpd_si512(qN1))};
    c = _mm512_min_epu64(
        c, _mm512_add_epi64(c, _mm512_set1_epi64(modulus_type::get_modulus())));
    return c;
#elif defined(__ARM_FEATURE_SVE)
    constexpr std::uint64_t mask52{(std::uint64_t{1} << 52) - 1};
    constexpr double two104{0x1p104}, two104_two52{0x1p104 + 0x1p52};
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svcvt_f64_x(svptrue_b64(), ai)},
        b{svreinterpret_f64(bi)}, bp{svreinterpret_f64(bpi)};
    const svfloat64_t q{svcvt_f64_x(
        svptrue_b64(),
        svand_x(
            svptrue_b64(),
            svreinterpret_u64(svmad_x(
                svptrue_b64(), a, bp,
                svsubr_x(svptrue_b64(), svmad_x(svptrue_b64(), a, bp, two104),
                         two104_two52))),
            mask52))};
    const svfloat64_t ab1{svmad_x(svptrue_b64(), a, b, two104)};
    const svfloat64_t qN1{svmad_x(svptrue_b64(), q, svdup_f64(N), two104)};
    svuint64_t c;
    c = svsub_x(svptrue_b64(), svreinterpret_u64(ab1), svreinterpret_u64(qN1));
    c = svmin_x(svptrue_b64(), c,
                svadd_x(svptrue_b64(), c, modulus_type::get_modulus()));
    return c;
#endif
  }
};

template <class modulus_type_> struct UnsignedBinary64ShoupRadixWise {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    std::uint64_t Ninv_lo, Ninv_hi;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const unsigned __int128 Ninv{((unsigned __int128){1} << 104) / N};
    return {
        .Ninv_lo{static_cast<std::uint64_t>(Ninv) &
                 ((std::uint64_t{1} << 52) - 1)},
        .Ninv_hi{static_cast<std::uint64_t>(Ninv >> 52)},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(a));
  }

  static std::uint64_t from_operand(const std::uint64_t a) {
    const double ad{std::bit_cast<double>(a)};
    assert(std::remainder(ad, 1.) == 0);
    assert(0 <= ad && ad <= modulus_type::get_modulus() - 1);
    return static_cast<std::uint64_t>(ad);
  }

  static std::uint64_t to_multiplicand(const std::uint64_t b) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(b));
  }

  static std::uint64_t
  precompute(const std::uint64_t b,
             [[maybe_unused]] const modulus_inverse_type modulus_inverse) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(
        (static_cast<unsigned __int128>(std::bit_cast<double>(b)) << 52) /
        modulus_type::get_modulus()));
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d two104{_mm512_set1_pd(0x1p104)},
        two104_two52{_mm512_set1_pd(0x1p104 + 0x1p52)},
        twom52{_mm512_set1_pd(0x1p-52)};
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d a{_mm512_castsi512_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d q{_mm512_mul_pd(
        _mm512_sub_pd(_mm512_fmadd_pd(a, bp, two104), two104), twom52)};
    const __m512d ab1{_mm512_fmadd_pd(a, b, two104)};
    const __m512d qN1{_mm512_fmadd_pd(q, N, two104)};
    const __m512d ab0{_mm512_fmadd_pd(a, b, _mm512_sub_pd(two104_two52, ab1))};
    const __m512d qN0{_mm512_fmadd_pd(q, N, _mm512_sub_pd(two104_two52, qN1))};
    const __m512d c1{_mm512_sub_pd(ab1, qN1)};
    const __m512d c0{_mm512_sub_pd(ab0, qN0)};
    __m512d c{_mm512_add_pd(c1, c0)};
    c = _mm512_mask_sub_pd(c, _mm512_cmpnlt_pd_mask(c, N), c, N);
    return _mm512_castpd_si512(c);
#elif defined(__ARM_FEATURE_SVE)
    constexpr double two104{0x1p104}, two104_two52{0x1p104 + 0x1p52},
        twom52{0x1p-52};
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svreinterpret_f64(ai)}, b{svreinterpret_f64(bi)},
        bp{svreinterpret_f64(bpi)};
    const svfloat64_t q{svmul_x(
        svptrue_b64(),
        svsub_x(svptrue_b64(), svmad_x(svptrue_b64(), a, bp, two104), two104),
        twom52)};
    const svfloat64_t ab1{svmad_x(svptrue_b64(), a, b, two104)};
    const svfloat64_t qN1{svmad_x(svptrue_b64(), q, svdup_f64(N), two104)};
    const svfloat64_t ab0{svmla_x(
        svptrue_b64(), svsubr_x(svptrue_b64(), ab1, two104_two52), a, b)};
    const svfloat64_t qN0{svmla_x(
        svptrue_b64(), svsubr_x(svptrue_b64(), qN1, two104_two52), q, N)};
    const svfloat64_t c1{svsub_x(svptrue_b64(), ab1, qN1)};
    const svfloat64_t c0{svsub_x(svptrue_b64(), ab0, qN0)};
    svfloat64_t c;
    c = svadd_x(svptrue_b64(), c1, c0);
    c = svsub_m(svcmpge(svptrue_b64(), c, N), c, N);
    return svreinterpret_u64(c);
#endif
  }
};

template <class modulus_type_> struct UnsignedBinary64MontgomeryRadixWise {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    double Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    std::uint64_t Ninv{(N * 3) ^ 2}; /* 5 */
    Ninv *= 2 - N * Ninv;            /* 10 */
    Ninv *= 2 - N * Ninv;            /* 20 */
    Ninv *= 2 - N * Ninv;            /* 40 */
    Ninv *= 2 - N * Ninv;            /* 64 */
    Ninv %= std::uint64_t{1} << 52;
    return {
        .Ninv{
            static_cast<double>(Ninv),
        },
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(a));
  }

  static std::uint64_t from_operand(const std::uint64_t au) {
    const double ad{std::bit_cast<double>(au)};
    assert(std::remainder(ad, 1.) == 0);
    assert(0 <= ad && ad <= modulus_type::get_modulus() - 1);
    return static_cast<std::uint64_t>(ad);
  }

  static std::uint64_t to_multiplicand(std::uint64_t bu) {
    bu = modulus_type::multiply(bu, std::uint64_t{1} << 52);
    return std::bit_cast<std::uint64_t>(static_cast<double>(bu));
  }

  static std::uint64_t precompute(const std::uint64_t bu,
                                  const modulus_inverse_type modulus_inverse) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(
        static_cast<std::uint64_t>(std::bit_cast<double>(bu)) *
        static_cast<std::uint64_t>(modulus_inverse.Ninv) %
        (std::uint64_t{1} << 52)));
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d two104{_mm512_set1_pd(0x1p104)},
        two104_two52{_mm512_set1_pd(0x1p104 + 0x1p52)},
        two52{_mm512_set1_pd(0x1p52)}, twom52{_mm512_set1_pd(0x1p-52)};
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d a{_mm512_castsi512_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d q{_mm512_sub_pd(
        _mm512_fmadd_pd(
            a, bp, _mm512_sub_pd(two104_two52, _mm512_fmadd_pd(a, bp, two104))),
        two52)};
    const __m512d ab1{_mm512_mul_pd(
        _mm512_sub_pd(_mm512_fmadd_pd(a, b, two104), two104), twom52)};
    const __m512d qN1{_mm512_mul_pd(
        _mm512_sub_pd(_mm512_fmadd_pd(q, N, two104), two104), twom52)};
    __m512d c{_mm512_sub_pd(ab1, qN1)};
    c = _mm512_mask_add_pd(c, _mm512_cmplt_pd_mask(ab1, qN1), c, N);
    return _mm512_castpd_si512(c);
#elif defined(__ARM_FEATURE_SVE)
    constexpr double two104{0x1p104}, two104_two52{0x1p104 + 0x1p52},
        two52{0x1p52}, twom52{0x1p-52};
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svreinterpret_f64(ai)}, b{svreinterpret_f64(bi)},
        bp{svreinterpret_f64(bpi)};
    const svfloat64_t q1{svsubr_x(
        svptrue_b64(), svmad_x(svptrue_b64(), a, bp, two104), two104_two52)};
    const svfloat64_t q0{
        svsub_x(svptrue_b64(), svmad_x(svptrue_b64(), a, bp, q1), two52)};
    const svfloat64_t ab1{svmul_x(
        svptrue_b64(),
        svsub_x(svptrue_b64(), svmad_x(svptrue_b64(), a, b, two104), two104),
        twom52)};
    const svfloat64_t qN1{svmul_x(
        svptrue_b64(),
        svsub_x(svptrue_b64(), svmad_x(svptrue_b64(), q0, svdup_f64(N), two104),
                two104),
        twom52)};
    svfloat64_t c;
    c = svsub_x(svptrue_b64(), ab1, qN1);
    c = svadd_m(svcmplt(svptrue_b64(), ab1, qN1), c, N);
    return svreinterpret_u64(c);
#endif
  }
};

template <class modulus_type_> struct UnsignedBinary64ShoupExplicitR2I {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    double Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    const std::uint64_t N{modulus_type::get_modulus()};
    return {
        .Ninv{1. / N},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(a));
  }

  static std::uint64_t from_operand(const std::uint64_t au) {
    const double ad{std::bit_cast<double>(au)};
    assert(std::remainder(ad, 1.) == 0);
    assert(0 <= ad && ad <= modulus_type::get_modulus() - 1);
    return static_cast<std::uint64_t>(ad);
  }

  static std::uint64_t to_multiplicand(const std::uint64_t b) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(b));
  }

  static std::uint64_t
  precompute(const std::uint64_t b,
             [[maybe_unused]] const modulus_inverse_type modulus_inverse) {
    return std::bit_cast<std::uint64_t>(std::bit_cast<double>(b) /
                                        modulus_type::get_modulus());
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d a{_mm512_castsi512_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d ab1{_mm512_mul_pd(a, b)};
    const __m512d ab0{_mm512_fmsub_pd(a, b, ab1)};
    const __m512d q{
        _mm512_roundscale_pd(_mm512_mul_pd(a, bp), _MM_FROUND_CUR_DIRECTION)};
    const __m512d c1{_mm512_fnmadd_pd(q, N, ab1)};
    __m512d c0{_mm512_add_pd(c1, ab0)};
    c0 = _mm512_mask_sub_pd(c0, _mm512_cmpnlt_pd_mask(c0, N), c0, N);
    return _mm512_castpd_si512(c0);
#elif defined(__ARM_FEATURE_SVE)
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svreinterpret_f64(ai)}, b{svreinterpret_f64(bi)},
        bp{svreinterpret_f64(bpi)};
    const svfloat64_t ab1{svmul_x(svptrue_b64(), a, b)};
    const svfloat64_t ab0{svnmsb_x(svptrue_b64(), a, b, ab1)};
    const svfloat64_t q{
        svrinti_x(svptrue_b64(), svmul_x(svptrue_b64(), a, bp))};
    const svfloat64_t c1{svmls_x(svptrue_b64(), ab1, q, N)};
    svfloat64_t c0;
    c0 = svadd_x(svptrue_b64(), c1, ab0);
    c0 = svsub_m(svcmpge(svptrue_b64(), c0, N), c0, N);
    return svreinterpret_u64(c0);
#endif
  }
};

template <class modulus_type_> struct UnsignedBinary64ShoupImplicitR2I {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    double Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    const std::uint64_t N{modulus_type::get_modulus()};
    return {
        .Ninv{1. / N},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(a));
  }

  static std::uint64_t from_operand(const std::uint64_t au) {
    const double ad{std::bit_cast<double>(au)};
    assert(std::remainder(ad, 1.) == 0);
    assert(0 <= ad && ad <= modulus_type::get_modulus() - 1);
    return static_cast<std::uint64_t>(ad);
  }

  static std::uint64_t to_multiplicand(const std::uint64_t b) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(b));
  }

  static std::uint64_t
  precompute(const std::uint64_t b,
             [[maybe_unused]] const modulus_inverse_type modulus_inverse) {
    return std::bit_cast<std::uint64_t>(std::bit_cast<double>(b) /
                                        modulus_type::get_modulus());
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d round{_mm512_set1_pd(0x1p52 + 0x1p51)};
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d a{_mm512_castsi512_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d ab1{_mm512_mul_pd(a, b)};
    const __m512d ab0{_mm512_fmsub_pd(a, b, ab1)};
    const __m512d q{_mm512_sub_pd(_mm512_fmadd_pd(a, bp, round), round)};
    const __m512d c1{_mm512_fnmadd_pd(q, N, ab1)};
    __m512d c0{_mm512_add_pd(c1, ab0)};
    c0 = _mm512_mask_sub_pd(c0, _mm512_cmpnlt_pd_mask(c0, N), c0, N);
    return _mm512_castpd_si512(c0);
#elif defined(__ARM_FEATURE_SVE)
    constexpr double round{0x1p52 + 0x1p51};
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svreinterpret_f64(ai)}, b{svreinterpret_f64(bi)},
        bp{svreinterpret_f64(bpi)};
    const svfloat64_t ab1{svmul_x(svptrue_b64(), a, b)};
    const svfloat64_t ab0{svnmsb_x(svptrue_b64(), a, b, ab1)};
    const svfloat64_t q{
        svsub_x(svptrue_b64(), svmad_x(svptrue_b64(), a, bp, round), round)};
    const svfloat64_t c1{svmls_x(svptrue_b64(), ab1, q, N)};
    svfloat64_t c0;
    c0 = svadd_x(svptrue_b64(), c1, ab0);
    c0 = svsub_m(svcmpge(svptrue_b64(), c0, N), c0, N);
    return svreinterpret_u64(c0);
#endif
  }
};

template <class modulus_type_> struct UnsignedBinary64MontgomeryExplicitR2I {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    double Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    std::uint64_t Ninv{(N * 3) ^ 2}; /* 5 */
    Ninv *= 2 - N * Ninv;            /* 10 */
    Ninv *= 2 - N * Ninv;            /* 20 */
    Ninv *= 2 - N * Ninv;            /* 40 */
    Ninv *= 2 - N * Ninv;            /* 64 */
    Ninv %= std::uint64_t{1} << 53;
    return {
        .Ninv{
            static_cast<double>(Ninv),
        },
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(a));
  }

  static std::uint64_t from_operand(const std::uint64_t au) {
    const double ad{std::bit_cast<double>(au)};
    assert(std::remainder(ad, 1.) == 0);
    assert(0 <= ad && ad <= modulus_type::get_modulus() - 1);
    return static_cast<std::uint64_t>(ad);
  }

  static std::uint64_t to_multiplicand(std::uint64_t bu) {
    bu = modulus_type::multiply(bu, std::uint64_t{1} << 53);
    double bd{static_cast<double>(bu)};
    bd = std::scalbn(bd, -53);
    return std::bit_cast<std::uint64_t>(bd);
  }

  static std::uint64_t precompute(const std::uint64_t bu,
                                  const modulus_inverse_type modulus_inverse) {
#if defined(__AVX512F__)
    const __m128d Ninv{_mm_set_sd(modulus_inverse.Ninv)};
    const __m128d b{_mm_set_sd(std::bit_cast<double>(bu))};
    const __m128d bp1{_mm_roundscale_sd(_mm_set_sd({}), _mm_mul_sd(b, Ninv),
                                        _MM_FROUND_CUR_DIRECTION)};
    const __m128d bp0{_mm_fmsub_sd(b, Ninv, bp1)};
    return std::bit_cast<std::uint64_t>(_mm_cvtsd_f64(bp0));
#elif defined(__ARM_FEATURE_SVE)
    const svbool_t first{svptrue_pat_b64(SV_VL1)};
    const svfloat64_t Ninv{svdup_f64_x(first, modulus_inverse.Ninv)};
    const svfloat64_t b{svdup_f64_x(first, std::bit_cast<double>(bu))};
    const svfloat64_t bp1{svrinti_x(first, svmul_x(first, b, Ninv))};
    const svfloat64_t bp0{svnmsb_x(first, b, Ninv, bp1)};
    return std::bit_cast<std::uint64_t>(svlastb(first, bp0));
#endif
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d a{_mm512_castsi512_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d ab1{
        _mm512_roundscale_pd(_mm512_mul_pd(a, b), _MM_FROUND_CUR_DIRECTION)};
    const __m512d abp1{
        _mm512_roundscale_pd(_mm512_mul_pd(a, bp), _MM_FROUND_CUR_DIRECTION)};
    const __m512d q{_mm512_fmsub_pd(a, bp, abp1)};
    const __m512d qN1{
        _mm512_roundscale_pd(_mm512_mul_pd(q, N), _MM_FROUND_CUR_DIRECTION)};
    __m512d c{_mm512_sub_pd(ab1, qN1)};
    c = _mm512_mask_add_pd(c, _mm512_cmplt_pd_mask(ab1, qN1), c, N);
    return _mm512_castpd_si512(c);
#elif defined(__ARM_FEATURE_SVE)
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svreinterpret_f64(ai)}, b{svreinterpret_f64(bi)},
        bp{svreinterpret_f64(bpi)};
    const svfloat64_t ab1{
        svrinti_x(svptrue_b64(), svmul_x(svptrue_b64(), a, b))};
    const svfloat64_t abp1{
        svrinti_x(svptrue_b64(), svmul_x(svptrue_b64(), a, bp))};
    const svfloat64_t q{svnmsb_x(svptrue_b64(), a, bp, abp1)};
    const svfloat64_t qN1{
        svrinti_x(svptrue_b64(), svmul_x(svptrue_b64(), q, N))};
    svfloat64_t c;
    c = svsub_x(svptrue_b64(), ab1, qN1);
    c = svadd_m(svcmplt(svptrue_b64(), ab1, qN1), c, N);
    return svreinterpret_u64(c);
#endif
  }
};

template <class modulus_type_> struct UnsignedBinary64MontgomeryImplicitR2I {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    double Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    std::uint64_t Ninv{(N * 3) ^ 2}; /* 5 */
    Ninv *= 2 - N * Ninv;            /* 10 */
    Ninv *= 2 - N * Ninv;            /* 20 */
    Ninv *= 2 - N * Ninv;            /* 40 */
    Ninv *= 2 - N * Ninv;            /* 64 */
    Ninv %= std::uint64_t{1} << 53;
    return {
        .Ninv{
            static_cast<double>(Ninv),
        },
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) {
    return std::bit_cast<std::uint64_t>(static_cast<double>(a));
  }

  static std::uint64_t from_operand(const std::uint64_t a) {
    const double ad{std::bit_cast<double>(a)};
    assert(std::remainder(ad, 1.) == 0);
    assert(0 <= ad && ad <= modulus_type::get_modulus() - 1);
    return static_cast<std::uint64_t>(ad);
  }

  static std::uint64_t to_multiplicand(std::uint64_t bu) {
    bu = modulus_type::multiply(bu, std::uint64_t{1} << 53);
    double bd{static_cast<double>(bu)};
    bd = std::scalbn(bd, -53);
    return std::bit_cast<std::uint64_t>(bd);
  }

  static std::uint64_t precompute(const std::uint64_t bu,
                                  const modulus_inverse_type modulus_inverse) {
#if defined(__AVX512F__)
    const __m128d round{_mm_set_sd(0x1p52)};
    const __m128d Ninv{_mm_set_sd(modulus_inverse.Ninv)};
    const __m128d b{_mm_set_sd(std::bit_cast<double>(bu))};
    const __m128d bp1{_mm_sub_sd(_mm_fmadd_sd(b, Ninv, round), round)};
    const __m128d bp0{_mm_fmsub_sd(b, Ninv, bp1)};
    return std::bit_cast<std::uint64_t>(_mm_cvtsd_f64(bp0));
#elif defined(__ARM_FEATURE_SVE)
    constexpr double round{0x1p52};
    const svbool_t first{svptrue_pat_b64(SV_VL1)};
    const svfloat64_t Ninv{svdup_f64_x(first, modulus_inverse.Ninv)};
    const svfloat64_t b{svdup_f64_x(first, std::bit_cast<double>(bu))};
    const svfloat64_t bp1{
        svsub_x(first, svmad_x(first, b, Ninv, round), round)};
    const svfloat64_t bp0{svnmsb_x(first, b, Ninv, bp1)};
    return std::bit_cast<std::uint64_t>(svlastb(first, bp0));
#endif
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d round{_mm512_set1_pd(0x1p52)};
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d a{_mm512_castsi512_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d ab1{_mm512_fmadd_pd(a, b, round)};
    const __m512d abp1{_mm512_sub_pd(_mm512_fmadd_pd(a, bp, round), round)};
    const __m512d q{_mm512_fmsub_pd(a, bp, abp1)};
    const __m512d qN1{_mm512_fmadd_pd(q, N, round)};
    __m512d c{_mm512_sub_pd(ab1, qN1)};
    c = _mm512_mask_add_pd(c, _mm512_cmplt_pd_mask(ab1, qN1), c, N);
    return _mm512_castpd_si512(c);
#elif defined(__ARM_FEATURE_SVE)
    constexpr double round{0x1p52};
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svreinterpret_f64(ai)}, b{svreinterpret_f64(bi)},
        bp{svreinterpret_f64(bpi)};
    const svfloat64_t ab1{svmad_x(svptrue_b64(), a, b, round)};
    const svfloat64_t abp1{
        svsub_x(svptrue_b64(), svmad_x(svptrue_b64(), a, bp, round), round)};
    const svfloat64_t q{svnmsb_x(svptrue_b64(), a, bp, abp1)};
    const svfloat64_t qN1{svmad_x(svptrue_b64(), q, svdup_f64(N), round)};
    svfloat64_t c;
    c = svsub_x(svptrue_b64(), ab1, qN1);
    c = svadd_m(svcmplt(svptrue_b64(), ab1, qN1), c, N);
    return svreinterpret_u64(c);
#endif
  }
};

template <class modulus_type_> struct SignedBinary64ShoupExplicitR2I {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    double Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    const std::uint64_t N{modulus_type::get_modulus()};
    return {
        .Ninv{1. / N},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t au) {
    return std::bit_cast<std::uint64_t>(
        static_cast<double>(modulus_type::reduce_to_signed(au)));
  }

  static std::uint64_t from_operand(const std::uint64_t au) {
    double ad{std::bit_cast<double>(au)};
    assert(std::remainder(ad, 1.) == 0);
    assert(modulus_type::get_modulus() / -2. <= ad &&
           ad < modulus_type::get_modulus() / 2.);
    return modulus_type::reduce_to_unsigned(static_cast<std::int64_t>(ad));
  }

  static std::uint64_t to_multiplicand(const std::uint64_t bu) {
    return std::bit_cast<std::uint64_t>(
        static_cast<double>(modulus_type::reduce_to_signed(bu)));
  }

  static std::uint64_t
  precompute(const std::uint64_t b,
             [[maybe_unused]] const modulus_inverse_type modulus_inverse) {
    return std::bit_cast<std::uint64_t>(std::bit_cast<double>(b) /
                                        modulus_type::get_modulus());
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d half_N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()) / 2)},
        neg_half_N{_mm512_set1_pd(
            -static_cast<double>(modulus_type::get_modulus()) / 2)},
        half_thrice_N{_mm512_set1_pd(
            static_cast<double>(modulus_type::get_modulus() * 3 / 2))};
    const __m512d a{_mm512_castsi512_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d ab1{_mm512_mul_pd(a, b)};
    const __m512d ab0{_mm512_fmsub_pd(a, b, ab1)};
    const __m512d q{
        _mm512_roundscale_pd(_mm512_mul_pd(a, bp), _MM_FROUND_CUR_DIRECTION)};
    const __m512d c1{_mm512_fnmadd_pd(q, N, ab1)};
    const __m512d c0{_mm512_add_pd(c1, ab0)};
    __m512d t{c0};
    if constexpr (modulus_type::get_modulus() % 2 == 0) {
      t = _mm512_mask_sub_pd(t, _mm512_cmpeq_pd_mask(c0, half_thrice_N), t, N);
    }
    t = _mm512_mask_sub_pd(t, _mm512_cmpnlt_pd_mask(c0, half_N), t, N);
    t = _mm512_mask_add_pd(t, _mm512_cmplt_pd_mask(c0, neg_half_N), c0, N);
    return _mm512_castpd_si512(t);
#elif defined(__ARM_FEATURE_SVE)
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svreinterpret_f64(ai)}, b{svreinterpret_f64(bi)},
        bp{svreinterpret_f64(bpi)};
    const svfloat64_t ab1{svmul_x(svptrue_b64(), a, b)};
    const svfloat64_t ab0{svnmsb_x(svptrue_b64(), a, b, ab1)};
    const svfloat64_t q{
        svrinta_x(svptrue_b64(), svmul_x(svptrue_b64(), a, bp))};
    const svfloat64_t c1{svmls_x(svptrue_b64(), ab1, q, N)};
    const svfloat64_t c0{svadd_x(svptrue_b64(), c1, ab0)};
    svfloat64_t t{c0};
    if constexpr (modulus_type::get_modulus() % 2 == 0) {
      t = svsub_m(svcmpeq(svptrue_b64(), c0, N / 2 * 3), t, N);
    }
    t = svsub_m(svcmpge(svptrue_b64(), c0, N / 2), t, N);
    t = svadd_m(svcmplt(svptrue_b64(), c0, -N / 2), c0, N);
    return svreinterpret_u64(t);
#endif
  }
};

template <class modulus_type_> struct SignedBinary64ShoupImplicitR2I {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    double Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    const std::uint64_t N{modulus_type::get_modulus()};
    return {
        .Ninv{1. / N},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) {
    return std::bit_cast<std::uint64_t>(
        static_cast<double>(modulus_type::reduce_to_signed(a)));
  }

  static std::uint64_t from_operand(const std::uint64_t au) {
    const double ad{std::bit_cast<double>(au)};
    assert(std::remainder(ad, 1.) == 0);
    assert(modulus_type::get_modulus() / -2. <= ad &&
           ad < modulus_type::get_modulus() / 2.);
    return modulus_type::reduce_to_unsigned(static_cast<std::int64_t>(ad));
  }

  static std::uint64_t to_multiplicand(const std::uint64_t b) {
    return std::bit_cast<std::uint64_t>(
        static_cast<double>(modulus_type::reduce_to_signed(b)));
  }

  static std::uint64_t
  precompute(const std::uint64_t b,
             [[maybe_unused]] const modulus_inverse_type modulus_inverse) {
    return std::bit_cast<std::uint64_t>(std::bit_cast<double>(b) /
                                        modulus_type::get_modulus());
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d round{_mm512_set1_pd(0x1p52 + 0x1p51)};
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d half_N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()) / 2)},
        neg_half_N{_mm512_set1_pd(
            -static_cast<double>(modulus_type::get_modulus()) / 2)},
        half_thrice_N{_mm512_set1_pd(
            static_cast<double>(modulus_type::get_modulus()) / 2 * 3)};
    const __m512d a{_mm512_castsi512_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d ab1{_mm512_mul_pd(a, b)};
    const __m512d ab0{_mm512_fmsub_pd(a, b, ab1)};
    const __m512d q{_mm512_sub_pd(_mm512_fmadd_pd(a, bp, round), round)};
    const __m512d c1{_mm512_fnmadd_pd(q, N, ab1)};
    const __m512d c0{_mm512_add_pd(c1, ab0)};
    __m512d t{c0};
    if constexpr (modulus_type::get_modulus() % 2 == 0) {
      t = _mm512_mask_sub_pd(t, _mm512_cmpeq_pd_mask(c0, half_thrice_N), t, N);
    }
    t = _mm512_mask_sub_pd(t, _mm512_cmpnlt_pd_mask(c0, half_N), t, N);
    t = _mm512_mask_add_pd(t, _mm512_cmplt_pd_mask(c0, neg_half_N), c0, N);
    return _mm512_castpd_si512(t);
#elif defined(__ARM_FEATURE_SVE)
    constexpr double round{0x1p52 + 0x1p51};
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svreinterpret_f64(ai)}, b{svreinterpret_f64(bi)},
        bp{svreinterpret_f64(bpi)};
    const svfloat64_t ab1{svmul_x(svptrue_b64(), a, b)};
    const svfloat64_t ab0{svnmsb_x(svptrue_b64(), a, b, ab1)};
    svfloat64_t q{
        svsub_x(svptrue_b64(), svmad_x(svptrue_b64(), a, bp, round), round)};
    const svfloat64_t c1{svmls_x(svptrue_b64(), ab1, q, N)};
    const svfloat64_t c0{svadd_x(svptrue_b64(), c1, ab0)};
    svfloat64_t t{c0};
    if constexpr (modulus_type::get_modulus() % 2 == 0) {
      t = svsub_m(svcmpeq(svptrue_b64(), c0, N / 2 * 3), t, N);
    }
    t = svsub_m(svcmpge(svptrue_b64(), c0, N / 2), t, N);
    t = svadd_m(svcmplt(svptrue_b64(), c0, -N / 2), c0, N);
    return svreinterpret_u64(t);
#endif
  }
};

template <class modulus_type_> struct SignedBinary64MontgomeryExplicitR2I {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    double Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    std::uint64_t Ninv{(N * 3) ^ 2}; /* 5 */
    Ninv *= 2 - N * Ninv;            /* 10 */
    Ninv *= 2 - N * Ninv;            /* 20 */
    Ninv *= 2 - N * Ninv;            /* 40 */
    Ninv *= 2 - N * Ninv;            /* 64 */
    Ninv %= std::uint64_t{1} << 53;
    if (Ninv >> 52) {
      Ninv -= std::uint64_t{1} << 53;
    }
    return {
        .Ninv{
            static_cast<double>(static_cast<std::int64_t>(Ninv)),
        },
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) {
    return std::bit_cast<std::uint64_t>(
        static_cast<double>(modulus_type::reduce_to_signed(a)));
  }

  static std::uint64_t from_operand(const std::uint64_t au) {
    const double ad{std::bit_cast<double>(au)};
    assert(std::remainder(ad, 1.) == 0);
    assert(modulus_type::get_modulus() / -2. <= ad &&
           ad < modulus_type::get_modulus() / 2.);
    return modulus_type::reduce_to_unsigned(static_cast<std::int64_t>(ad));
  }

  static std::uint64_t to_multiplicand(const std::uint64_t bu) {
    const std::int64_t bs{modulus_type::multiply(
        modulus_type::reduce_to_signed(bu), std::int64_t{1} << 53)};
    double bd{static_cast<double>(bs)};
    bd = std::scalbn(bd, -53);
    return std::bit_cast<std::uint64_t>(bd);
  }

  static std::uint64_t precompute(const std::uint64_t bu,
                                  const modulus_inverse_type modulus_inverse) {
#if defined(__AVX512F__)
    const __m128d round{_mm_set_sd(0x1p52 + 0x1p51)};
    const __m128d Ninv{_mm_set_sd(modulus_inverse.Ninv)};
    const __m128d b{_mm_set_sd(std::bit_cast<double>(bu))};
    const __m128d bp1{_mm_sub_sd(_mm_fmadd_sd(b, Ninv, round), round)};
    const __m128d bp0{_mm_fmsub_sd(b, Ninv, bp1)};
    return std::bit_cast<std::uint64_t>(_mm_cvtsd_f64(bp0));
#elif defined(__ARM_FEATURE_SVE)
    const svbool_t first{svptrue_pat_b64(SV_VL1)};
    const svfloat64_t Ninv{svdup_f64_x(first, modulus_inverse.Ninv)};
    const svfloat64_t b{svdup_f64_x(first, std::bit_cast<double>(bu))};
    const svfloat64_t bp1{svrinti_x(first, svmul_x(first, b, Ninv))};
    const svfloat64_t bp0{svnmsb_x(first, b, Ninv, bp1)};
    return std::bit_cast<std::uint64_t>(svlastb(first, bp0));
#endif
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d half{_mm512_set1_pd(.5)};
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d half_N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()) / 2)},
        neg_half_N{_mm512_set1_pd(
            -static_cast<double>(modulus_type::get_modulus()) / 2)};
    const __m512d a{_mm512_castsi512_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d ab1{_mm512_roundscale_pd(_mm512_fmadd_pd(a, b, half),
                                           _MM_FROUND_CUR_DIRECTION)};
    const __m512d abp1{_mm512_roundscale_pd(_mm512_fmadd_pd(a, bp, half),
                                            _MM_FROUND_CUR_DIRECTION)};
    const __m512d q{_mm512_fmsub_pd(a, bp, abp1)};
    const __m512d qN1{_mm512_roundscale_pd(_mm512_fmadd_pd(q, N, half),
                                           _MM_FROUND_CUR_DIRECTION)};
    const __m512d c{_mm512_sub_pd(ab1, qN1)};
    __m512d t{c};
    t = _mm512_mask_add_pd(t, _mm512_cmplt_pd_mask(c, neg_half_N), c, N);
    t = _mm512_mask_sub_pd(t, _mm512_cmpnlt_pd_mask(c, half_N), c, N);
    return _mm512_castpd_si512(t);
#elif defined(__ARM_FEATURE_SVE)
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svreinterpret_f64(ai)}, b{svreinterpret_f64(bi)},
        bp{svreinterpret_f64(bpi)};
    const svfloat64_t ab1{
        svrinti_x(svptrue_b64(), svmad_x(svptrue_b64(), a, b, .5))};
    const svfloat64_t abp1{
        svrinti_x(svptrue_b64(), svmad_x(svptrue_b64(), a, bp, .5))};
    const svfloat64_t q{svnmsb_x(svptrue_b64(), a, bp, abp1)};
    const svfloat64_t qN1{
        svrinti_x(svptrue_b64(), svmad_x(svptrue_b64(), q, svdup_f64(N), .5))};
    const svfloat64_t c{svsub_x(svptrue_b64(), ab1, qN1)};
    const svbool_t is_excess{svcmpgt(svptrue_b64(), c, N / 2)},
        is_deficient{svacgt(is_excess, c, N / 2)};
    svfloat64_t t;
    t = svsub_m(is_excess, c, N);
    t = svadd_m(is_deficient, t, N);
    return svreinterpret_u64(t);
#endif
  }
};

template <class modulus_type_> struct SignedBinary64MontgomeryImplicitR2I {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    double Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    std::uint64_t Ninv{(N * 3) ^ 2}; /* 5 */
    Ninv *= 2 - N * Ninv;            /* 10 */
    Ninv *= 2 - N * Ninv;            /* 20 */
    Ninv *= 2 - N * Ninv;            /* 40 */
    Ninv *= 2 - N * Ninv;            /* 64 */
    Ninv %= std::uint64_t{1} << 53;
    if (Ninv >> 52) {
      Ninv -= std::uint64_t{1} << 53;
    }
    return {
        .Ninv{
            static_cast<double>(static_cast<std::int64_t>(Ninv)),
        },
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) {
    return std::bit_cast<std::uint64_t>(
        static_cast<double>(modulus_type::reduce_to_signed(a)));
  }

  static std::uint64_t from_operand(const std::uint64_t au) {
    const double ad{std::bit_cast<double>(au)};
    assert(std::remainder(ad, 1.) == 0);
    assert(modulus_type::get_modulus() / -2. <= ad &&
           ad < modulus_type::get_modulus() / 2.);
    return modulus_type::reduce_to_unsigned(static_cast<std::int64_t>(ad));
  }

  static std::uint64_t to_multiplicand(std::uint64_t bu) {
    const std::int64_t bs{modulus_type::multiply(
        modulus_type::reduce_to_signed(bu), std::int64_t{1} << 53)};
    double bd{static_cast<double>(bs)};
    bd = std::scalbn(bd, -53);
    return std::bit_cast<std::uint64_t>(bd);
  }

  static std::uint64_t precompute(const std::uint64_t bu,
                                  const modulus_inverse_type modulus_inverse) {
#if defined(__AVX512F__)
    const __m128d round{_mm_set_sd(0x1p52 + 0x1p51)};
    const __m128d Ninv{_mm_set_sd(modulus_inverse.Ninv)};
    const __m128d b{_mm_set_sd(std::bit_cast<double>(bu))};
    const __m128d bp1{_mm_sub_sd(_mm_fmadd_sd(b, Ninv, round), round)};
    const __m128d bp0{_mm_fmsub_sd(b, Ninv, bp1)};
    return std::bit_cast<std::uint64_t>(_mm_cvtsd_f64(bp0));
#elif defined(__ARM_FEATURE_SVE)
    constexpr double round{0x1p52 + 0x1p51};
    const svbool_t first{svptrue_pat_b64(SV_VL1)};
    const svfloat64_t Ninv{svdup_f64_x(first, modulus_inverse.Ninv)};
    const svfloat64_t b{svdup_f64_x(first, std::bit_cast<double>(bu))};
    const svfloat64_t bp1{
        svsub_x(first, svmad_x(first, b, Ninv, round), round)};
    const svfloat64_t bp0{svnmsb_x(first, b, Ninv, bp1)};
    return std::bit_cast<std::uint64_t>(svlastb(first, bp0));
#endif
  }

  static vector_register_type multiply(const vector_register_type ai,
                                       const vector_register_type bi,
                                       const vector_register_type bpi) {
#if defined(__AVX512F__)
    const __m512d round{_mm512_set1_pd(0x1p52 + 0x1p51)};
    const __m512d N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()))};
    const __m512d half_N{
        _mm512_set1_pd(static_cast<double>(modulus_type::get_modulus()) / 2)},
        neg_half_N{_mm512_set1_pd(
            -static_cast<double>(modulus_type::get_modulus()) / 2)};
    const __m512d a{_mm512_castsi512_pd(ai)}, b{_mm512_castsi512_pd(bi)},
        bp{_mm512_castsi512_pd(bpi)};
    const __m512d ab1{_mm512_fmadd_pd(a, b, round)};
    const __m512d abp1{_mm512_sub_pd(_mm512_fmadd_pd(a, bp, round), round)};
    const __m512d q{_mm512_fmsub_pd(a, bp, abp1)};
    const __m512d qN1{_mm512_fmadd_pd(q, N, round)};
    const __m512d c{_mm512_sub_pd(ab1, qN1)};
    __m512d t{c};
    t = _mm512_mask_add_pd(t, _mm512_cmplt_pd_mask(c, neg_half_N), c, N);
    t = _mm512_mask_sub_pd(t, _mm512_cmpnlt_pd_mask(c, half_N), c, N);
    return _mm512_castpd_si512(t);
#elif defined(__ARM_FEATURE_SVE)
    constexpr double round{0x1p52 + 0x1p51};
    constexpr double N{static_cast<double>(modulus_type::get_modulus())};
    const svfloat64_t a{svreinterpret_f64(ai)}, b{svreinterpret_f64(bi)},
        bp{svreinterpret_f64(bpi)};
    const svfloat64_t ab1{svmad_x(svptrue_b64(), a, b, round)};
    const svfloat64_t abp1{
        svsub_x(svptrue_b64(), svmad_x(svptrue_b64(), a, bp, round), round)};
    const svfloat64_t q{svnmsb_x(svptrue_b64(), a, bp, abp1)};
    const svfloat64_t qN1{svmad_x(svptrue_b64(), q, svdup_f64(N), round)};
    const svfloat64_t c{svsub_x(svptrue_b64(), ab1, qN1)};
    const svbool_t is_excess{svcmpgt(svptrue_b64(), c, N / 2)},
        is_deficient{svacgt(is_excess, c, N / 2)};
    svfloat64_t t;
    t = svsub_m(is_excess, c, N);
    t = svadd_m(is_deficient, t, N);
    return svreinterpret_u64(t);
#endif
  }
};

#if defined(__AVX512IFMA__)

template <class modulus_type_> struct UnsignedInt52Shoup {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    std::uint64_t Ninv_lo, Ninv_hi;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t mask52{(std::uint64_t{1} << 52) - 1};
    const std::uint64_t N{modulus_type::get_modulus()};
    const unsigned __int128 Ninv{((unsigned __int128){1} << 104) / N};
    return {
        .Ninv_lo{static_cast<std::uint64_t>(Ninv) & mask52},
        .Ninv_hi{static_cast<std::uint64_t>(Ninv >> 52)},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) { return a; }

  static std::uint64_t from_operand(const std::uint64_t a) {
    assert(a <= modulus_type::get_modulus() - 1);
    return a;
  }

  static std::uint64_t to_multiplicand(const std::uint64_t b) { return b; }

  static std::uint64_t precompute(const std::uint64_t b,
                                  const modulus_inverse_type modulus_inverse) {
    unsigned __int128 Ninv{modulus_inverse.Ninv_hi};
    Ninv <<= 52;
    Ninv |= modulus_inverse.Ninv_lo;
    Ninv *= b;
    Ninv >>= 52;
    return Ninv;
  }

  static __m512i multiply(const __m512i a, const __m512i b, const __m512i bp) {
    const __m512i zero{_mm512_setzero_si512()};
    const __m512i mask52{_mm512_set1_epi64((std::uint64_t{1} << 52) - 1)};
    const __m512i N_neg{_mm512_set1_epi64(-modulus_type::get_modulus())};
    const __m512i q{_mm512_madd52hi_epu64(zero, a, bp)};
    const __m512i ab{_mm512_madd52lo_epu64(zero, a, b)};
    __m512i c{_mm512_madd52lo_epu64(ab, q, N_neg)};
    c = _mm512_and_epi64(c, mask52);
    c = _mm512_min_epu64(c, _mm512_add_epi64(c, N_neg));
    return c;
  }
};

template <class modulus_type_> struct UnsignedInt52Montgomery {

  using modulus_type = modulus_type_;
  static_assert(modulus_type::get_modulus() % 2 == 1);

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    std::uint64_t Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    const std::uint64_t N{modulus_type::get_modulus()};
    std::uint64_t Ninv{(N * 3) ^ 2}; /* 5 */
    Ninv *= 2 - N * Ninv;            /* 10 */
    Ninv *= 2 - N * Ninv;            /* 20 */
    Ninv *= 2 - N * Ninv;            /* 40 */
    Ninv *= 2 - N * Ninv;            /* 64 */
    return {
        .Ninv{Ninv},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) { return a; }

  static std::uint64_t from_operand(const std::uint64_t a) {
    assert(a <= modulus_type::get_modulus() - 1);
    return a;
  }

  static std::uint64_t to_multiplicand(const std::uint64_t b) {
    return modulus_type::multiply(b, std::uint64_t{1} << 52);
  }

  static std::uint64_t precompute(const std::uint64_t b,
                                  const modulus_inverse_type modulus_inverse) {
    return b * modulus_inverse.Ninv;
  }

  static __m512i multiply(const __m512i a, const __m512i b, const __m512i bp) {
    const __m512i zero{_mm512_setzero_si512()};
    const __m512i N{_mm512_set1_epi64(modulus_type::get_modulus())};
    const __m512i q{_mm512_madd52lo_epu64(zero, a, bp)};
    const __m512i ab{_mm512_madd52hi_epu64(zero, a, b)};
    const __m512i qN{_mm512_madd52hi_epu64(zero, q, N)};
    __m512i c{_mm512_sub_epi64(ab, qN)};
    c = _mm512_min_epu64(c, _mm512_add_epi64(c, N));
    return c;
  }
};

#endif /* defined(__AVX512IFMA__) */

#if defined(__ARM_FEATURE_SVE)

template <class modulus_type_> struct UnsignedInt64Shoup {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    std::uint64_t Ninv_lo, Ninv_hi;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    /* For 2^128 / N == (2^128 - 1) / N. */
    static_assert(N % 2 == 1);
    const unsigned __int128 Ninv{~(unsigned __int128){0} /
                                 modulus_type::get_modulus()};
    return {
        .Ninv_lo{static_cast<std::uint64_t>(Ninv)},
        .Ninv_hi{static_cast<std::uint64_t>(Ninv >> 64)},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) { return a; }

  static std::uint64_t from_operand(const std::uint64_t a) {
    assert(a <= modulus_type::get_modulus() - 1);
    return a;
  }

  static std::uint64_t to_multiplicand(const std::uint64_t b) { return b; }

  static std::uint64_t precompute(const std::uint64_t b,
                                  const modulus_inverse_type modulus_inverse) {
    std::uint64_t bp;
    bp = static_cast<unsigned __int128>(modulus_inverse.Ninv_lo) * b >> 64;
    bp += modulus_inverse.Ninv_hi * b;
    return bp;
  }

  static vector_register_type multiply(const vector_register_type a,
                                       const vector_register_type b,
                                       const vector_register_type bp) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svuint64_t q{svmulh_x(svptrue_b64(), a, bp)};
    const svuint64_t ab{svmul_x(svptrue_b64(), a, b)};
    svuint64_t c;
    c = svmls_x(svptrue_b64(), ab, q, N);
    c = svmin_x(svptrue_b64(), c, svsub_x(svptrue_b64(), c, N));
    return c;
  }
};

template <class modulus_type_> struct UnsignedInt64Montgomery {

  using modulus_type = modulus_type_;

  using vector_register_type = VectorRegister;

  struct modulus_inverse_type {
    std::uint64_t Ninv;
  };

  static constexpr modulus_inverse_type get_modulus_inverse(void) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    static_assert(N % 2 == 1);
    std::uint64_t Ninv{(N * 3) ^ 2}; /* 5 */
    Ninv *= 2 - N * Ninv;            /* 10 */
    Ninv *= 2 - N * Ninv;            /* 20 */
    Ninv *= 2 - N * Ninv;            /* 40 */
    Ninv *= 2 - N * Ninv;            /* 64 */
    return {
        .Ninv{Ninv},
    };
  }

  static std::uint64_t to_operand(const std::uint64_t a) { return a; }

  static std::uint64_t from_operand(const std::uint64_t a) {
    assert(a <= modulus_type::get_modulus() - 1);
    return a;
  }

  static std::uint64_t to_multiplicand(const std::uint64_t b) {
    return (static_cast<unsigned __int128>(b) << 64) %
           modulus_type::get_modulus();
    return modulus_type::multiply(b, -modulus_type::get_modulus());
  }

  static std::uint64_t precompute(const std::uint64_t b,
                                  const modulus_inverse_type modulus_inverse) {
    return b * modulus_inverse.Ninv;
  }

  static vector_register_type multiply(const vector_register_type a,
                                       const vector_register_type b,
                                       const vector_register_type bp) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svuint64_t q{svmul_x(svptrue_b64(), a, bp)};
    const svuint64_t ab1{svmulh_x(svptrue_b64(), a, b)};
    const svuint64_t qN1{svmulh_x(svptrue_b64(), q, N)};
    svuint64_t c;
    c = svsub_x(svptrue_b64(), ab1, qN1);
    c = svmin_x(svptrue_b64(), c, svadd_x(svptrue_b64(), c, N));
    return c;
  }
};

#endif /* defined(__ARM_FEATURE_SVE) */

#endif /* FPMODMUL_MODMUL_HPP_INCLUDED */
