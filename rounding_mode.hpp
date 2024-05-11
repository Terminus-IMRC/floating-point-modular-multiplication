// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2024 Yukimasa Sugizaki

#ifndef FPMODMUL_ROUNDING_MODE_HPP_INCLUDED
#define FPMODMUL_ROUNDING_MODE_HPP_INCLUDED

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <cstdint>
#else
#error unsupported target
#endif

class RoundingMode {

public:
  enum class mode_type {
#if defined(__x86_64__)
    round_to_nearest = _MM_ROUND_NEAREST,
    round_toward_infinity = _MM_ROUND_UP,
    round_toward_neg_infinity = _MM_ROUND_DOWN,
    round_toward_zero = _MM_ROUND_TOWARD_ZERO,
#elif defined(__aarch64__)
    round_to_nearest = 0,
    round_toward_infinity = 1,
    round_toward_neg_infinity = 2,
    round_toward_zero = 3,
#endif
  };

  static std::float_round_style mode_to_style(const mode_type mode) {
    switch (mode) {
    case mode_type::round_to_nearest:
      return std::round_to_nearest;
    case mode_type::round_toward_infinity:
      return std::round_toward_infinity;
    case mode_type::round_toward_neg_infinity:
      return std::round_toward_neg_infinity;
    case mode_type::round_toward_zero:
      return std::round_toward_zero;
    default:
      throw std::runtime_error{
          "Invalid round mode: " +
          std::to_string(static_cast<std::underlying_type_t<mode_type>>(mode))};
    }
  }

  static mode_type style_to_mode(const std::float_round_style style) {
    switch (style) {
    case std::round_to_nearest:
      return mode_type::round_to_nearest;
    case std::round_toward_infinity:
      return mode_type::round_toward_infinity;
    case std::round_toward_neg_infinity:
      return mode_type::round_toward_neg_infinity;
    case std::round_toward_zero:
      return mode_type::round_toward_zero;
    default:
      throw std::runtime_error{"Invalid round style: " + std::to_string(style)};
    }
  }

  static mode_type get_mode(void) {
#if defined(__x86_64__)
    return static_cast<mode_type>(_MM_GET_ROUNDING_MODE());
#elif defined(__aarch64__)
    std::uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return static_cast<mode_type>((fpcr >> 22) & 3);
#endif
  }

  static std::float_round_style get_style(void) {
    return mode_to_style(get_mode());
  }

  static mode_type set(const mode_type new_mode) {
    const mode_type old_mode{get_mode()};
#if defined(__x86_64__)
    _MM_SET_ROUNDING_MODE(static_cast<unsigned>(new_mode));
#elif defined(__aarch64__)
    std::uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr = (fpcr & ~(std::uint64_t{3} << 22)) |
           (static_cast<std::uint64_t>(new_mode) << 22);
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
    return old_mode;
  }

  static std::float_round_style set(const std::float_round_style style) {
    return mode_to_style(set(style_to_mode(style)));
  }
};

class RoundingModePush {

  const std::float_round_style style;

public:
  RoundingModePush(const std::float_round_style style)
      : style{RoundingMode::set(style)} {}

  ~RoundingModePush(void) { static_cast<void>(RoundingMode::set(style)); }
};

#endif /* FPMODMUL_ROUNDING_MODE_HPP_INCLUDED */
