// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2024 Yukimasa Sugizaki

#ifndef FPMODMUL_MODULUS_HPP_INCLUDED
#define FPMODMUL_MODULUS_HPP_INCLUDED

#include <cstdint>
#include <utility>

template <std::uint64_t modulus> struct Modulus {

  static_assert(1 <= modulus && modulus < std::uint64_t{1} << 63);

  static constexpr std::uint64_t get_modulus(void) { return modulus; }

  static std::int64_t reduce_to_signed(const std::uint64_t a) {
    const std::int64_t b{static_cast<std::int64_t>(a % modulus)};
    return std::cmp_less(b, modulus / 2 + (modulus % 2)) ? b : b - modulus;
  }

  static std::uint64_t reduce_to_unsigned(std::int64_t a) {
    a %= std::int64_t{modulus};
    return a < 0 ? a + modulus : a;
  }

  static std::uint64_t multiply(const std::uint64_t a, const std::uint64_t b) {
    return static_cast<unsigned __int128>(a) * b % modulus;
  }

  static std::int64_t multiply(const std::int64_t a, const std::int64_t b) {
    const std::int64_t c{static_cast<std::int64_t>(static_cast<__int128>(a) *
                                                   b % std::int64_t{modulus})};
    return c < -std::int64_t{modulus / 2}                  ? c + modulus
           : std::cmp_less(c, modulus / 2 + (modulus % 2)) ? c
                                                           : c - modulus;
  }

  template <class T>
    requires std::same_as<T, std::uint64_t> || std::same_as<T, std::int64_t>
  static T power(T a, std::uint64_t e) {
    T b{1};
    for (; e; e >>= 1) {
      if (e % 2 == 1) {
        b = multiply(a, b);
      }
      a = multiply(a, a);
    }
    return b;
  }
};

#endif /* FPMODMUL_MODULUS_HPP_INCLUDED */
