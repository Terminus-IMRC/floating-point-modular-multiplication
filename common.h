// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2024 Yukimasa Sugizaki

#ifndef FPMODMUL_COMMON_HPP_INCLUDED
#define FPMODMUL_COMMON_HPP_INCLUDED

#include <utility>

template <class sequence_type, class... sequence_types, class function_type>
static constexpr void for_each_sequence(function_type function) {
  [function]<class value_type, value_type... values>(
      std::integer_sequence<value_type, values...>) {
    if constexpr (sizeof...(sequence_types) == 0) {
      (function.template operator()<values>(), ...);
    } else {
      (for_each_sequence<sequence_types...>(
           [function]<typename sequence_types::value_type... arguments> {
             function.template operator()<values, arguments...>();
           }),
       ...);
    }
  }(sequence_type{});
}

#endif /* FPMODMUL_COMMON_HPP_INCLUDED */
