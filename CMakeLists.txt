# For the cxx_std_20 feature.
cmake_minimum_required(VERSION 3.12)

project(floating_point_modmul LANGUAGES CXX)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(TARGET_UARCH
      sapphirerapids
      CACHE STRING "Target microarchitecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(TARGET_UARCH
      neoverse-v2
      CACHE STRING "Target microarchitecture")
  set(SVE_VECTOR_BITS
      128
      CACHE STRING "SVE vector width in bits")
endif()

find_package(benchmark REQUIRED)

add_executable(bench_repeated_multiplications
               bench_repeated_multiplications.cpp)
target_compile_features(bench_repeated_multiplications PRIVATE cxx_std_20)
target_include_directories(bench_repeated_multiplications PRIVATE .)
target_compile_options(bench_repeated_multiplications PRIVATE -W -Wall -Wextra)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_compile_options(
    bench_repeated_multiplications PRIVATE "-march=${TARGET_UARCH}"
                                           "-mtune=${TARGET_UARCH}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_compile_options(
    bench_repeated_multiplications
    PRIVATE "-mcpu=${TARGET_UARCH}" "-mtune=${TARGET_UARCH}"
            "-msve-vector-bits=${SVE_VECTOR_BITS}")
endif()
target_link_libraries(bench_repeated_multiplications benchmark::benchmark)
