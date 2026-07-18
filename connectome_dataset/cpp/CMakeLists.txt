cmake_minimum_required(VERSION 3.24)
project(connectome_bench LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(CONNECTOME_BENCH_BUILD_BENCHMARKS "Build Google Benchmark targets" OFF)
option(CONNECTOME_BENCH_BUILD_CUDA "Build CUDA baselines (cuSPARSE)" OFF)

find_package(Eigen3 REQUIRED)

add_library(connectome_bench_core
  src/cases.cc
  src/loaders.cc
  src/measure.cc
  src/records.cc
  src/registry.cc
  src/runner.cc
  src/writers.cc
)

add_library(connectome_bench::core ALIAS connectome_bench_core)

target_compile_features(connectome_bench_core PUBLIC cxx_std_20)
target_link_libraries(connectome_bench_core PUBLIC Eigen3::Eigen)
target_include_directories(
  connectome_bench_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_executable(connectome_bench
  apps/connectome_bench_main.cc
  baselines/reference/spmv.cc
  baselines/reference/spgemm.cc
)
target_link_libraries(connectome_bench PRIVATE connectome_bench_core)
target_include_directories(connectome_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/baselines)

# ── cuSPARSE baseline (opt-in; cuSPARSE is a host API, so no nvcc / CUDA language) ──
if(CONNECTOME_BENCH_BUILD_CUDA OR CONNECTOME_BENCH_BUILD_SPUTNIK)
  find_package(CUDAToolkit REQUIRED)
  target_sources(connectome_bench PRIVATE baselines/cusparse/spmv.cc)
  target_compile_definitions(connectome_bench PRIVATE CONNECTOME_BENCH_HAVE_CUSPARSE)
  target_link_libraries(connectome_bench PRIVATE CUDA::cusparse CUDA::cudart)
endif()

# ── Sputnik SpMM baseline (opt-in; vendored external/sputnik, needs CUDA language) ──
option(CONNECTOME_BENCH_BUILD_SPUTNIK "Build the Sputnik SpMM baseline (external/sputnik + CUDA)" OFF)
if(CONNECTOME_BENCH_BUILD_SPUTNIK)
  enable_language(CUDA)
  set(CONNECTOME_BENCH_CUDA_ARCH "120" CACHE STRING "CUDA arch for vendored kernels (e.g. 120 = Blackwell)")
  set(SPUTNIK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../external/sputnik" CACHE PATH "Sputnik repo root")
  set(_sputnik_spmm "${SPUTNIK_ROOT}/sputnik/spmm/cuda_spmm.cu.cc")
  if(NOT EXISTS "${_sputnik_spmm}")
    message(FATAL_ERROR "Sputnik not found at ${SPUTNIK_ROOT}; run scripts/fetch_external.sh")
  endif()
  # Compile Sputnik's SpMM as CUDA; a minimal glog shim (third_party/glog_shim) stands in
  # for glog, whose modern version is source-incompatible with Sputnik.
  set_source_files_properties(${_sputnik_spmm} PROPERTIES LANGUAGE CUDA)
  target_sources(connectome_bench PRIVATE baselines/sputnik/spmv.cc ${_sputnik_spmm})
  target_include_directories(connectome_bench PRIVATE
    ${SPUTNIK_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glog_shim)
  target_compile_definitions(connectome_bench PRIVATE CONNECTOME_BENCH_HAVE_SPUTNIK)
  target_compile_options(connectome_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(connectome_bench PROPERTIES CUDA_ARCHITECTURES "${CONNECTOME_BENCH_CUDA_ARCH}")

  # Shared C-ABI library for the out-of-tree Python provider (loaded via ctypes).
  find_package(CUDAToolkit REQUIRED)
  add_library(connectome_sputnik SHARED baselines/sputnik/sputnik_c.cc ${_sputnik_spmm})
  target_include_directories(connectome_sputnik PRIVATE
    ${SPUTNIK_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glog_shim)
  target_compile_options(connectome_sputnik PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(connectome_sputnik PROPERTIES CUDA_ARCHITECTURES "${CONNECTOME_BENCH_CUDA_ARCH}")
  target_link_libraries(connectome_sputnik PRIVATE CUDA::cudart)
endif()

install(
  TARGETS connectome_bench_core connectome_bench
  EXPORT connectome_benchTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/connectome_bench DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  EXPORT connectome_benchTargets
  FILE connectome_benchTargets.cmake
  NAMESPACE connectome_bench::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/connectome_bench
)

configure_package_config_file(
  cmake/connectome_benchConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/connectome_benchConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/connectome_bench
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/connectome_benchConfigVersion.cmake
  VERSION 0.1.0
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/connectome_benchConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/connectome_benchConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/connectome_bench
)

# ── Google Benchmark targets (opt-in) ────────────────────────────────────────
if(CONNECTOME_BENCH_BUILD_BENCHMARKS)
  find_package(benchmark QUIET)
  if(NOT benchmark_FOUND)
    message(STATUS "google/benchmark not found via find_package — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
      googlebenchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.9.1
      GIT_SHALLOW    TRUE
    )
    set(BENCHMARK_ENABLE_TESTING        OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS    OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS          OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
  endif()

  # benchmarks/eigen/bench_spmv — Eigen CSR SpMV / SpMM
  add_executable(bench_spmv benchmarks/eigen/bench_spmv.cc)
  target_link_libraries(bench_spmv PRIVATE connectome_bench_core benchmark::benchmark_main)
  target_compile_features(bench_spmv PRIVATE cxx_std_20)
endif()
