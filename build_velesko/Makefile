KOKKOS_SETTINGS=KOKKOS_SRC_PATH=/home/pvelesko/local/kokkos_liber CXX=/soft/restricted/intel_dga/modulefiles/dpcpp/2020.03.15.001_wrapper/dpcpp KOKKOS_DEVICES=SYCL GTEST_PATH=/home/pvelesko/local/kokkos_liber/tpls/gtest KOKKOS_OPTIONS=disable_profiling KOKKOS_INTERNAL_GCC_TOOLCHAIN=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64 KOKKOS_PATH=/home/pvelesko/local/kokkos_liber

kokkoslib:
	cd core; \
	$(MAKE) -f /home/pvelesko/local/kokkos_liber/core/src/Makefile KOKKOS_SRC_PATH=/home/pvelesko/local/kokkos_liber CXX=/soft/restricted/intel_dga/modulefiles/dpcpp/2020.03.15.001_wrapper/dpcpp KOKKOS_DEVICES=SYCL GTEST_PATH=/home/pvelesko/local/kokkos_liber/tpls/gtest KOKKOS_OPTIONS=disable_profiling KOKKOS_INTERNAL_GCC_TOOLCHAIN=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64 KOKKOS_PATH=/home/pvelesko/local/kokkos_liber PREFIX=/home/pvelesko/local/kokkos_liber/build_velesko/install build-lib

install: kokkoslib
	cd core; \
	$(MAKE) -f /home/pvelesko/local/kokkos_liber/core/src/Makefile KOKKOS_SRC_PATH=/home/pvelesko/local/kokkos_liber CXX=/soft/restricted/intel_dga/modulefiles/dpcpp/2020.03.15.001_wrapper/dpcpp KOKKOS_DEVICES=SYCL GTEST_PATH=/home/pvelesko/local/kokkos_liber/tpls/gtest KOKKOS_OPTIONS=disable_profiling KOKKOS_INTERNAL_GCC_TOOLCHAIN=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64 KOKKOS_PATH=/home/pvelesko/local/kokkos_liber PREFIX=/home/pvelesko/local/kokkos_liber/build_velesko/install install

kokkoslib-test:
	cd core; \
	$(MAKE) -f /home/pvelesko/local/kokkos_liber/core/src/Makefile KOKKOS_SRC_PATH=/home/pvelesko/local/kokkos_liber CXX=/soft/restricted/intel_dga/modulefiles/dpcpp/2020.03.15.001_wrapper/dpcpp KOKKOS_DEVICES=SYCL GTEST_PATH=/home/pvelesko/local/kokkos_liber/tpls/gtest KOKKOS_OPTIONS=disable_profiling KOKKOS_INTERNAL_GCC_TOOLCHAIN=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64 KOKKOS_PATH=/home/pvelesko/local/kokkos_liber PREFIX=/home/pvelesko/local/kokkos_liber/build_velesko/install build-lib

install-test: kokkoslib-test
	cd core; \
	$(MAKE) -f /home/pvelesko/local/kokkos_liber/core/src/Makefile KOKKOS_SRC_PATH=/home/pvelesko/local/kokkos_liber CXX=/soft/restricted/intel_dga/modulefiles/dpcpp/2020.03.15.001_wrapper/dpcpp KOKKOS_DEVICES=SYCL GTEST_PATH=/home/pvelesko/local/kokkos_liber/tpls/gtest KOKKOS_OPTIONS=disable_profiling KOKKOS_INTERNAL_GCC_TOOLCHAIN=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64 KOKKOS_PATH=/home/pvelesko/local/kokkos_liber PREFIX=/home/pvelesko/local/kokkos_liber/build_velesko/install install

build-test: install-test
	$(MAKE) -C core/unit_test
	$(MAKE) -C core/perf_test
	$(MAKE) -C containers/unit_tests
	$(MAKE) -C containers/performance_tests
	$(MAKE) -C algorithms/unit_tests
	$(MAKE) -C example/fixture
	$(MAKE) -C example/feint
	$(MAKE) -C example/fenl
	$(MAKE) -C example/make_buildlink build
	$(MAKE) -C example/tutorial build

test: build-test
	$(MAKE) -C core/unit_test test
	$(MAKE) -C core/perf_test test
	$(MAKE) -C containers/unit_tests test
	$(MAKE) -C containers/performance_tests test
	$(MAKE) -C algorithms/unit_tests test
	$(MAKE) -C example/fixture test
	$(MAKE) -C example/feint test
	$(MAKE) -C example/fenl test
	$(MAKE) -C example/make_buildlink test
	$(MAKE) -C example/tutorial test

unit-tests-only:
	$(MAKE) -C core/unit_test test
	$(MAKE) -C containers/unit_tests test
	$(MAKE) -C algorithms/unit_tests test

clean:
	$(MAKE) -C core/unit_test clean
	$(MAKE) -C core/perf_test clean
	$(MAKE) -C containers/unit_tests clean
	$(MAKE) -C containers/performance_tests clean
	$(MAKE) -C algorithms/unit_tests clean
	$(MAKE) -C example/fixture clean
	$(MAKE) -C example/feint clean
	$(MAKE) -C example/fenl clean
	$(MAKE) -C example/make_buildlink clean
	$(MAKE) -C example/tutorial clean
	cd core; \
	$(MAKE) -f /home/pvelesko/local/kokkos_liber/core/src/Makefile KOKKOS_SRC_PATH=/home/pvelesko/local/kokkos_liber CXX=/soft/restricted/intel_dga/modulefiles/dpcpp/2020.03.15.001_wrapper/dpcpp KOKKOS_DEVICES=SYCL GTEST_PATH=/home/pvelesko/local/kokkos_liber/tpls/gtest KOKKOS_OPTIONS=disable_profiling KOKKOS_INTERNAL_GCC_TOOLCHAIN=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64 KOKKOS_PATH=/home/pvelesko/local/kokkos_liber clean
