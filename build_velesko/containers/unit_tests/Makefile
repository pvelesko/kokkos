KOKKOS_SETTINGS=KOKKOS_SRC_PATH=/home/pvelesko/local/kokkos_liber CXX=/soft/restricted/intel_dga/modulefiles/dpcpp/2020.03.15.001_wrapper/dpcpp KOKKOS_DEVICES=SYCL GTEST_PATH=/home/pvelesko/local/kokkos_liber/tpls/gtest KOKKOS_OPTIONS=disable_profiling KOKKOS_INTERNAL_GCC_TOOLCHAIN=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64 KOKKOS_PATH=/home/pvelesko/local/kokkos_liber

all:
	$(MAKE) -f /home/pvelesko/local/kokkos_liber/containers/unit_tests/Makefile KOKKOS_SRC_PATH=/home/pvelesko/local/kokkos_liber CXX=/soft/restricted/intel_dga/modulefiles/dpcpp/2020.03.15.001_wrapper/dpcpp KOKKOS_DEVICES=SYCL GTEST_PATH=/home/pvelesko/local/kokkos_liber/tpls/gtest KOKKOS_OPTIONS=disable_profiling KOKKOS_INTERNAL_GCC_TOOLCHAIN=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64 KOKKOS_PATH=/home/pvelesko/local/kokkos_liber

test: all
	$(MAKE) -f /home/pvelesko/local/kokkos_liber/containers/unit_tests/Makefile KOKKOS_SRC_PATH=/home/pvelesko/local/kokkos_liber CXX=/soft/restricted/intel_dga/modulefiles/dpcpp/2020.03.15.001_wrapper/dpcpp KOKKOS_DEVICES=SYCL GTEST_PATH=/home/pvelesko/local/kokkos_liber/tpls/gtest KOKKOS_OPTIONS=disable_profiling KOKKOS_INTERNAL_GCC_TOOLCHAIN=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64 KOKKOS_PATH=/home/pvelesko/local/kokkos_liber test

clean:
	$(MAKE) -f /home/pvelesko/local/kokkos_liber/containers/unit_tests/Makefile KOKKOS_SRC_PATH=/home/pvelesko/local/kokkos_liber CXX=/soft/restricted/intel_dga/modulefiles/dpcpp/2020.03.15.001_wrapper/dpcpp KOKKOS_DEVICES=SYCL GTEST_PATH=/home/pvelesko/local/kokkos_liber/tpls/gtest KOKKOS_OPTIONS=disable_profiling KOKKOS_INTERNAL_GCC_TOOLCHAIN=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64 KOKKOS_PATH=/home/pvelesko/local/kokkos_liber clean
