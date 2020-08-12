#ifndef KOKKOS_SYCL_KERNELLAUNCH_HPP_
#define KOKKOS_SYCL_KERNELLAUNCH_HPP_

#include <SYCL/Kokkos_TypeChecks.hpp>
#include <SYCL/Kokkos_SYCL_Error.hpp>
#include <CL/sycl.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template<class Driver>
void sycl_launch(const Driver driver) {
       driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
       driver.m_policy.space().impl_internal_space_instance()->m_queue->submit([&] (cl::sycl::handler& cgh) {
         cgh.parallel_for (
            cl::sycl::range<1>(driver.m_policy.end()-driver.m_policy.begin()), [=] (cl::sycl::item<1> item) {
              int id = item.get_linear_id();
                driver.m_functor(id);        
         });
      });
      driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
}


} // Impl namespace
} // Experimental namespace
} // Kokkos namespace

#endif // KOKKOS_SYCL_KERNELLAUNCH_HPP_
