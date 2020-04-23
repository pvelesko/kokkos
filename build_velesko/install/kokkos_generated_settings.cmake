#Global Settings used to generate this library
set(KOKKOS_PATH CACHE FILEPATH "Kokkos installation path" FORCE)
set(KOKKOS_GMAKE_DEVICES "SYCL" CACHE STRING "Kokkos devices list" FORCE)
set(KOKKOS_GMAKE_ARCH "" CACHE STRING "Kokkos architecture flags" FORCE)
set(KOKKOS_DEBUG_CMAKE OFF CACHE BOOL "Kokkos debug enabled ?" FORCE)
set(KOKKOS_GMAKE_USE_TPLS "" CACHE STRING "Kokkos templates list" FORCE)
set(KOKKOS_CXX_STANDARD c++11 CACHE STRING "Kokkos C++ standard" FORCE)
set(KOKKOS_GMAKE_OPTIONS "disable_profiling" CACHE STRING "Kokkos options" FORCE)
set(KOKKOS_GMAKE_CUDA_OPTIONS "" CACHE STRING "Kokkos Cuda options" FORCE)
set(KOKKOS_GMAKE_TPL_INCLUDE_DIRS "" CACHE STRING "Kokkos TPL include directories" FORCE)
set(KOKKOS_GMAKE_TPL_LIBRARY_DIRS "" CACHE STRING "Kokkos TPL library directories" FORCE)
set(KOKKOS_GMAKE_TPL_LIBRARY_NAMES " dl" CACHE STRING "Kokkos TPL library names" FORCE)
if(NOT DEFINED ENV{NVCC_WRAPPER})
set(NVCC_WRAPPER /home/pvelesko/local/kokkos_liber/bin/nvcc_wrapper CACHE FILEPATH "Path to command nvcc_wrapper" FORCE)
else()
  set(NVCC_WRAPPER $ENV{NVCC_WRAPPER} CACHE FILEPATH "Path to command nvcc_wrapper")
endif()

#Source and Header files of Kokkos relative to KOKKOS_PATH
set(KOKKOS_HEADERS /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Core_fwd.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Pair.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_NumericTraits.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Timer.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_hwloc.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Vectorization.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Extents.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Layout.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Crs.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ROCm.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_MasterLock.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Threads.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_HBWSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_SYCL.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Serial.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Parallel_Reduce.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Qthreads.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_TaskScheduler_fwd.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Cuda.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_View.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_TaskPolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_PointerOwnership.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_WorkGraphPolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_AnonymousSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Atomic.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_OpenMPTarget.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_TaskScheduler.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_UniqueToken.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Core.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ScratchSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Parallel.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ROCmSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/KokkosExp_MDRangePolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_MemoryPool.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_SYCL_Space.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_CudaSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_OpenMPTargetSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_OpenMP.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Array.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Future.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Profiling_ProfileSection.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_CopyViews.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Complex.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_HostSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Concepts.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ExecPolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_HPX.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Macros.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_MemoryTraits.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewFillCopyETIDecl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Spinwait.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Traits.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_SingleTaskQueue.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewMapping.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Compare_Exchange_Strong.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Decrement.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueueMultiple_impl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_AnalyzePolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_FunctorAdapter.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ChaseLev.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_FunctorAnalysis.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Fetch_Add.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueueMultiple.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskPolicyData.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Tags.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_LIFO.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Store.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_HostThreadTeam.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Profiling_Interface.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewArray.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_HostBarrier.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Windows.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueueCommon.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_LinkedListNode.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_CPUDiscovery.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/KokkosExp_Host_IterateTile.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_HostSpace_deepcopy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Profiling_DeviceInfo.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_SharedAlloc.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Generic.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Load.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_FixedBufferMemoryPool.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Fetch_Or.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_SimpleTaskScheduler.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Assembly.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ClockTic.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewTile.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_OptionalRef.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ConcurrentBitset.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewFillCopyETIAvail.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Utilities.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Increment.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_BitOps.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Memory_Fence.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/KokkosExp_ViewMapping.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskBase.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewUniformType.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Exchange.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_OldMacros.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Volatile_Load.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskTeamMember.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Fetch_And.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Fetch_Sub.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Serial_WorkGraphPolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueue.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueueMemoryManager.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewLayoutTiled.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_PhysicalLayout.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewCtor.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskResult.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Compare_Exchange_Weak.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_View.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Serial_Task.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_MultipleTaskQueue.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskNode.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_VLAEmulation.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_EBO.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Error.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueue_impl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Timer.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_MemoryPoolAllocator.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Memory_Order.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Functional.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_StaticCrsGraph.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ErrorReporter.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_OffsetView.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_DualView.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Vector.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ScatterView.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Bitset.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_UnorderedMap.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_DynamicView.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_DynRankView.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_StaticCrsGraph_factory.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Functional_impl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Bitset_impl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_UnorderedMap_impl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Random.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Sort.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Error.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_View.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Instance.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Parallel_MDRange.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Parallel_Team.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Atomic.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Parallel_Range.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_KernelLaunch.hpp CACHE STRING "Kokkos headers list" FORCE)
set(KOKKOS_HEADERS_IMPL CACHE STRING "Kokkos headers impl list" FORCE)
set(KOKKOS_HEADERS_CUDA CACHE STRING "Kokkos headers Cuda list" FORCE)
set(KOKKOS_HEADERS_OPENMP CACHE STRING "Kokkos headers OpenMP list" FORCE)
set(KOKKOS_HEADERS_HPX CACHE STRING "Kokkos headers HPX list" FORCE)
set(KOKKOS_HEADERS_ROCM CACHE STRING "Kokkos headers ROCm list" FORCE)
set(KOKKOS_HEADERS_THREADS CACHE STRING "Kokkos headers Threads list" FORCE)
set(KOKKOS_HEADERS_QTHREADS CACHE STRING "Kokkos headers QThreads list" FORCE)

#Variables used in application Makefiles
set(KOKKOS_OS Linux CACHE STRING "" FORCE)
set(KOKKOS_CPP_DEPENDS KokkosCore_config.h /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Serial.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_hwloc.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ExecPolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Macros.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Qthreads.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_NumericTraits.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Core_fwd.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Pair.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_UniqueToken.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_PointerOwnership.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Profiling_ProfileSection.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_OpenMP.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Vectorization.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Atomic.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_OpenMPTargetSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Crs.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_CudaSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ScratchSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Layout.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_CopyViews.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Parallel_Reduce.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_SYCL_Space.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_AnonymousSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_OpenMPTarget.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ROCmSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Core.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_HBWSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_TaskScheduler.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Cuda.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_View.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/KokkosExp_MDRangePolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Parallel.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Complex.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Future.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_HostSpace.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_MasterLock.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_WorkGraphPolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_MemoryPool.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_TaskPolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_TaskScheduler_fwd.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_MemoryTraits.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Timer.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Concepts.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Array.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Threads.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_HPX.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ROCm.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Extents.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_SYCL.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ConcurrentBitset.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Fetch_Sub.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Serial_Task.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Windows.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskTeamMember.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_FunctorAnalysis.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueue.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_HostThreadTeam.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Fetch_And.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/KokkosExp_ViewMapping.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_MultipleTaskQueue.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_SharedAlloc.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Spinwait.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Compare_Exchange_Weak.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_LinkedListNode.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Volatile_Load.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewArray.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Fetch_Add.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Generic.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueue_impl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskNode.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_FunctorAdapter.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Serial_WorkGraphPolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_SingleTaskQueue.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Assembly.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_AnalyzePolicy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Utilities.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewUniformType.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Load.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_VLAEmulation.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_CPUDiscovery.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Profiling_Interface.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_OptionalRef.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_HostBarrier.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Memory_Fence.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Decrement.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Error.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_View.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_EBO.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskPolicyData.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ChaseLev.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueueMultiple.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Traits.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/KokkosExp_Host_IterateTile.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Compare_Exchange_Strong.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Exchange.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Memory_Order.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_BitOps.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewFillCopyETIAvail.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueueMultiple_impl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Profiling_DeviceInfo.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskResult.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueueCommon.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Increment.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewCtor.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_PhysicalLayout.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Timer.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewMapping.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_FixedBufferMemoryPool.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_HostSpace_deepcopy.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewLayoutTiled.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Fetch_Or.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewFillCopyETIDecl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_LIFO.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ClockTic.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Tags.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskQueueMemoryManager.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_OldMacros.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Atomic_Store.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_SimpleTaskScheduler.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_TaskBase.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_MemoryPoolAllocator.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_ViewTile.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_DynRankView.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Vector.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_StaticCrsGraph.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Functional.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_DualView.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_DynamicView.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Bitset.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_UnorderedMap.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ScatterView.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_ErrorReporter.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_OffsetView.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_StaticCrsGraph_factory.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Functional_impl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_Bitset_impl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/impl/Kokkos_UnorderedMap_impl.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Random.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/Kokkos_Sort.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Error.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Parallel_Range.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_View.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_KernelLaunch.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Instance.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Parallel_MDRange.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Parallel_Team.hpp /home/pvelesko/local/kokkos_liber/build_velesko/install/include/SYCL/Kokkos_SYCL_Atomic.hpp CACHE STRING "" FORCE)
set(KOKKOS_LINK_DEPENDS libkokkos.a CACHE STRING "" FORCE)
set(KOKKOS_CXXFLAGS -I./ -I/home/pvelesko/local/kokkos_liber/build_velesko/install/include -I/home/pvelesko/local/kokkos_liber/build_velesko/install/include -I/home/pvelesko/local/kokkos_liber/build_velesko/install/include -I/home/pvelesko/local/kokkos_liber/build_velesko/install/include/eti --std=c++11 CACHE STRING "" FORCE)
set(KOKKOS_CPPFLAGS -fsycl -fsycl-unnamed-lambda -DSYCL_JUST_DONT_NAME_KERNELS CACHE STRING "" FORCE)
set(KOKKOS_LDFLAGS -L/home/pvelesko/local/kokkos_liber/build_velesko/install/lib -fsycl CACHE STRING "" FORCE)
set(KOKKOS_CXXLDFLAGS -L/home/pvelesko/local/kokkos_liber/build_velesko/install/lib CACHE STRING "" FORCE)
set(KOKKOS_LIBS -lkokkos -ldl -lOpenCL -lsycl CACHE STRING "" FORCE)
set(KOKKOS_EXTRA_LIBS -ldl -lOpenCL -lsycl CACHE STRING "" FORCE)
set(KOKKOS_LINK_FLAGS CACHE STRING "extra flags to the link step (e.g. OpenMP)" FORCE)

#Internal settings which need to propagated for Kokkos examples
set(KOKKOS_INTERNAL_USE_CUDA 0 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_OPENMP 0 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_HPX 0 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_PTHREADS 0 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_SERIAL 1 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_ROCM 0 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_HPX 0 CACHE STRING "" FORCE)
set(KOKKOS_INTERNAL_USE_QTHREADS 0 CACHE STRING "" FORCE)
set(KOKKOS_SRC /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_HostBarrier.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_ExecPolicy.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_HostThreadTeam.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_Serial_Task.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_Serial.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_SharedAlloc.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_Spinwait.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_HostSpace_deepcopy.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_Profiling_Interface.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_hwloc.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_CPUDiscovery.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_MemoryPool.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_Error.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_Core.cpp /home/pvelesko/local/kokkos_liber/core/src/impl/Kokkos_HostSpace.cpp /home/pvelesko/local/kokkos_liber/containers/src/impl/Kokkos_UnorderedMap_impl.cpp /home/pvelesko/local/kokkos_liber/core/src/SYCL/Kokkos_SYCL_Space.cpp /home/pvelesko/local/kokkos_liber/core/src/SYCL/Kokkos_SYCL_Instance.cpp CACHE STRING "Kokkos source list" FORCE)
set(KOKKOS_CXX_FLAGS -I./ -I/home/pvelesko/local/kokkos_liber/core/src -I/home/pvelesko/local/kokkos_liber/containers/src -I/home/pvelesko/local/kokkos_liber/algorithms/src -I/home/pvelesko/local/kokkos_liber/core/src/eti --std=c++11)
set(KOKKOS_CPP_FLAGS -fsycl -fsycl-unnamed-lambda -DSYCL_JUST_DONT_NAME_KERNELS)
set(KOKKOS_LD_FLAGS -L/home/pvelesko/local/kokkos_liber/build_velesko/core -fsycl)
set(KOKKOS_LIBS_LIST "-lkokkos -ldl -lOpenCL -lsycl")
set(KOKKOS_EXTRA_LIBS_LIST "-ldl -lOpenCL -lsycl")
set(KOKKOS_LINK_FLAGS )
