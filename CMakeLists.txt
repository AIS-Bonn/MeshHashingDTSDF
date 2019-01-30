cmake_minimum_required(VERSION 3.5)
PROJECT(mesh-hashing C CXX)

set(OpenGL_GL_PREFERENCE GLVND)

#----------
# C++ version setting
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall")

#----------
# CUDA compilation settings
set(CUDA_SEPARABLE_COMPILATION ON) # enable splitting code in .h and .cu files (otherwise linking issues)
LIST(APPEND CUDA_NVCC_FLAGS
  -gencode=arch=compute_50,code=sm_50
  -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored # suppress __host__/__device__ annotation warnings
  )

#----------
# Project variable configurations
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

#----------
# Find packages
FIND_PACKAGE(PkgConfig)
LIST(APPEND
  CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake_modules)

# Graphics dependent
FIND_PACKAGE(GLFW3 REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
# CV utilities
FIND_PACKAGE(OpenCV 3 REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
# Parallel computation
FIND_PACKAGE(CUDA REQUIRED)

# Log utilities
FIND_PACKAGE(Glog REQUIRED)

FIND_PACKAGE(yaml-cpp REQUIRED)
INCLUDE_DIRECTORIES(${YAML_CPP_INCLUDE_DIR})

#----------
## Found Headers
GET_PROPERTY(included_dirs
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY INCLUDE_DIRECTORIES)
MESSAGE(STATUS "Found include paths:")
foreach (included_dir ${included_dirs})
  MESSAGE(STATUS ${included_dir})
endforeach ()

#----------
## Found Libs
SET(OPENGL_DEPENDENCIES "")
LIST(APPEND
  OPENGL_DEPENDENCIES
  ${OPENGL_LIBRARY}
  ${GLEW_LIBRARY})
if (APPLE)
  LIST(APPEND OPENGL_DEPENDENCIES
    ${GLFW3_LIBRARIES})
else ()
  LIST(APPEND OPENGL_DEPENDENCIES
    ${GLFW3_STATIC_LIBRARIES})
endif ()

SET(CUDA_DEPENDENCIES "")
LIST(APPEND
  CUDA_DEPENDENCIES
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CUDA_LIBRARY})

MESSAGE(STATUS "Found libraries:")
foreach (library ${OPENGL_DEPENDENCIES})
  MESSAGE(STATUS ${library})
endforeach ()
foreach (library ${CUDA_DEPENDENCIES})
  MESSAGE(STATUS ${library})
endforeach ()
foreach (library ${OTHER_DEPENDENCIES})
  MESSAGE(STATUS ${library})
endforeach ()
#---------

#----------
SET(VH ${PROJECT_SOURCE_DIR}/src)
SET(GL_WRAPPER ${VH}/extern/opengl-wrapper)

INCLUDE_DIRECTORIES(
  ${VH}
  ${VH}/extern/cuda
  /opt/ros/kinetic/include
  ${GL_WRAPPER}/include
  ${EIGEN3_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_INCLUDE})

#----------
## Building Libraries
### 1. OpenGL util
ADD_LIBRARY(gl-util
  ${GL_WRAPPER}/src/core/args.cc
  ${GL_WRAPPER}/src/core/program.cc
  ${GL_WRAPPER}/src/core/window.cc
  ${GL_WRAPPER}/src/core/uniforms.cc
  ${GL_WRAPPER}/src/core/camera.cc)
SET_TARGET_PROPERTIES(gl-util
  PROPERTIES
  COMPILE_DEFINITIONS USE_CUDA_GL)
TARGET_LINK_LIBRARIES(gl-util
  ${OPENGL_DEPENDENCIES}
  ${GLOG_LIBRARIES})


### 2. CUDA
CUDA_ADD_LIBRARY(mesh-hashing-cuda
  ${CORE_HEADERS}
  ${VH}/core/cuda_memory_heap.cu
  ${VH}/core/hash_table.cu
  ${VH}/core/entry_array.cu
  ${VH}/core/block_array.cu
  ${VH}/core/mesh.cu
  ${VH}/core/collect_block_array.cu
  ${VH}/core/directional_tsdf.cu

  ${VH}/sensor/rgbd_sensor.cu
  ${VH}/sensor/preprocess.cu

  ${VH}/localizing/point_to_psdf.cu
  ${VH}/mapping/allocate.cu
  ${VH}/mapping/update_simple.cu
  ${VH}/mapping/update_bayesian.cu
  ${VH}/mapping/update_raycasting.cu
  ${VH}/mapping/recycle.cu

  ${VH}/optimize/linear_equations.cu
  #        ${VH}/optimize/primal_dual.cu

  ${VH}/meshing/marching_cubes.cu
  ${VH}/meshing/marching_cubes_directional.cu

  ${VH}/visualization/colorize.cu
  ${VH}/visualization/compact_mesh.cu
  ${VH}/visualization/bounding_box.cu
  ${VH}/visualization/trajectory.cu
  ${VH}/visualization/compress_mesh.cu
  ${VH}/visualization/extract_bounding_box.cu
  ${VH}/visualization/ray_caster.cu
  )

TARGET_LINK_LIBRARIES(mesh-hashing-cuda
  ${CUDA_DEPENDENCIES}
  ${GLOG_LIBRARIES}
  ${OpenCV_LIBRARIES})

### 3. C++
ADD_LIBRARY(mesh-hashing
  ${VH}/engine/main_engine.cc
  ${VH}/engine/mapping_engine.cc
  ${VH}/engine/logging_engine.cc
  ${VH}/engine/visualizing_engine.cc

  ${VH}/io/config_manager.cc
  ${VH}/io/mesh_writer.cc

  ${VH}/sensor/rgbd_data_provider.cc)
#${VH}/tool/cpp/debugger.cc)
SET_TARGET_PROPERTIES(mesh-hashing
  PROPERTIES
  COMPILE_DEFINITIONS USE_CUDA_GL)
TARGET_LINK_LIBRARIES(mesh-hashing
  stdc++fs
  mesh-hashing-cuda
  ${YAML_CPP_LIBRARIES}
  gl-util)

#----------
### Loop over
MESSAGE(STATUS ${GLOG_LIBRARIES})
ADD_EXECUTABLE(reconstruction src/app/reconstruction.cc)
SET_TARGET_PROPERTIES(reconstruction
  PROPERTIES
  COMPILE_DEFINITIONS USE_CUDA_GL)
TARGET_LINK_LIBRARIES(reconstruction
  mesh-hashing
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES})

ADD_EXECUTABLE(slam src/app/slam.cc)
SET_TARGET_PROPERTIES(slam
  PROPERTIES
  COMPILE_DEFINITIONS USE_CUDA_GL)
TARGET_LINK_LIBRARIES(slam
  mesh-hashing
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES})

ADD_EXECUTABLE(block_analysis src/app/block_analysis.cc)
SET_TARGET_PROPERTIES(block_analysis
  PROPERTIES
  COMPILE_DEFINITIONS USE_CUDA_GL)
TARGET_LINK_LIBRARIES(block_analysis
  gl-util
  mesh-hashing
  ${OpenCV_LIBRARIES}
  ${GLOG_LIBRARIES})

add_subdirectory(test)

### An ORB app
#OPTION(WITH_ORBSLAM2 "Build with orb slam" ON)
#if (WITH_ORBSLAM2)
#    SET(ORB_SLAM2_PATH ${VH}/extern/orb_slam2)
#    MESSAGE(STATUS "Build with ORB-SLAM2")
#
#    FIND_PACKAGE(Pangolin REQUIRED)
#    FIND_PACKAGE(Eigen3 REQUIRED)
#    INCLUDE_DIRECTORIES(
#            ${ORB_SLAM2_PATH}
#            ${ORB_SLAM2_PATH}/include
#            ${EIGEN3_INCLUDE_DIR})
#    LINK_DIRECTORIES(${ORB_SLAM2_PATH}/lib)
#
#    ADD_EXECUTABLE(orb_slam2 src/app/orb_slam2.cc)
#    TARGET_LINK_LIBRARIES(orb_slam2
#            mesh-hashing
#            ORB_SLAM2
#            ${OpenCV_LIBS}
#            ${Pangolin_LIBRARIES}
#            ${GLOG_LIBRARIES})
#
#    SET_TARGET_PROPERTIES(orb_slam2
#            PROPERTIES
#            COMPILE_DEFINITIONS USE_CUDA_GL)
#endif(WITH_ORBSLAM2)
