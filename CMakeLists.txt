cmake_minimum_required(VERSION 3.10)
project(demo C)

# Use MPI wrapper compiler
enable_language(C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRE on)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# PETSc paths
set(PETSC_DIR "/work/mae-wangzq/apps/petsc")
#set(PETSC_ARCH "arch-linux-c-debug")
set(PETSC_ARCH ".")
set(PETSC_INCLUDE_DIRS
  ${PETSC_DIR}/include
  ${PETSC_DIR}/${PETSC_ARCH}/include
)
set(PETSC_LIB_DIR ${PETSC_DIR}/${PETSC_ARCH}/lib)
set(PETSC_LIBRARIES petsc)

# HDF5 paths
set(HDF5_ROOT "/apps/mae-wangzq/apps/hdf5")
set(HDF5_INCLUDE_DIRS ${HDF5_ROOT}/include)
set(HDF5_LIB_DIR ${HDF5_ROOT}/lib)
set(HDF5_LIBRARIES hdf5)

# Add executable
add_executable(exp exp.c)
add_executable(imp imp.c)
add_executable(cn CN.c)
add_executable(imp2d imp2d.c)

# Include directories
target_include_directories(imp2d PRIVATE
  ${PETSC_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIRS}
  ${MPI_INCLUDE_PATH}
)

# Link directories and libraries
target_link_directories(imp2d PRIVATE
  ${PETSC_LIB_DIR}
  ${HDF5_LIB_DIR}
)

target_link_libraries(imp2d PRIVATE
  m
  ${PETSC_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${MPI_C_LIBRARIES}
)

# Set rpath so runtime can find shared libs
set_target_properties(imp2d PROPERTIES
  BUILD_RPATH "${PETSC_LIB_DIR};${HDF5_LIB_DIR}"
)

# Include directories
target_include_directories(exp PRIVATE
  ${PETSC_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIRS}
  ${MPI_INCLUDE_PATH}
)

# Link directories and libraries
target_link_directories(exp PRIVATE
  ${PETSC_LIB_DIR}
  ${HDF5_LIB_DIR}
)

target_link_libraries(exp PRIVATE
  m
  ${PETSC_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${MPI_C_LIBRARIES}
)

# Set rpath so runtime can find shared libs
set_target_properties(exp PROPERTIES
  BUILD_RPATH "${PETSC_LIB_DIR};${HDF5_LIB_DIR}"
)

# Include directories
target_include_directories(imp PRIVATE
  ${PETSC_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIRS}
  ${MPI_INCLUDE_PATH}
)

# Link directories and libraries
target_link_directories(imp PRIVATE
  ${PETSC_LIB_DIR}
  ${HDF5_LIB_DIR}
)

target_link_libraries(imp PRIVATE
  m
  ${PETSC_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${MPI_C_LIBRARIES}
)

# Set rpath so runtime can find shared libs
set_target_properties(imp PROPERTIES
  BUILD_RPATH "${PETSC_LIB_DIR};${HDF5_LIB_DIR}"
)

# Include directories
target_include_directories(cn PRIVATE
  ${PETSC_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIRS}
  ${MPI_INCLUDE_PATH}
)

# Link directories and libraries
target_link_directories(cn PRIVATE
  ${PETSC_LIB_DIR}
  ${HDF5_LIB_DIR}
)

target_link_libraries(cn PRIVATE
  m
  ${PETSC_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${MPI_C_LIBRARIES}
)

# Set rpath so runtime can find shared libs
set_target_properties(cn PROPERTIES
  BUILD_RPATH "${PETSC_LIB_DIR};${HDF5_LIB_DIR}"
)
