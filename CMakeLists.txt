cmake_minimum_required(VERSION 3.9)
project(minipic LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${PROJECT_SOURCE_DIR})

#### User configuration Options ####
option(ENABLE_GPU OFF)
option(ENABLE_SERIAL OFF)

option(ENABLE_TESTS OFF)
option(ENABLE_COVERAGE_BUILD OFF)
#### End User configuration Options ####

##### SET SOLVES TYPE #####
# Flag for switching between electromagnetic and electrostatic solver
set(SOLVER_TYPE "EM" CACHE STRING "Selected Solver Type")
set(SolverTypes EM ES) # List allowable solver types
# hint the tools the allowed values
set_property(CACHE SOLVER_TYPE PROPERTY STRINGS ${SolverTypes})
if (${SOLVER_TYPE} STREQUAL "EM")
    add_definitions(-DEM_FIELD_SOLVER=YES)
elseif (${SOLVER_TYPE} STREQUAL "ES")
    add_definitions(-DES_FIELD_SOLVER=YES)
else()
    message(FATAL_ERROR "SOLVER_TYPE is not support (EM/ES only)")
endif()
##### END SET SOLVES TYPE #####

##### SET DIMENSIONALITY #####
set(DIMENSIONALITY "3" CACHE STRING "Selected Solver Type")
set(ALLOWABLE_DIMENSIONS 1 2 3) # List allowable values
# hint the tools the allowed values
set_property(CACHE DIMENSIONALITY PROPERTY STRINGS ${ALLOWABLE_DIMENSIONS})

if (NOT ${DIMENSIONALITY} STREQUAL "3")
    message(FATAL_ERROR "DIMENSIONALITY != 3 not yet supported")
endif()
##### END SET DIMENSIONALITY #####

##### SET REAL_TYPE (real_t) #####
set(REAL_TYPE "float" CACHE STRING "Selected type for real numbers")
set(ALLOWABLE_REALS "float" "double") # List allowable values
set_property(CACHE REAL_TYPE PROPERTY STRINGS ${ALLOWABLE_REALS})
add_definitions(-DREAL_TYPE=${REAL_TYPE})
##### END SET REAL_TYPE #####

# Allow user to select input deck to build against
set(INPUT_DECK "" CACHE STRING "Path to input deck")

if (NOT ${INPUT_DECK} STREQUAL "")
    # TODO: normalize these paths?
    if(EXISTS ${PROJECT_SOURCE_DIR}/${INPUT_DECK})
        add_definitions(-DUSER_INPUT_DECK=${PROJECT_SOURCE_DIR}/${INPUT_DECK})
    elseif(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_DECK})
        add_definitions(-DUSER_INPUT_DECK=${CMAKE_CURRENT_BINARY_DIR}/${INPUT_DECK})
    else()
    message(FATAL_ERROR "Cannot find user specified input deck: ${INPUT_DECK}")
    endif()
endif()

if (ENABLE_SERIAL)
    add_definitions(-DUSE_SERIAL_CPU=ON)
endif(ENABLE_SERIAL)

if (ENABLE_GPU)
    add_definitions(-DUSE_GPU=ON)
    find_package(CUDA)
    if (CUDA_FOUND)
        message("Found CUDA")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda -g")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored") # supress express cuda warnings
    else()
        message( FATAL_ERROR "Requested CUDA but cant find it!")
    endif(CUDA_FOUND)
endif(ENABLE_GPU)

INCLUDE_DIRECTORIES(./src)
file(GLOB SOURCES "src/*.cpp")


##### Build Executable ########
INCLUDE_DIRECTORIES(${CABANA_DIR}/include/)
INCLUDE_DIRECTORIES(${CABANA_DIR}/include/Cabana/)
LINK_DIRECTORIES(${CABANA_DIR}/lib)
LINK_DIRECTORIES(${CABANA_DIR}/lib64)

INCLUDE_DIRECTORIES(${KOKKOS_DIR}/include)
LINK_DIRECTORIES(${KOKKOS_DIR}/lib)

add_executable(minipic ${SOURCES} ${INPUT_DECK})
target_link_libraries(minipic cabanacore kokkos dl)


##### TESTS ######
if (ENABLE_TESTS)
  enable_testing()
  set(TEST_DIR "./tests/include")
  include_directories(${TEST_DIR})
  add_subdirectory(tests)
endif(ENABLE_TESTS)

if(ENABLE_COVERAGE_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif(ENABLE_COVERAGE_BUILD)

