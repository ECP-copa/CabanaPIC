cmake_minimum_required(VERSION 3.9)
project(CabanaPIC LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR})
include(GNUInstallDirs)

# TODO: Tag this once we have a new release
find_package(Cabana)

#### User configuration Options ####
option(REQUIRE_HOST ON "Build with the default host execution space.")
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
    message(FATAL_ERROR "SOLVER_TYPE is not supported (EM/ES only)")
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

###### Allow user to select input deck to build against ######
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
####### End User Deck ######

if(ENABLE_COVERAGE_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif(ENABLE_COVERAGE_BUILD)

add_subdirectory(src)
set(CabanaPIC_EXAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/example/policy)
add_subdirectory(example/policy)

##### TESTS ######
if (ENABLE_TESTS)
  enable_testing()
  set(TEST_DIR "./tests/include")
  include_directories(${TEST_DIR})
  add_subdirectory(tests)
endif(ENABLE_TESTS)
