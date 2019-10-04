cmake_minimum_required(VERSION 3.9)
project(minipic LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB SOURCES "src/*.cpp")

set(INPUT_DECK "" CACHE STRING "Path to input deck")

if (NOT ${INPUT_DECK} STREQUAL "")
    # TODO: normalize these paths?
    if(EXISTS ${PROJECT_SOURCE_DIR}/${INPUT_DECK})
        # Append deck to sources list
        list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/${INPUT_DECK}")
        add_definitions(-DUSER_INPUT_DECK=${PROJECT_SOURCE_DIR}/${INPUT_DECK})
    elseif(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_DECK})
        list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${INPUT_DECK}")
        add_definitions(-DUSER_INPUT_DECK=${CMAKE_CURRENT_BINARY_DIR}/${INPUT_DECK})
    else()
    message( FATAL_ERROR "Cannot find user specified input deck: ${INPUT_DECK}")
    endif()
endif()


option(ENABLE_GPU OFF)
option(ENABLE_SERIAL OFF)

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

# Try guess where Cabana lives..
if(NOT DEFINED CABANA_DIR)
    set(CABANA_DIR ~/tools/Cabana/)
endif()

if(NOT DEFINED KOKKOS_DIR)
    set(KOKKOS_DIR ~/tools/kokkos/)
endif()

INCLUDE_DIRECTORIES(./src)

INCLUDE_DIRECTORIES(${CABANA_DIR}/include/)
INCLUDE_DIRECTORIES(${CABANA_DIR}/include/Cabana/)
LINK_DIRECTORIES(${CABANA_DIR}/lib)
LINK_DIRECTORIES(${CABANA_DIR}/lib64)

INCLUDE_DIRECTORIES(${KOKKOS_DIR}/include)
LINK_DIRECTORIES(${KOKKOS_DIR}/lib)

add_executable(minipic ${SOURCES})
target_link_libraries(minipic cabanacore kokkos hwloc dl)
