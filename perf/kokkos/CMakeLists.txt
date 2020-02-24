set(TARGET perf_kokkos)
file(GLOB SOURCES "*.cpp")
add_executable(${TARGET} ${SOURCES} ${INPUT_DECK})
target_link_libraries(${TARGET} cabanacore kokkos)
