# TODO: this will trigger a full rebuild of all files..

LIST(APPEND TESTS custom_init)

# TODO: we shouldn't have to duplicate all this
message(${CMAKE_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CABANA_DIR}/include)
INCLUDE_DIRECTORIES(${KOKKOS_DIR}/include)
LINK_DIRECTORIES(${CABANA_DIR}/lib)
LINK_DIRECTORIES(${KOKKOS_DIR}/lib)

message(${SOURCES})
foreach (test ${TESTS})
    set(DECK_PATH "${PROJECT_SOURCE_DIR}/decks/${test}")
    message(${DECK_PATH})
    add_executable(${test} ${SOURCES} ${DECK_PATH})
    target_compile_definitions(${test} PRIVATE USER_INPUT_DECK=${DECK_PATH})
    target_link_libraries(${test} cabanacore kokkos dl)
    add_test(NAME ${test} COMMAND ./${test})
endforeach(test)
