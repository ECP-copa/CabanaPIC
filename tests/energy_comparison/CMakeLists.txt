# TODO: this will trigger a full rebuild of all files..

LIST(APPEND TESTS 2stream-em)

# TODO: we shouldn't have to duplicate all this
#INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src)

list(APPEND gold_file "${CMAKE_CURRENT_SOURCE_DIR}/energies_gold")

# TODO: avoid duplicating this link line everywhere
foreach (test ${TESTS})
    set(DECK_PATH "${test}")
    add_executable(${test} ${SOURCES} ${DECK_PATH})
    target_compile_definitions(${test} PRIVATE USER_INPUT_DECK=${DECK_PATH})
    target_compile_definitions(${test} PRIVATE GOLD_ENERGY_FILE=${gold_file}.${test}.${REAL_TYPE})
    target_link_libraries(${test} Cabana::cabanacore dl hwloc)
    add_test(NAME ${test} COMMAND ./${test})
endforeach(test)
