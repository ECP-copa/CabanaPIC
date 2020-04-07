# TODO: this will trigger a full rebuild of all files..

LIST(APPEND TESTS custom_init)

# TODO: we shouldn't have to duplicate all this
#INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src)

# TODO: this should live somewhere more common to reduce code duplication
foreach (test ${TESTS})
    set(DECK_PATH "${PROJECT_SOURCE_DIR}/decks/${test}")
    add_executable(${test} ${SOURCES} ${DECK_PATH})
    target_compile_definitions(${test} PRIVATE USER_INPUT_DECK=${DECK_PATH})
    target_link_libraries(${test} Cabana::cabanacore dl hwloc)
    add_test(NAME ${test} COMMAND ./${test})
endforeach(test)
