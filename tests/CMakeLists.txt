add_subdirectory(decks)
if (${SOLVER_TYPE} STREQUAL "EM")
  add_subdirectory(energy_comparison)
endif()

#LIST(APPEND TESTS example)

foreach (test ${TESTS})
    add_executable(${test} ./${test}.cpp)
    target_link_libraries(${test} CabanaPIC)
    add_test(NAME ${test} COMMAND ./${test})
endforeach(test)
