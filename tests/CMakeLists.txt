LIST(APPEND TESTS example)

foreach (test ${TESTS})
    add_executable(${test} ./${test}.cpp)
    add_test(NAME ${test} COMMAND ./${test})
endforeach(test)
