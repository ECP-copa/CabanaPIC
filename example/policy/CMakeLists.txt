add_executable(cbnpic example.cpp ${PROJECT_SOURCE_DIR}/${INPUT_DECK})
target_link_libraries(cbnpic PUBLIC CabanaPIC)
target_include_directories( cbnpic PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
install(TARGETS cbnpic DESTINATION ${CMAKE_INSTALL_BINDIR})
