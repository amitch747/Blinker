cmake_minimum_required(VERSION 3.10)
project(Blinker)

set(CMAKE_CXX_STANDARD 17)
add_executable(cursor_server cursor_server.cpp)
target_link_libraries(cursor_server ws2_32)
