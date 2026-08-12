cmake_minimum_required(VERSION 3.20)
project(market_making_simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mm_order_book cpp/src/order_book.cpp)
target_include_directories(mm_order_book PUBLIC cpp/include)
target_compile_options(mm_order_book PRIVATE -Wall -Wextra -Wpedantic)

add_executable(order_book_tests cpp/tests/order_book_tests.cpp)
target_link_libraries(order_book_tests PRIVATE mm_order_book)
target_compile_options(order_book_tests PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME order_book_tests COMMAND order_book_tests)
