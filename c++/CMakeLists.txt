cmake_minimum_required(VERSION 3.16)
project(manga_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(manga src/main.cpp)
target_link_libraries(manga PRIVATE ${OpenCV_LIBS})
target_include_directories(manga PRIVATE ${OpenCV_INCLUDE_DIRS})
