cmake_minimum_required(VERSION 3.10)
project(convolution LANGUAGES CUDA)
set(src src/main.cu)
add_executable(${PROJECT_NAME} ${src})

