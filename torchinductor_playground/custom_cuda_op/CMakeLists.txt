cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(my_ops)

find_package(Python COMPONENTS Interpreter Development)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Torch REQUIRED)

# Define our library target
add_library(my_ops SHARED custom_op.cpp)
# Enable C++14
target_compile_features(my_ops PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(my_ops "${TORCH_LIBRARIES}" "${PYTHON_LIBRARIES}")
