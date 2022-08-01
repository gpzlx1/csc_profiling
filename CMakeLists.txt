cmake_minimum_required(VERSION 3.15.0)
project(csc_profiling LANGUAGES CUDA CXX)

set(CMAKE_PREFIX_PATH "$(python -c 'import torch;print(torch.utils.cmake_prefix_path)'")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Define our library target
add_library(csc_profiling SHARED
    src/hello_world.cu
    )

set_target_properties(csc_profiling PROPERTIES CUDA_ARCHITECTURES "86")
target_compile_features(csc_profiling PRIVATE cxx_std_14)
target_link_libraries(csc_profiling ${TORCH_LIBRARIES} ${CUDA_cusparse_LIBRARY})
