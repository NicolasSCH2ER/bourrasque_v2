# Bourrasque — core MLS-MPM CUDA
#
# Build (Windows, VS2026 Build Tools, CUDA >= 13.0 pour sm_120 / RTX 50xx) :
#   cmake -B build -G "Visual Studio 18 2026" -A x64
#   cmake --build build --config Release
#
# VS2022 + CUDA 12.8 reste supporté : -G "Visual Studio 17 2022"
# Autre GPU : -DCMAKE_CUDA_ARCHITECTURES=native
cmake_minimum_required(VERSION 3.24)

# Doit précéder project() : la détection du compilateur CUDA renseigne elle-même
# CMAKE_CUDA_ARCHITECTURES avec le défaut de nvcc (sm_75), ce qui rendrait un
# garde placé après project() inopérant.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)   # Blackwell (RTX 50xx)
endif()

project(bourrasque LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(bourrasque SHARED core/src/mlsmpm.cu)
target_include_directories(bourrasque PUBLIC core/include)
set_target_properties(bourrasque PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)

add_executable(bourrasque_headless core/headless/main.cpp)
target_link_libraries(bourrasque_headless PRIVATE bourrasque)

# La DLL atterrit dans l'extension pour le packaging Blender
add_custom_command(TARGET bourrasque POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_SOURCE_DIR}/extension/bin
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:bourrasque>
            ${CMAKE_SOURCE_DIR}/extension/bin/)
