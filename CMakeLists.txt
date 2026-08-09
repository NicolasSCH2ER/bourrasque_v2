# Bourrasque — core MLS-MPM CUDA
#
# Build (Windows, VS2022 Build Tools, CUDA >= 12.8) :
#   cmake -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
#
# VS2026 + CUDA 13.x reste supporté : -G "Visual Studio 18 2026"
#
# CUDA >= 12.8 est la borne du poste Blackwell (sm_120). Sur le poste Ampere
# (sm_86), toute version >= 11.1 conviendrait ; on garde la même pour n'avoir
# qu'une chaîne à connaître.
cmake_minimum_required(VERSION 3.24)

# Doit précéder project() : la détection du compilateur CUDA renseigne elle-même
# CMAKE_CUDA_ARCHITECTURES avec le défaut de nvcc (sm_75), ce qui rendrait un
# garde placé après project() inopérant.
#
# `native` : CMake interroge le GPU présent et ne compile que pour lui. C'est le bon
# défaut tant que le développement alterne entre plusieurs machines — chacune produit
# le binaire qui lui convient, sans réglage à maintenir ni architecture compilée pour
# rien. Le développement se fait aujourd'hui sur RTX 3060 (sm_86) et RTX 50xx (sm_120).
#
# Contrepartie, à connaître : la DLL obtenue ne s'exécute QUE sur la machine qui l'a
# compilée. Ne jamais transporter `extension/bin/bourrasque.dll` ni un paquet de
# `dist/` d'un poste à l'autre — recompiler sur place. Pour un binaire portable
# (distribution, ou paquet unique pour les deux postes), donner la liste explicitement :
#   -DCMAKE_CUDA_ARCHITECTURES="86;120"   (sm_120 exige CUDA >= 12.8)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(bourrasque LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(bourrasque SHARED core/src/mlsmpm.cu core/src/mesher.cu core/src/whitewater.cu)
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
