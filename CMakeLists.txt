cmake_minimum_required(VERSION 3.6)

project(parkbox CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE
    src/main.cpp
    src/impl.cpp

    src/gfx/context.cpp
    src/gfx/vk_helpers.cpp
    src/gfx/deletion_queue.cpp
    src/gfx/allocator.cpp
    src/gfx/frame_context.cpp
    src/gfx/renderer.cpp
    src/gfx/pbr.cpp
    src/gfx/render_graph.cpp
    src/gfx/cmd_pool.cpp
    src/gfx/shader_cache.cpp
    src/gfx/descriptor_cache.cpp
    src/gfx/pipeline_cache.cpp
    src/gfx/mesh.cpp
    src/gfx/composite.cpp
    src/gfx/sampler_cache.cpp
    src/gfx/indirect.cpp
    src/gfx/resolve.cpp
    src/gfx/shadow.cpp

    src/world/world.cpp
    src/world/mesh.cpp
    src/world/passive.cpp
    src/world/camera.cpp
)

add_executable(parkbox ${SOURCE})

target_compile_definitions(parkbox PRIVATE PK_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res")

add_library(shaderc UNKNOWN IMPORTED)
if(WIN32)
	# use the version in the SDK	
	set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION $ENV{VULKAN_SDK}/Lib/shaderc_shared.lib)
	set_property(TARGET shaderc PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{VULKAN_SDK}/Include)
else()
	# TODO
endif()

add_library(shaderc_util UNKNOWN IMPORTED)
if(WIN32)
	# use the version in the SDK	
	set_target_properties(shaderc_util PROPERTIES IMPORTED_LOCATION $ENV{VULKAN_SDK}/Lib/shaderc_util.lib)
	set_property(TARGET shaderc_util PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{VULKAN_SDK}/Include)
else()
	# TODO
endif()

find_package(Vulkan REQUIRED)

if (WIN32)
	set(CMAKE_AR lib.exe)
	set(CMAKE_ASM_MASM_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <LINK_FLAGS> <OBJECTS>")
endif()

set(FTL_BUILD_TESTS OFF CACHE BOOL "Build FiberTaskingLib tests")
set(FTL_BUILD_BENCHMARKS OFF CACHE BOOL "Build FiberTaskingLib benchmarks")
set(FTL_BUILD_EXAMPLES OFF CACHE BOOL "Build FiberTaskingLib examples")
set(FTL_CPP_17 ON CACHE BOOL "Enable C++17 features")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/glfw)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/volk)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/vkb)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/spdlog)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/entt)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/tinyobjloader)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/ftl)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/glm)

target_compile_definitions(parkbox PRIVATE VK_NO_PROTOTYPES GLM_FORCE_DEPTH_ZERO_TO_ONE)

target_link_libraries(parkbox PRIVATE glfw volk vk-bootstrap spdlog shaderc shaderc_util EnTT tinyobjloader ftl glm)

target_include_directories(parkbox PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/VMA
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/span
)
