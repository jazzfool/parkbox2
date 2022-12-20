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
    src/gfx/ssao.cpp
    src/gfx/prepass.cpp
    src/gfx/rt_cache.cpp
    src/gfx/ui.cpp
    src/gfx/scene.cpp
    src/gfx/material.cpp

    src/world/world.cpp
    src/world/mesh.cpp
    src/world/camera.cpp
    src/world/meshlib.cpp
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

set(FTL_BUILD_TESTS OFF)
set(FTL_BUILD_BENCHMARKS OFF)
set(FTL_BUILD_EXAMPLES OFF)
set(FTL_CPP_17 ON)

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/glfw)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/volk)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/vkb)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/spdlog)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/entt)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/tinyobjloader)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/ftl)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/glm)

find_package(Vulkan REQUIRED)

add_library(imgui
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/imgui.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/imgui.cpp"

    "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/imgui_demo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/imgui_draw.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/imgui_widgets.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/imgui_tables.cpp"

    "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/backends/imgui_impl_vulkan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/backends/imgui_impl_glfw.cpp"
)

target_include_directories(imgui
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui"
        "${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/backends"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/ext/glfw/include"
)

target_link_libraries(imgui PRIVATE Vulkan::Vulkan)
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_VULKAN_NO_PROTOTYPES)

target_compile_definitions(parkbox PRIVATE VK_NO_PROTOTYPES GLM_FORCE_DEPTH_ZERO_TO_ONE)

target_link_libraries(parkbox PRIVATE glfw volk vk-bootstrap spdlog shaderc shaderc_util EnTT tinyobjloader ftl glm imgui)

target_include_directories(parkbox PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/VMA
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/span
)
