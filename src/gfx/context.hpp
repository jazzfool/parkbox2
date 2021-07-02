#pragma once

#include "allocator.hpp"
#include "cmd_pool.hpp"
#include "shader_cache.hpp"
#include "pipeline_cache.hpp"
#include "descriptor_cache.hpp"
#include "sampler_cache.hpp"
#include "scene.hpp"
#include "signal.hpp"
#include "render_graph.hpp"
#include "rt_cache.hpp"

#include <volk.h>
#include <vk_mem_alloc.h>
#include <entt/entt.hpp>
#include <VkBootstrap.h>

struct GLFWwindow;

namespace gfx {

class Renderer;

struct Context {
    bool init(GLFWwindow* window);
    void sc_init(int32_t width, int32_t height);
    void post_init(FrameContext& fcx);
    void pre_cleanup(FrameContext& fcx);
    void sc_cleanup();
    void cleanup();

    GLFWwindow* window;
    uint32_t width;
    uint32_t height;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkPhysicalDevice phys_dev;
    VkDevice dev;
    vkb::Device vkb_dev;

    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_views;
    VkFormat swapchain_format;

    VkQueue gfx_queue;
    VkQueue transfer_queue;
    VkQueue present_queue;
    VkQueue compute_queue;

    uint32_t gfx_queue_idx;
    uint32_t transfer_queue_idx;
    uint32_t present_queue_idx;
    uint32_t compute_queue_idx;

    Allocator alloc;
    CommandPool frame_pool;
    ShaderCache shader_cache;
    DescriptorCache descriptor_cache;
    PipelineCache pipeline_cache;
    SamplerCache sampler_cache;
    RenderGraphCache rg_cache;
    RenderTargetCache rt_cache;

    Scene scene;

    Signal<double, double> on_mouse_move;
    Signal<double, double> on_scroll;
    Signal<int32_t, int32_t> on_resize;

    Renderer* renderer; // TODO(jazzfool): this pointer is ugly to have here, remove it at some point
};

} // namespace gfx