#include "context.hpp"

#include "vk_helpers.hpp"

#include <VkBootstrap.h>
#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>

namespace gfx {

void glfw_mouse_move(GLFWwindow* window, double x, double y) {
    Context* cx = reinterpret_cast<Context*>(glfwGetWindowUserPointer(window));
    cx->on_mouse_move(x, y);
}

void glfw_scroll(GLFWwindow* window, double x, double y) {
    Context* cx = reinterpret_cast<Context*>(glfwGetWindowUserPointer(window));
    cx->on_scroll(x, y);
}

bool Context::init(GLFWwindow* window) {
    this->window = window;

    glfwSetWindowUserPointer(window, this);
    glfwSetCursorPosCallback(window, glfw_mouse_move);
    glfwSetScrollCallback(window, glfw_scroll);

    int fb_width = 0, fb_height = 0;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);

    width = fb_width;
    height = fb_height;

    vkb::InstanceBuilder vkb_instance_builder;
    vkb_instance_builder.set_app_name("Parkbox");
    vkb_instance_builder.require_api_version(1, 2);
    vkb_instance_builder.use_default_debug_messenger();
    vkb_instance_builder.set_debug_callback([](VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity, VkDebugUtilsMessageTypeFlagsEXT msg_type,
                                                const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data) -> VkBool32 {
        auto ms = vkb::to_string_message_severity(msg_severity);
        auto mt = vkb::to_string_message_type(msg_type);
        spdlog::warn("Vulkan {} {}: {}", mt, ms, callback_data->pMessage);
        return VK_FALSE;
    });
    vkb_instance_builder.request_validation_layers(
#if defined(NDEBUG)
        false
#else
        true
#endif
    );
    vkb_instance_builder.enable_layer("VK_LAYER_LUNARG_monitor");

    vkb::detail::Result<vkb::Instance> vkb_instance_result = vkb_instance_builder.build();

    if (!vkb_instance_result.has_value()) {
        spdlog::error("failed to init vkb instance: {}", vkb_instance_result.error().value());
        return false;
    }

    instance = vkb_instance_result.value().instance;
    debug_messenger = vkb_instance_result.value().debug_messenger;

    volkLoadInstance(instance);

    vk_log(glfwCreateWindowSurface(instance, window, nullptr, &surface));

    VkPhysicalDeviceFeatures features = {};
    features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    features.multiDrawIndirect = VK_TRUE;
    features.samplerAnisotropy = VK_TRUE;
    features.depthClamp = VK_TRUE;
    features.fragmentStoresAndAtomics = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features_12 = {};
    features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features_12.runtimeDescriptorArray = true;

    vkb::PhysicalDeviceSelector vkb_physdev_selector{vkb_instance_result.value()};
    vkb_physdev_selector.set_surface(surface);
    vkb_physdev_selector.require_present();
    vkb_physdev_selector.set_minimum_version(1, 2);
    vkb_physdev_selector.set_required_features(features);
    vkb_physdev_selector.set_required_features_12(features_12);
    vkb_physdev_selector.prefer_gpu_device_type(vkb::PreferredDeviceType::discrete);

    vkb::detail::Result<vkb::PhysicalDevice> vkb_physdev_result = vkb_physdev_selector.select();

    if (!vkb_physdev_result.has_value()) {
        spdlog::error("failed to select device via vkb: {}", vkb_physdev_result.error().value());
        return false;
    }

    phys_dev = vkb_physdev_result->physical_device;

    vkb::DeviceBuilder vkb_device_builder{vkb_physdev_result.value()};
    vkb::detail::Result<vkb::Device> vkb_device = vkb_device_builder.build();

    dev = vkb_device->device;

    gfx_queue = *vkb_device->get_queue(vkb::QueueType::graphics);
    transfer_queue = *vkb_device->get_queue(vkb::QueueType::transfer);
    present_queue = *vkb_device->get_queue(vkb::QueueType::present);
    compute_queue = *vkb_device->get_queue(vkb::QueueType::compute);

    gfx_queue_idx = *vkb_device->get_queue_index(vkb::QueueType::graphics);
    transfer_queue_idx = *vkb_device->get_queue_index(vkb::QueueType::transfer);
    present_queue_idx = *vkb_device->get_queue_index(vkb::QueueType::present);
    compute_queue_idx = *vkb_device->get_queue_index(vkb::QueueType::compute);

    int32_t width, height;
    glfwGetWindowSize(window, &width, &height);

    vkb::SwapchainBuilder vkb_swapchain_builder{*vkb_device};
    vkb_swapchain_builder.set_desired_extent(width, height);
    vkb_swapchain_builder.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR);
    vkb::detail::Result<vkb::Swapchain> vkb_swapchain = vkb_swapchain_builder.build();

    swapchain = vkb_swapchain->swapchain;
    swapchain_images = *vkb_swapchain->get_images();
    swapchain_views = *vkb_swapchain->get_image_views();
    swapchain_format = vkb_swapchain->image_format;

    alloc.init(*this);
    frame_pool.init(*this);
    shader_cache.init(*this);
    descriptor_cache.init(dev);
    pipeline_cache.init(dev, descriptor_cache);
    sampler_cache.init(dev);
    rg_cache.init(dev);

    return true;
}

void Context::post_init(FrameContext& fcx) {
    scene.pass.init(fcx);
    scene.storage.init(fcx);
}

void Context::pre_cleanup(FrameContext& fcx) {
    scene.pass.cleanup(fcx);
    scene.storage.cleanup(fcx);
}

void Context::cleanup() {
    rg_cache.cleanup();
    sampler_cache.cleanup();
    pipeline_cache.cleanup();
    descriptor_cache.cleanup();
    shader_cache.cleanup();
    frame_pool.cleanup();
    alloc.cleanup();

    vkDestroySwapchainKHR(dev, swapchain, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
    vkDestroyInstance(instance, nullptr);
}

} // namespace gfx