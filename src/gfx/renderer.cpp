#include "renderer.hpp"

#include "context.hpp"
#include "frame_context.hpp"
#include "render_graph.hpp"

#include <GLFW/glfw3.h>

#include <chrono>
#include <spdlog/spdlog.h>

namespace gfx {

void Renderer::init(Context& cx) {
    this->cx = &cx;

    present_semaphore = vk_create_semaphore(cx.dev);
    render_semaphore = vk_create_semaphore(cx.dev);
    render_fence = vk_create_fence(cx.dev, true);

    FrameContext fcx{cx};
    fcx.begin();

    cx.post_init(fcx);

    pbr_pass.init(fcx);
    composite_pass.init(fcx);
    resolve_pass.init(fcx);
    shadow_pass.init(fcx);

    world.begin(fcx);

    fcx.end();
    std::move(fcx).submit(cx.gfx_queue).get();
}

void Renderer::cleanup() {
    FrameContext fcx{*cx};
    fcx.begin();

    world.end(fcx);

    shadow_pass.cleanup(fcx);
    resolve_pass.cleanup(fcx);
    composite_pass.cleanup(fcx);
    pbr_pass.cleanup(fcx);

    vkDestroySemaphore(cx->dev, present_semaphore, nullptr);
    vkDestroySemaphore(cx->dev, render_semaphore, nullptr);
    vkDestroyFence(cx->dev, render_fence, nullptr);

    cx->pre_cleanup(fcx);

    fcx.end();
    std::move(fcx).submit(cx->gfx_queue).get();
}

void Renderer::run() {
    while (!glfwWindowShouldClose(cx->window)) {
        glfwPollEvents();
        render();
    }
}

void Renderer::render() {
    vk_log(vkWaitForFences(cx->dev, 1, &render_fence, true, 1000000000));
    vk_log(vkResetFences(cx->dev, 1, &render_fence));

    uint32_t swap_idx = 0;
    vk_log(vkAcquireNextImageKHR(cx->dev, cx->swapchain, 1000000000, present_semaphore, nullptr, &swap_idx));

    FrameContext fcx{*cx};
    fcx.begin();

    world.update(fcx);

    cx->scene.storage.update(fcx);
    cx->scene.pass.prepare(fcx);

    RenderGraph graph;

    PassAttachment attachment = {};
    attachment.tex.image.image = cx->swapchain_images[swap_idx];
    attachment.tex.image.format = cx->swapchain_format;
    attachment.tex.image.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.tex.view = cx->swapchain_views[swap_idx];
    attachment.subresource = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    graph.push_attachment({"composite.out"}, attachment);

    for (GFXPass* pass : {
             static_cast<GFXPass*>(&pbr_pass),
             static_cast<GFXPass*>(&composite_pass),
             static_cast<GFXPass*>(&resolve_pass),
             static_cast<GFXPass*>(&shadow_pass),
         }) {
        pass->add_resources(graph);
        for (RenderPass p : pass->pass(fcx))
            graph.push_pass(p);
    }

    graph.set_output({"composite.out"}, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    graph.exec(fcx, cx->rg_cache);

    fcx.end();

    VkSubmitInfo submit = {};

    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pWaitDstStageMask = &wait_stage;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &present_semaphore;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &render_semaphore;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &fcx.cmd;

    vk_log(vkQueueSubmit(cx->gfx_queue, 1, &submit, render_fence));

    VkPresentInfoKHR present = {};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.pSwapchains = &cx->swapchain;
    present.swapchainCount = 1;
    present.pWaitSemaphores = &render_semaphore;
    present.waitSemaphoreCount = 1;
    present.pImageIndices = &swap_idx;

    vk_log(vkQueuePresentKHR(cx->gfx_queue, &present));

    std::move(fcx).wait(render_fence);
}

} // namespace gfx
