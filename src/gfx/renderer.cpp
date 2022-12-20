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

    frame_data.resize(FRAMES_IN_FLIGHT);
    for (uint64_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        frame_data[i].present_semaphore = vk_create_semaphore(cx.dev);
        frame_data[i].render_semaphore = vk_create_semaphore(cx.dev);
        frame_data[i].render_fence = vk_create_fence(cx.dev, true);
    }

    FrameContext fcx{cx};
    fcx.begin();

    cx.post_init(fcx);

    pbr_pass.init(fcx);
    composite_pass.init(fcx);
    resolve_pass.init(fcx);
    shadow_pass.init(fcx);
    ssao_pass.init(fcx);
    prepass_pass.init(fcx);

    world.begin(fcx);

    fcx.end();
    std::move(fcx).submit(cx.gfx_queue).get();
}

void Renderer::cleanup() {
    FrameContext fcx{*cx};
    fcx.begin();

    world.end(fcx);

    ui.cleanup(fcx.cx);

    prepass_pass.cleanup(fcx);
    ssao_pass.cleanup(fcx);
    shadow_pass.cleanup(fcx);
    resolve_pass.cleanup(fcx);
    composite_pass.cleanup(fcx);
    pbr_pass.cleanup(fcx);

    for (const auto& frame : frame_data) {
        vkDestroySemaphore(cx->dev, frame.present_semaphore, nullptr);
        vkDestroySemaphore(cx->dev, frame.render_semaphore, nullptr);
        vkDestroyFence(cx->dev, frame.render_fence, nullptr);
    }

    cx->pre_cleanup(fcx);

    fcx.end();
    std::move(fcx).submit(cx->gfx_queue).get();
}

void Renderer::run() {
    time = 0.0;
    curr_time = glfwGetTime();

    while (!glfwWindowShouldClose(cx->window)) {
        glfwPollEvents();
        render();
    }
}

void Renderer::render() {
    auto& frame = current_frame();

    vk_log(vkWaitForFences(cx->dev, 1, &frame.render_fence, true, 1000000000));
    vk_log(vkResetFences(cx->dev, 1, &frame.render_fence));

    uint32_t swap_idx = 0;
    vk_log(vkAcquireNextImageKHR(cx->dev, cx->swapchain, 1000000000, frame.present_semaphore, nullptr, &swap_idx));

    FrameContext fcx{*cx};
    fcx.begin();

    composite_pass.ui = &ui;

    if (ui.begin()) {
        const double new_time = glfwGetTime();
        double frame_time = new_time - curr_time;
        curr_time = new_time;
        while (frame_time > 0.0) {
            const float dt = std::min(frame_time, 1.0 / 60.0);
            world.update(fcx, dt);
            frame_time -= dt;
            time += dt;
        }

        world.ui();
    }

    cx->scene.update(fcx);

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
             static_cast<GFXPass*>(&prepass_pass),
             static_cast<GFXPass*>(&ssao_pass),
         }) {
        pass->add_resources(fcx, graph);
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
    submit.pWaitSemaphores = &frame.present_semaphore;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &frame.render_semaphore;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &fcx.cmd;

    vk_log(vkQueueSubmit(cx->gfx_queue, 1, &submit, frame.render_fence));

    VkPresentInfoKHR present = {};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.pSwapchains = &cx->swapchain;
    present.swapchainCount = 1;
    present.pWaitSemaphores = &frame.render_semaphore;
    present.waitSemaphoreCount = 1;
    present.pImageIndices = &swap_idx;

    vk_log(vkQueuePresentKHR(cx->gfx_queue, &present));

    std::move(fcx).wait(frame.render_fence);
}

Renderer::FrameData& Renderer::current_frame() {
    return frame_data[frame_num % FRAMES_IN_FLIGHT];
}

} // namespace gfx
