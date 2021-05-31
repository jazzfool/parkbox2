#include "resolve.hpp"

#include "render_graph.hpp"
#include "frame_context.hpp"
#include "context.hpp"

namespace gfx {

void ResolvePass::init(FrameContext& fcx) {
    load_shader(fcx.cx.shader_cache, "resolve.fs", VK_SHADER_STAGE_FRAGMENT_BIT);
}

void ResolvePass::cleanup(FrameContext& fcx) {
}

void ResolvePass::add_resources(RenderGraph& rg) {
}

std::vector<RenderPass> ResolvePass::pass(FrameContext& fcx) {
    RenderPass pass;

    pass.width = fcx.cx.width;
    pass.height = fcx.cx.height;
    pass.layers = 1;

    pass.push_texture_input({"pbr.out"});
    pass.push_color_output({"composite.in"}, vk_clear_color({0.f, 0.f, 0.f, 1.f}));

    pass.set_exec([this](FrameContext& fcx, const RenderGraph& rg, VkRenderPass rp) { render(fcx, rg, rp); });

    return {pass};
}

void ResolvePass::render(FrameContext& fcx, const RenderGraph& rg, VkRenderPass rp) {
    const VkViewport viewport = vk_viewport(0.f, 0.f, static_cast<float>(fcx.cx.width), static_cast<float>(fcx.cx.height), 0.f, 1.f);
    const VkRect2D scissor = vk_rect(0, 0, fcx.cx.width, fcx.cx.height);

    DescriptorSetInfo set_info;
    set_info.bind_texture(
        rg.attachment({"pbr.out"}).tex, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    if (!fcx.cx.pipeline_cache.contains("resolve.pipeline")) {
        VkPipelineRasterizationStateCreateInfo prsci = vk_rasterization_state_create_info(VK_POLYGON_MODE_FILL);
        prsci.cullMode = VK_CULL_MODE_NONE;

        SimplePipelineBuilder builder = SimplePipelineBuilder::begin(fcx.cx.dev, nullptr, fcx.cx.descriptor_cache, fcx.cx.pipeline_cache);
        builder.set_rasterization_state(prsci);
        builder.add_shader(fcx.cx.shader_cache.get("fullscreen.vs"), VK_SHADER_STAGE_VERTEX_BIT);
        builder.add_shader(fcx.cx.shader_cache.get("resolve.fs"), VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.add_attachment(vk_color_blend_attachment_state());
        builder.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        builder.set_samples(VK_SAMPLE_COUNT_1_BIT);
        builder.push_constant(0, sizeof(glm::vec2), VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.push_desc_set(set_info);

        fcx.cx.pipeline_cache.add("resolve.pipeline", builder.info());
    }

    const Pipeline pipeline = fcx.cx.pipeline_cache.get(rp, 0, "resolve.pipeline");
    const DescriptorSet set = fcx.cx.descriptor_cache.get_set(key, set_info);

    const glm::vec2 dims = {static_cast<float>(fcx.cx.width), static_cast<float>(fcx.cx.height)};

    vkCmdBindPipeline(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    vkCmdBindDescriptorSets(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &set.set, 0, nullptr);
    vkCmdPushConstants(fcx.cmd, pipeline.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec2), &dims);

    vkCmdSetViewport(fcx.cmd, 0, 1, &viewport);
    vkCmdSetScissor(fcx.cmd, 0, 1, &scissor);

    vkCmdDraw(fcx.cmd, 3, 1, 0, 0);
}

} // namespace gfx
