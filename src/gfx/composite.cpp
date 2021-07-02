#include "composite.hpp"

#include "render_graph.hpp"
#include "frame_context.hpp"
#include "context.hpp"

namespace gfx {

void CompositePass::init(FrameContext& fcx) {
    load_shader(fcx.cx.shader_cache, "fullscreen.vs", VK_SHADER_STAGE_VERTEX_BIT);
    load_shader(fcx.cx.shader_cache, "composite.fs", VK_SHADER_STAGE_FRAGMENT_BIT);
}

void CompositePass::cleanup(FrameContext& fcx) {
}

void CompositePass::add_resources(FrameContext& fcx, RenderGraph& rg) {
    TextureDesc in_desc;
    in_desc.width = fcx.cx.width;
    in_desc.height = fcx.cx.height;
    in_desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    in_desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    in_desc.samples = VK_SAMPLE_COUNT_1_BIT;

    const Texture in = fcx.cx.rt_cache.get("composite.in", in_desc);

    PassAttachment pa_in;
    pa_in.tex = in;
    pa_in.subresource = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);

    rg.push_attachment({"composite.in"}, pa_in);
}

std::vector<RenderPass> CompositePass::pass(FrameContext& fcx) {
    RenderPass pass;

    pass.width = fcx.cx.width;
    pass.height = fcx.cx.height;
    pass.layers = 1;

    pass.push_color_output({"composite.out"}, vk_clear_color(0.f, 0.f, 0.f, 1.f));
    pass.push_texture_input({"composite.in"});
    pass.set_exec([this](FrameContext& fcx, const RenderGraph& rg, VkRenderPass rp) { render(fcx, rg, rp); });

    return {pass};
}

void CompositePass::render(FrameContext& fcx, const RenderGraph& rg, VkRenderPass rp) {
    const VkViewport viewport = vk_viewport(0.f, 0.f, static_cast<float>(fcx.cx.width), static_cast<float>(fcx.cx.height), 0.f, 1.f);
    const VkRect2D scissor = vk_rect(0, 0, fcx.cx.width, fcx.cx.height);

    DescriptorSetInfo set_info;
    set_info.bind_texture(
        rg.attachment({"composite.in"}).tex, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    if (!fcx.cx.pipeline_cache.contains("composite.pipeline")) {
        VkPipelineRasterizationStateCreateInfo prsci = vk_rasterization_state_create_info(VK_POLYGON_MODE_FILL);
        prsci.cullMode = VK_CULL_MODE_NONE;

        SimplePipelineBuilder builder = SimplePipelineBuilder::begin(fcx.cx.dev, nullptr, fcx.cx.descriptor_cache, fcx.cx.pipeline_cache);
        builder.set_rasterization_state(prsci);
        builder.add_shader(fcx.cx.shader_cache.get("fullscreen.vs"), VK_SHADER_STAGE_VERTEX_BIT);
        builder.add_shader(fcx.cx.shader_cache.get("composite.fs"), VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.add_attachment(vk_color_blend_attachment_state());
        builder.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        builder.set_samples(VK_SAMPLE_COUNT_1_BIT);
        builder.push_desc_set(set_info);

        fcx.cx.pipeline_cache.add("composite.pipeline", builder.info());
    }

    const Pipeline pipeline = fcx.cx.pipeline_cache.get(rp, 0, "composite.pipeline");
    const DescriptorSet set = fcx.cx.descriptor_cache.get_set(key, set_info);

    vkCmdBindPipeline(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    vkCmdBindDescriptorSets(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &set.set, 0, nullptr);

    vkCmdSetViewport(fcx.cmd, 0, 1, &viewport);
    vkCmdSetScissor(fcx.cmd, 0, 1, &scissor);

    vkCmdDraw(fcx.cmd, 3, 1, 0, 0);
}

} // namespace gfx
