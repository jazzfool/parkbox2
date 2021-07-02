#include "prepass.hpp"

#include "frame_context.hpp"
#include "context.hpp"
#include "render_graph.hpp"

namespace gfx {

void PrepassPass::init(FrameContext& fcx) {
    load_shader(fcx.cx.shader_cache, "prepass.vs", VK_SHADER_STAGE_VERTEX_BIT);
    load_shader(fcx.cx.shader_cache, "prepass.fs", VK_SHADER_STAGE_FRAGMENT_BIT);
}

void PrepassPass::cleanup(FrameContext& fcx) {
}

void PrepassPass::add_resources(FrameContext& fcx, RenderGraph& rg) {
    TextureDesc depth_desc;
    depth_desc.width = fcx.cx.width;
    depth_desc.height = fcx.cx.height;
    depth_desc.layers = 1;
    depth_desc.depth = 1;
    depth_desc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_desc.format = VK_FORMAT_D32_SFLOAT;
    depth_desc.mips = 1;
    depth_desc.samples = VK_SAMPLE_COUNT_4_BIT;
    depth_desc.type = VK_IMAGE_TYPE_2D;
    depth_desc.view_type = VK_IMAGE_VIEW_TYPE_2D;
    depth_desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    const Texture depth_msaa = fcx.cx.rt_cache.get("prepass.depth.msaa", depth_desc);

    TextureDesc depth_normal_desc = depth_desc;
    depth_normal_desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    depth_normal_desc.format = VK_FORMAT_R32G32B32A32_SFLOAT; // FIXME(jazzfool): this is WAY too much memory
    depth_normal_desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    const Texture depth_normal_msaa = fcx.cx.rt_cache.get("prepass.depth_normal.msaa", depth_normal_desc);

    depth_normal_desc.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_normal_desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    const Texture depth_normal = fcx.cx.rt_cache.get("prepass.depth_normal", depth_normal_desc);

    PassAttachment depth_pa;
    depth_pa.tex = depth_msaa;
    depth_pa.subresource = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_DEPTH_BIT);
    rg.push_attachment({"prepass.depth.msaa"}, depth_pa);

    PassAttachment depth_normal_msaa_pa;
    depth_normal_msaa_pa.tex = depth_normal_msaa;
    depth_normal_msaa_pa.subresource = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    rg.push_attachment({"prepass.depth_normal.msaa"}, depth_normal_msaa_pa);

    PassAttachment depth_normal_pa;
    depth_normal_pa.tex = depth_normal;
    depth_normal_pa.subresource = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    rg.push_attachment({"prepass.depth_normal"}, depth_normal_pa);
}

std::vector<RenderPass> PrepassPass::pass(FrameContext& fcx) {
    const Texture depth = fcx.cx.rt_cache.get("prepass.depth.msaa", {});

    RenderPass pass;
    pass.width = depth.image.extent.width;
    pass.height = depth.image.extent.height;
    pass.layers = 1;
    pass.push_color_output({"prepass.depth_normal.msaa"}, vk_clear_color(glm::vec4{0.f, 0.f, 0.f, 1.f}));
    pass.push_resolve_output({"prepass.depth_normal"}, vk_clear_color(glm::vec4{0.f}));
    pass.set_depth_stencil({"prepass.depth.msaa"}, vk_clear_depth(1.f, 0));
    pass.set_exec(std::bind(&PrepassPass::render, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    return {pass};
}

void PrepassPass::render(FrameContext& fcx, const RenderGraph& rg, VkRenderPass pass) {
    const VkViewport viewport = vk_viewport(0.f, 0.f, static_cast<float>(fcx.cx.width), static_cast<float>(fcx.cx.height), 0.f, 1.f);
    const VkRect2D scissor = vk_rect(0, 0, fcx.cx.width, fcx.cx.height);

    DescriptorSetInfo set_info;
    set_info.bind_buffer(fcx.cx.scene.pass.instance_buffer(), VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set_info.bind_buffer(fcx.cx.scene.pass.instance_indices_buffer(), VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set_info.bind_buffer(rg.buffer({"pbr.ubo"}).buffer, VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    const DescriptorSet set = fcx.cx.descriptor_cache.get_set(desc_key, set_info);

    if (!fcx.cx.pipeline_cache.contains("prepass.pipeline")) {
        SimplePipelineBuilder builder = SimplePipelineBuilder::begin(fcx.cx.dev, nullptr, fcx.cx.descriptor_cache, fcx.cx.pipeline_cache);
        builder.add_shader(fcx.cx.shader_cache.get("prepass.vs"), VK_SHADER_STAGE_VERTEX_BIT);
        builder.add_shader(fcx.cx.shader_cache.get("prepass.fs"), VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.add_attachment(vk_color_blend_attachment_state());
        builder.set_depth_stencil_state(vk_depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL));
        builder.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        builder.vertex_input<Vertex>(Vertex::Position | Vertex::Normal);
        builder.set_samples(VK_SAMPLE_COUNT_4_BIT);
        builder.push_desc_set(set_info);

        fcx.cx.pipeline_cache.add("prepass.pipeline", builder.info());
    }

    const Pipeline pipeline = fcx.cx.pipeline_cache.get(pass, 0, "prepass.pipeline");

    vkCmdBindPipeline(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    vkCmdBindDescriptorSets(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &set.set, 0, nullptr);

    vkCmdSetViewport(fcx.cmd, 0, 1, &viewport);
    vkCmdSetScissor(fcx.cmd, 0, 1, &scissor);

    fcx.cx.scene.pass.execute(fcx.cmd, fcx.cx.scene.storage);
}

} // namespace gfx
