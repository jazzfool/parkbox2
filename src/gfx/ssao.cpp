#include "ssao.hpp"

#include "render_graph.hpp"
#include "frame_context.hpp"
#include "context.hpp"
#include "renderer.hpp"

#include <glm/ext/scalar_constants.hpp>
#include <random>

namespace gfx {

void SSAOPass::init(FrameContext& fcx) {
    load_shader(fcx.cx.shader_cache, "fullscreen.vs", VK_SHADER_STAGE_VERTEX_BIT);
    load_shader(fcx.cx.shader_cache, "hbao.fs", VK_SHADER_STAGE_FRAGMENT_BIT);

    fcx.cx.on_resize.connect_delegate(delegate<&SSAOPass::resize>(this));

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = sizeof(glm::mat4) * 3;
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    ubo = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_TO_GPU, true);

    use_a = true;
    first = true;
}

void SSAOPass::cleanup(FrameContext& fcx) {
    fcx.cx.alloc.destroy(ubo);
}

void SSAOPass::add_resources(FrameContext& fcx, RenderGraph& rg) {
    TextureDesc desc;
    desc.width = static_cast<uint32_t>(static_cast<float>(fcx.cx.width) * RESOLUTION);
    desc.height = static_cast<uint32_t>(static_cast<float>(fcx.cx.height) * RESOLUTION);
    desc.layers = 1;
    desc.depth = 1;
    desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    desc.format = VK_FORMAT_R8_UNORM;
    desc.mips = 1;
    desc.samples = VK_SAMPLE_COUNT_1_BIT;
    desc.type = VK_IMAGE_TYPE_2D;
    desc.view_type = VK_IMAGE_VIEW_TYPE_2D;
    desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    const Texture out_a = fcx.cx.rt_cache.get("gtao.out.a", desc);
    const Texture out_b = fcx.cx.rt_cache.get("gtao.out.b", desc);

    PassAttachment pa;
    pa.tex = use_a ? out_a : out_b;
    pa.subresource = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    rg.push_attachment({"gtao.out"}, pa);

    PassAttachment prev_pa;
    prev_pa.tex = use_a ? out_b : out_a;
    prev_pa.subresource = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    rg.push_attachment({"gtao.prev"}, prev_pa);

    if (!first) {
        rg.push_initial_layout({"gtao.prev"}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        first = false;
    }

    use_a = !use_a;
}

std::vector<RenderPass> SSAOPass::pass(FrameContext& fcx) {
    const glm::mat4 mats[3] = {glm::inverse(fcx.cx.renderer->pbr_pass.uniforms.cam_proj), fcx.cx.renderer->pbr_pass.uniforms.cam_view, prev_vp};
    vk_mapped_write(fcx.cx.alloc, ubo, &mats[0], sizeof(glm::mat4) * 3);
    prev_vp = fcx.cx.renderer->pbr_pass.uniforms.cam_proj;

    const Texture out_a = fcx.cx.rt_cache.get("gtao.out.a", {});

    RenderPass pass;
    pass.width = out_a.image.extent.width;
    pass.height = out_a.image.extent.height;
    pass.layers = 1;
    pass.push_color_output({"gtao.out"}, vk_clear_color({0.f, 0.f, 0.f, 0.f}));
    pass.push_texture_input({"gtao.prev"});
    pass.push_texture_input({"prepass.depth_normal"});
    pass.set_exec(std::bind(&SSAOPass::render, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    return {pass};
}

void SSAOPass::render(FrameContext& fcx, const RenderGraph& rg, VkRenderPass pass) {
    const Texture out_a = fcx.cx.rt_cache.get("gtao.out.a", {});

    const VkViewport viewport = vk_viewport(0.f, 0.f, static_cast<float>(out_a.image.extent.width), static_cast<float>(out_a.image.extent.height), 0.f, 1.f);
    const VkRect2D scissor = vk_rect(0, 0, out_a.image.extent.width, out_a.image.extent.height);

    DescriptorSetInfo set_info;
    set_info.bind_texture(
        rg.attachment({"prepass.depth_normal"}).tex, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    set_info.bind_texture(
        rg.attachment({"gtao.prev"}).tex, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    set_info.bind_buffer(ubo, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    const DescriptorSet set = fcx.cx.descriptor_cache.get_set(desc_key, set_info);

    if (!fcx.cx.pipeline_cache.contains("gtao.pipeline")) {
        VkPipelineRasterizationStateCreateInfo prsci = vk_rasterization_state_create_info(VK_POLYGON_MODE_FILL);
        prsci.cullMode = VK_CULL_MODE_NONE;

        SimplePipelineBuilder builder = SimplePipelineBuilder::begin(fcx.cx.dev, nullptr, fcx.cx.descriptor_cache, fcx.cx.pipeline_cache);
        builder.add_shader(fcx.cx.shader_cache.get("fullscreen.vs"), VK_SHADER_STAGE_VERTEX_BIT);
        builder.add_shader(fcx.cx.shader_cache.get("hbao.fs"), VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.set_rasterization_state(prsci);
        builder.add_attachment(vk_color_blend_attachment_state());
        builder.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        builder.set_samples(VK_SAMPLE_COUNT_1_BIT);
        builder.push_desc_set(set_info);
        builder.push_constant(0, sizeof(float), VK_SHADER_STAGE_FRAGMENT_BIT);

        fcx.cx.pipeline_cache.add("gtao.pipeline", builder.info());
    }

    const Pipeline pipeline = fcx.cx.pipeline_cache.get(pass, 0, "gtao.pipeline");

    vkCmdBindPipeline(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    vkCmdBindDescriptorSets(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &set.set, 0, nullptr);

    static std::uniform_real_distribution<float> dist{0.f, 2.f * glm::pi<float>()};
    static std::mt19937_64 mt{std::random_device{}()};
    const float jitter = dist(mt);

    vkCmdPushConstants(fcx.cmd, pipeline.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &jitter);

    vkCmdSetViewport(fcx.cmd, 0, 1, &viewport);
    vkCmdSetScissor(fcx.cmd, 0, 1, &scissor);

    vkCmdDraw(fcx.cmd, 3, 1, 0, 0);
}

void SSAOPass::resize(int32_t, int32_t) {
    first = true;
}

} // namespace gfx
