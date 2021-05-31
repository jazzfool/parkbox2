#include "shadow.hpp"

#include "render_graph.hpp"
#include "frame_context.hpp"
#include "context.hpp"
#include "renderer.hpp"
#include "helpers.hpp"

#include <spdlog/fmt/fmt.h>
#include <array>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <random>

namespace gfx {

ShadowPass::Uniforms ShadowPass::compute_cascades(Context& cx, glm::vec3 jitter) {
    static constexpr float SPLIT_LAMBDA = 0.95;

    Uniforms out;

    float cascade_splits[ShadowPass::NUM_CASCADES];

    // TODO(jazzfool): get these values from a single source of truth
    float near_clip = 0.1f;
    float far_clip = 100.f;
    float clip_range = far_clip - near_clip;

    float min_z = near_clip;
    float max_z = near_clip + clip_range;

    float range = max_z - min_z;
    float ratio = max_z / min_z;

    // Calculate split depths based on view camera frustum
    // Based on method presented in https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch10.html
    for (uint8_t i = 0; i < ShadowPass::NUM_CASCADES; i++) {
        float p = (i + 1) / static_cast<float>(ShadowPass::NUM_CASCADES);
        float log = min_z * std::pow(ratio, p);
        float uniform = min_z + range * p;
        float d = SPLIT_LAMBDA * (log - uniform) + uniform;
        cascade_splits[i] = (d - near_clip) / clip_range;
    }

    // Calculate orthographic projection matrix for each cascade
    float last_split_dist = 0.f;
    for (uint8_t i = 0; i < ShadowPass::NUM_CASCADES; i++) {
        float split_dist = cascade_splits[i];

        std::array<glm::vec3, 8> frustum_corners = {
            glm::vec3{-1.0f, 1.0f, -1.0f},
            glm::vec3{1.0f, 1.0f, -1.0f},
            glm::vec3{1.0f, -1.0f, -1.0f},
            glm::vec3{-1.0f, -1.0f, -1.0f},
            glm::vec3{-1.0f, 1.0f, 1.0f},
            glm::vec3{1.0f, 1.0f, 1.0f},
            glm::vec3{1.0f, -1.0f, 1.0f},
            glm::vec3{-1.0f, -1.0f, 1.0f},
        };

        // Project frustum corners into world space
        glm::mat4 inv_cam = glm::inverse(cx.renderer->pbr_pass.uniforms.cam_proj);
        for (glm::vec3& corner : frustum_corners) {
            glm::vec4 inv_corner = inv_cam * glm::vec4{corner, 1.0f};
            corner = glm::vec3{inv_corner / inv_corner.w};
        }

        for (uint8_t j = 0; j < 4; j++) {
            glm::vec3 dist = frustum_corners[j + 4] - frustum_corners[j];
            frustum_corners[j + 4] = frustum_corners[j] + (dist * split_dist);
            frustum_corners[j] = frustum_corners[j] + (dist * last_split_dist);
        }

        glm::vec3 frustum_center = glm::vec3{0.f};
        for (glm::vec3 corner : frustum_corners) {
            frustum_center += corner;
            frustum_center /= 8.f;
        }

        float radius = 0.f;
        for (glm::vec3 corner : frustum_corners) {
            float distance = glm::length(corner - frustum_center);
            radius = std::max(radius, distance);
        }
        radius = std::ceil(radius * 16.0f) / 16.0f;

        glm::vec3 max_extents = glm::vec3{radius};
        glm::vec3 min_extents = -max_extents;

        glm::mat4 light_view_matrix =
            glm::lookAt(frustum_center - glm::vec3{cx.renderer->pbr_pass.uniforms.sun_dir} * -min_extents.z, frustum_center, {0.f, 1.f, 0.f}) *
            glm::eulerAngleXYZ(jitter.x, jitter.y, jitter.z);
        glm::mat4 light_ortho_matrix = glm::ortho(min_extents.x, max_extents.x, min_extents.y, max_extents.y, 0.f, max_extents.z - min_extents.z);

        out.views[i] = light_view_matrix;
        out.cascade_splits[i] = (near_clip + split_dist * clip_range) * -1.f;
        out.projs[i] = light_ortho_matrix * light_view_matrix;

        last_split_dist = cascade_splits[i];
    }

    return out;
}

void ShadowPass::init(FrameContext& fcx) {
    load_shader(fcx.cx.shader_cache, "shadow.vs", VK_SHADER_STAGE_VERTEX_BIT);
    load_shader(fcx.cx.shader_cache, "shadow.fs", VK_SHADER_STAGE_FRAGMENT_BIT);
    load_shader(fcx.cx.shader_cache, "fullscreen.vs", VK_SHADER_STAGE_VERTEX_BIT);

    TextureDesc map_desc;
    map_desc.width = DIM;
    map_desc.height = DIM;
    map_desc.layers = NUM_CASCADES;
    map_desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    map_desc.format = VK_FORMAT_R32G32_SFLOAT;
    map_desc.mips = 1;
    map_desc.samples = VK_SAMPLE_COUNT_1_BIT;
    map_desc.depth = 1;
    map_desc.type = VK_IMAGE_TYPE_2D;
    map_desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    map_desc.view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;

    map = create_texture(fcx.cx, map_desc);
    map_2 = create_texture(fcx.cx, map_desc);

    for (uint8_t i = 0; i < NUM_CASCADES; ++i) {
        VkImageViewCreateInfo ivci = {};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = map.image.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.components = vk_no_swizzle();
        ivci.format = map.image.format;
        ivci.subresourceRange = vk_subresource_range(i, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);

        map_views[i] = create_texture(fcx.cx.dev, map.image, ivci);

        ivci.image = map_2.image.image;
        map_2_views[i] = create_texture(fcx.cx.dev, map_2.image, ivci);
    }

    TextureDesc depth_desc;
    depth_desc.width = DIM;
    depth_desc.height = DIM;
    depth_desc.layers = 1;
    depth_desc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_desc.format = VK_FORMAT_D32_SFLOAT;
    depth_desc.mips = 1;
    depth_desc.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_desc.depth = 1;
    depth_desc.type = VK_IMAGE_TYPE_2D;
    depth_desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depth_desc.view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;

    depth = create_texture(fcx.cx, depth_desc);

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = sizeof(Uniforms);
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    ubo = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    prev_ubo = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_TO_GPU, true);

    use_2 = true;
    first = true;
    jitter = 0.f;
    frames = 0;
    sample = 0;
}

void ShadowPass::cleanup(FrameContext& fcx) {
    for (Texture view : map_views) {
        vkDestroyImageView(fcx.cx.dev, view.view, nullptr);
    }

    for (Texture view : map_2_views) {
        vkDestroyImageView(fcx.cx.dev, view.view, nullptr);
    }

    destroy_texture(fcx.cx, map);
    destroy_texture(fcx.cx, map_2);
    destroy_texture(fcx.cx, depth);

    fcx.cx.alloc.destroy(ubo);
    fcx.cx.alloc.destroy(prev_ubo);
}

void ShadowPass::add_resources(RenderGraph& rg) {
    use_2 = !use_2;

    VkImageLayout src_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (first) {
        first = false;
        src_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    PassAttachment map_pa;
    map_pa.tex = use_2 ? map_2 : map;
    map_pa.subresource = vk_subresource_range(0, NUM_CASCADES, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    rg.push_attachment({"shadow.map"}, map_pa);

    PassAttachment src_map_pa;
    src_map_pa.tex = use_2 ? map : map_2;
    src_map_pa.subresource = vk_subresource_range(0, NUM_CASCADES, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    rg.push_attachment({"shadow.map.src"}, src_map_pa);
    rg.push_initial_layout({"shadow.map.src"}, src_layout);

    for (uint8_t i = 0; i < NUM_CASCADES; ++i) {
        PassAttachment map_view_pa;
        map_view_pa.tex = use_2 ? map_2_views[i] : map_views[i];
        map_view_pa.subresource = vk_subresource_range(i, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
        rg.push_attachment({fmt::format("shadow.map.cascade.{}", i)}, map_view_pa);
    }

    PassAttachment depth_pa;
    depth_pa.tex = depth;
    depth_pa.subresource = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_DEPTH_BIT);
    rg.push_attachment({"shadow.depth"}, depth_pa);
}

std::vector<RenderPass> ShadowPass::pass(FrameContext& fcx) {
    static std::random_device rd;
    static std::uniform_real_distribution<float> j{-1.f, 1.f};
    static std::mt19937_64 mt{rd()};

    frames++;

    float x = j(mt);
    float y = j(mt);
    float z = j(mt);

    vk_mapped_write(fcx.cx.alloc, prev_ubo, &prev, sizeof(Uniforms));
    const Uniforms uniforms = compute_cascades(fcx.cx, glm::normalize(glm::vec3{x, y, z}) * 0.025f);
    vk_mapped_write(fcx.cx.alloc, ubo, &uniforms, sizeof(Uniforms));
    prev = uniforms;

    std::vector<RenderPass> passes;

    for (uint8_t i = 0; i < NUM_CASCADES; ++i) {
        RenderPass pass;
        pass.width = DIM;
        pass.height = DIM;
        pass.layers = 1;
        pass.set_depth_stencil({"shadow.depth"}, vk_clear_depth(1.f, 0));
        pass.push_color_output({fmt::format("shadow.map.cascade.{}", i)}, vk_clear_color({0.f, 0.f, 0.f, 0.f}));
        pass.push_texture_input({"shadow.map.src"});
        pass.push_dependent(
            {"shadow.map"}, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        pass.set_exec(std::bind(&ShadowPass::render, this, i, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        passes.push_back(pass);
    }

    return passes;
}

void ShadowPass::render(uint32_t cascade, FrameContext& fcx, const RenderGraph& rg, VkRenderPass pass) {
    const VkViewport viewport = vk_viewport(0.f, 0.f, static_cast<float>(DIM), static_cast<float>(DIM), 0.f, 1.f);
    const VkRect2D scissor = vk_rect(0, 0, DIM, DIM);

    DescriptorSetInfo set_info;
    set_info.bind_buffer(fcx.cx.scene.pass.instance_buffer(), VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set_info.bind_buffer(fcx.cx.scene.pass.instance_indices_buffer(), VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set_info.bind_buffer(ubo, VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    set_info.bind_buffer(prev_ubo, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    set_info.bind_texture(
        rg.attachment({"shadow.map.src"}).tex, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    const DescriptorSet set = fcx.cx.descriptor_cache.get_set(desc_key, set_info);

    struct {
        uint32_t cascade;
        uint32_t frames;
    } push_consts{cascade, frames};

    if (!fcx.cx.pipeline_cache.contains("shadow.pipeline")) {
        VkPipelineRasterizationStateCreateInfo prsci = vk_rasterization_state_create_info(VK_POLYGON_MODE_FILL);
        prsci.depthClampEnable = VK_TRUE;
        // prsci.cullMode = VK_CULL_MODE_FRONT_BIT;
        // prsci.depthBiasEnable = VK_FALSE;

        SimplePipelineBuilder builder = SimplePipelineBuilder::begin(fcx.cx.dev, nullptr, fcx.cx.descriptor_cache, fcx.cx.pipeline_cache);
        builder.add_shader(fcx.cx.shader_cache.get("shadow.vs"), VK_SHADER_STAGE_VERTEX_BIT);
        builder.add_shader(fcx.cx.shader_cache.get("shadow.fs"), VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.set_rasterization_state(prsci);
        builder.add_attachment(vk_color_blend_attachment_state());
        builder.set_depth_stencil_state(vk_depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS));
        builder.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        builder.vertex_input<Vertex>(Vertex::Position);
        builder.set_samples(VK_SAMPLE_COUNT_1_BIT);
        builder.push_desc_set(set_info);
        builder.push_constant(0, sizeof(push_consts), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

        fcx.cx.pipeline_cache.add("shadow.pipeline", builder.info());
    }

    const Pipeline pipeline = fcx.cx.pipeline_cache.get(pass, 0, "shadow.pipeline");

    vkCmdBindPipeline(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    vkCmdBindDescriptorSets(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &set.set, 0, nullptr);

    vkCmdSetViewport(fcx.cmd, 0, 1, &viewport);
    vkCmdSetScissor(fcx.cmd, 0, 1, &scissor);
    vkCmdPushConstants(fcx.cmd, pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push_consts), &push_consts);

    fcx.cx.scene.pass.execute(fcx.cmd, fcx.cx.scene.storage);
}

} // namespace gfx
