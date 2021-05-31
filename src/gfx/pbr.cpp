#include "pbr.hpp"

#include "render_graph.hpp"
#include "frame_context.hpp"
#include "context.hpp"
#include "pipeline_cache.hpp"
#include "mesh.hpp"
#include "helpers.hpp"
#include "renderer.hpp"

#include <ftl/task_scheduler.h>
#include <ftl/task_counter.h>
#include <spdlog/fmt/fmt.h>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>

namespace gfx {

inline glm::vec2 hammersley(uint32_t i, float samples) {
    uint32_t bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    return glm::vec2{(float)i / samples, (float)bits / std::exp2f(32.f)};
}

inline float gdfg(float NoV, float NoL, float a) {
    const float a2 = a * a;
    const float ggxl = NoV * std::sqrt((-NoL * a2 + NoL) * NoL + a2);
    const float ggxv = NoL * std::sqrt((-NoV * a2 + NoV) * NoV + a2);
    return (2.f * NoL) / (ggxv + ggxl);
}

inline glm::vec3 importance_sample_ggx(glm::vec2 Xi, glm::vec3 N, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float phi = 2.0 * PI * Xi.x;
    float cos_theta = std::sqrt((1.0 - Xi.y) / (1.0 + (alpha2 - 1.0) * Xi.y));
    float sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

    glm::vec3 H;
    H.x = std::cos(phi) * sin_theta;
    H.y = std::sin(phi) * sin_theta;
    H.z = cos_theta;

    glm::vec3 up = std::abs(N.z) < 0.999 ? glm::vec3(0.0, 0.0, 1.0) : glm::vec3(1.0, 0.0, 0.0);
    glm::vec3 tangent = glm::normalize(glm::cross(up, N));
    glm::vec3 bitangent = glm::cross(N, tangent);

    glm::vec3 sample_vec = tangent * H.x + bitangent * H.y + N * H.z;
    return glm::normalize(sample_vec);
}

inline glm::vec2 dfg(float NoV, float a, bool ibl) {
    static constexpr uint32_t SAMPLE_COUNT = 256;

    const glm::vec3 N = {0.f, 0.f, 1.f};

    glm::vec3 V;
    V.x = std::sqrt(1.0f - NoV * NoV);
    V.y = 0.0f;
    V.z = NoV;

    glm::vec2 r{0.f, 0.f};
    for (uint32_t i = 0; i < SAMPLE_COUNT; i++) {
        const glm::vec2 Xi = hammersley(i, SAMPLE_COUNT);
        const glm::vec3 H = importance_sample_ggx(Xi, N, a);
        const glm::vec3 L = H * (2.0f * glm::dot(V, H)) - V;

        const float VoH = saturate(glm::dot(V, H));
        const float NoL = saturate(L.z);
        const float NoH = saturate(H.z);

        if (NoL > 0.0f) {
            float G = gdfg(NoV, NoL, a);
            float Gv = G * VoH / NoH;
            float Fc = std::pow(1 - VoH, 5.0f);
            if (ibl) {
                r.x += Gv * (1 - Fc);
                r.y += Gv * Fc;
            } else {
                r.x += Gv * Fc;
                r.y += Gv;
            }
        }
    }

    return r * (1.0f / (float)SAMPLE_COUNT);
}

// First vec is for the energy conserving and multiscattering DFG LUT, second vec is for IBL LUT.
inline std::array<std::vector<glm::vec2>, 2> integrate_dfg(ftl::TaskScheduler& scheduler, uint32_t dim, uint32_t rows_per_group) {
    assert(dim % rows_per_group == 0);

    struct Task final {
        uint32_t dim;
        uint32_t y;
        uint32_t count;
        std::vector<glm::vec2> ec_rows;
        std::vector<glm::vec2> ibl_rows;
    };

    static constexpr auto group = [](ftl::TaskScheduler* scheduler, void* arg) {
        Task* task = (Task*)arg;
        task->ec_rows.reserve(task->dim * task->count);
        task->ibl_rows.reserve(task->dim * task->count);
        for (uint32_t y = 0; y < task->count; ++y) {
            for (uint32_t x = 0; x < task->dim; ++x) {
                task->ec_rows.push_back(
                    dfg(static_cast<float>(x) / static_cast<float>(task->dim), static_cast<float>(y + task->y) / static_cast<float>(task->dim), false));
                task->ibl_rows.push_back(
                    dfg(static_cast<float>(x) / static_cast<float>(task->dim), static_cast<float>(y + task->y) / static_cast<float>(task->dim), true));
            }
        }
    };

    std::vector<glm::vec2> ec_img;
    std::vector<glm::vec2> ibl_img;
    ec_img.reserve(dim * dim);
    ibl_img.reserve(dim * dim);

    std::vector<ftl::Task> tasks;
    std::vector<Task> task_data;

    tasks.resize(dim / rows_per_group);
    task_data.resize(dim / rows_per_group);

    for (uint32_t i = 0; i < (dim / rows_per_group); ++i) {
        Task* data = &task_data[i];
        data->dim = dim;
        data->y = i * rows_per_group;
        data->count = rows_per_group;
        tasks[i] = {group, data};
    }

    ftl::TaskCounter counter{&scheduler};
    scheduler.AddTasks(dim / rows_per_group, tasks.data(), ftl::TaskPriority::High, &counter);

    scheduler.WaitForCounter(&counter, true);

    for (uint32_t i = 0; i < (dim / rows_per_group); ++i) {
        ec_img.insert(ec_img.end(), task_data[i].ec_rows.begin(), task_data[i].ec_rows.end());
        ibl_img.insert(ibl_img.end(), task_data[i].ibl_rows.begin(), task_data[i].ibl_rows.end());
    }

    return {ec_img, ibl_img};
}

void PBRGraphicsPass::init(FrameContext& fcx) {
    load_shader(fcx.cx.shader_cache, "pbr.vs", VK_SHADER_STAGE_VERTEX_BIT);
    load_shader(fcx.cx.shader_cache, "pbr.fs", VK_SHADER_STAGE_FRAGMENT_BIT);
    load_shader(fcx.cx.shader_cache, "cubemap.vs", VK_SHADER_STAGE_VERTEX_BIT);
    load_shader(fcx.cx.shader_cache, "equirectangular_to_cubemap.fs", VK_SHADER_STAGE_FRAGMENT_BIT);
    load_shader(fcx.cx.shader_cache, "prefilter.comp", VK_SHADER_STAGE_COMPUTE_BIT);
    // load_shader(fcx.cx.shader_cache, "irradiance.comp", VK_SHADER_STAGE_COMPUTE_BIT);

    TextureDesc desc;
    desc.width = fcx.cx.width;
    desc.height = fcx.cx.height;
    desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    desc.samples = VK_SAMPLE_COUNT_4_BIT;

    out = create_texture(fcx.cx, desc);

    TextureDesc depth_desc;
    depth_desc.width = fcx.cx.width;
    depth_desc.height = fcx.cx.height;
    depth_desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depth_desc.format = VK_FORMAT_D32_SFLOAT;
    depth_desc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_desc.samples = desc.samples;

    depth = create_texture(fcx.cx, depth_desc);

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = sizeof(SceneUniforms);
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    ubo = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_TO_GPU, true);

    TextureDesc dfg_lut_desc;
    dfg_lut_desc.width = 256;
    dfg_lut_desc.height = dfg_lut_desc.width;
    dfg_lut_desc.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    dfg_lut_desc.format = VK_FORMAT_R32G32_SFLOAT;

    ec_dfg_lut = create_texture(fcx.cx, dfg_lut_desc);
    ibl_dfg_lut = create_texture(fcx.cx, dfg_lut_desc);

    VkBufferCreateInfo dfg_staging_bci = {};
    dfg_staging_bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    dfg_staging_bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    dfg_staging_bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    dfg_staging_bci.size = 2 * sizeof(float) * 256 * 256;

    Buffer ec_dfg_staging = fcx.cx.alloc.create_buffer(dfg_staging_bci, VMA_MEMORY_USAGE_CPU_ONLY, true);
    fcx.bind(ec_dfg_staging);

    Buffer ibl_dfg_staging = fcx.cx.alloc.create_buffer(dfg_staging_bci, VMA_MEMORY_USAGE_CPU_ONLY, true);
    fcx.bind(ibl_dfg_staging);

    std::vector<glm::vec2> ec_dfg_lut_data;
    std::vector<glm::vec2> ibl_dfg_lut_data;

    {
        ftl::TaskScheduler sched;
        sched.Init();

        const auto out = integrate_dfg(sched, dfg_lut_desc.width, 1);
        ec_dfg_lut_data = out[0];
        ibl_dfg_lut_data = out[1];
    }

    vk_mapped_write(fcx.cx.alloc, ec_dfg_staging, ec_dfg_lut_data.data(), sizeof(glm::vec2) * ec_dfg_lut_data.size());
    vk_mapped_write(fcx.cx.alloc, ibl_dfg_staging, ibl_dfg_lut_data.data(), sizeof(glm::vec2) * ibl_dfg_lut_data.size());

    fcx.copy_to_image(ec_dfg_staging, ec_dfg_lut.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sizeof(float) * 2,
        vk_subresource_layers(0, 1, 0, VK_IMAGE_ASPECT_COLOR_BIT));
    fcx.copy_to_image(ibl_dfg_staging, ibl_dfg_lut.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sizeof(float) * 2,
        vk_subresource_layers(0, 1, 0, VK_IMAGE_ASPECT_COLOR_BIT));

    const std::vector<uint8_t> hdr_buf = read_binary(fmt::format("{}/textures/{}", PK_RESOURCE_DIR, "tiergarten_2k.hdr"));
    const std::vector<uint8_t> irrad_buf = read_binary(fmt::format("{}/textures/{}", PK_RESOURCE_DIR, "tiergarten_2k_irrad.hdr"));

    ImageLoadInfo hdr_load;
    hdr_load.loadf = true;
    hdr_load.bytes_per_pixel = sizeof(float) * 4;
    hdr_load.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    hdr_load.data = hdr_buf.data();
    hdr_load.data_size = hdr_buf.size();
    hdr_load.generate_mipmaps = false;

    Image hdr_img = load_image(fcx, hdr_load);

    ImageLoadInfo irrad_load;
    irrad_load.loadf = true;
    irrad_load.bytes_per_pixel = sizeof(float) * 4;
    irrad_load.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    irrad_load.data = irrad_buf.data();
    irrad_load.data_size = irrad_buf.size();
    irrad_load.generate_mipmaps = false;

    Image irrad_img = load_image(fcx, irrad_load);

    VkImageViewCreateInfo hdr_ivci = {};
    hdr_ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    hdr_ivci.image = hdr_img.image;
    hdr_ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    hdr_ivci.components = vk_no_swizzle();
    hdr_ivci.format = hdr_load.format;
    hdr_ivci.subresourceRange = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);

    Texture equirectangular_hdr = create_texture(fcx.cx.dev, hdr_img, hdr_ivci);
    fcx.bind([equirectangular_hdr, cx = &fcx.cx]() { destroy_texture(*cx, equirectangular_hdr); });

    hdr_ivci.image = irrad_img.image;
    hdr_ivci.format = irrad_load.format;

    Texture equirectangular_irrad = create_texture(fcx.cx.dev, irrad_img, hdr_ivci);
    fcx.bind([equirectangular_irrad, cx = &fcx.cx]() { destroy_texture(*cx, equirectangular_irrad); });

    TextureDesc hdr_desc;
    hdr_desc.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    hdr_desc.layers = 6;
    hdr_desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    hdr_desc.width = 512;
    hdr_desc.height = 512;
    hdr_desc.depth = 1;
    hdr_desc.mips = static_cast<uint32_t>(std::floor(std::log2(512))) + 1;
    hdr_desc.type = VK_IMAGE_TYPE_2D;
    hdr_desc.view_type = VK_IMAGE_VIEW_TYPE_CUBE;
    hdr_desc.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    hdr_desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    hdr_desc.samples = VK_SAMPLE_COUNT_1_BIT;

    Texture hdr = create_texture(fcx.cx, hdr_desc);
    fcx.bind([hdr, cx = &fcx.cx]() { destroy_texture(*cx, hdr); });

    TextureDesc irrad_desc = hdr_desc;
    irrad_desc.width = 128;
    irrad_desc.height = 128;
    irrad_desc.mips = 1;
    irrad_desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    irrad = create_texture(fcx.cx, irrad_desc);

    VkImageMemoryBarrier hdr_barrier = {};
    hdr_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    hdr_barrier.image = hdr.image.image;
    hdr_barrier.srcAccessMask = 0;
    hdr_barrier.dstAccessMask = 0;
    hdr_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    hdr_barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    hdr_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hdr_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hdr_barrier.subresourceRange = vk_subresource_range(0, 6, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);

    vkCmdPipelineBarrier(fcx.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &hdr_barrier);

    const LoadedMesh cube = load_mesh(fmt::format("{}/meshes/{}", PK_RESOURCE_DIR, "cube.obj"));

    VkBufferCreateInfo cube_bci = {};
    cube_bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    cube_bci.size = sizeof(Vertex) * cube.vertices.size();
    cube_bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    cube_bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    Buffer cube_verts = fcx.cx.alloc.create_buffer(cube_bci, VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    fcx.bind(cube_verts);

    cube_bci.size = sizeof(uint32_t) * cube.indices.size();
    cube_bci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    Buffer cube_inds = fcx.cx.alloc.create_buffer(cube_bci, VMA_MEMORY_USAGE_CPU_TO_GPU, true);
    fcx.bind(cube_inds);

    vk_mapped_write(fcx.cx.alloc, cube_verts, cube.vertices.data(), cube_verts.size);
    vk_mapped_write(fcx.cx.alloc, cube_inds, cube.indices.data(), cube_inds.size);

    RenderGraph rg;

    PassAttachment pa_hdr_eq;
    pa_hdr_eq.tex = equirectangular_hdr;
    pa_hdr_eq.subresource = hdr_ivci.subresourceRange;
    rg.push_attachment({"hdr.eq"}, pa_hdr_eq);

    PassAttachment pa_hdr;
    pa_hdr.tex = hdr;
    pa_hdr.subresource = vk_subresource_range(0, 6, 0, hdr_desc.mips, VK_IMAGE_ASPECT_COLOR_BIT);
    rg.push_attachment({"hdr"}, pa_hdr);

    PassAttachment pa_irrad;
    pa_irrad.tex = irrad;
    pa_irrad.subresource = vk_subresource_range(0, 6, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    rg.push_attachment({"irrad"}, pa_irrad);

    static const glm::mat4 capture_proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    static const glm::mat4 capture_views[] = {glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))};

    // Equirectangular -> Cubemap conversion
    for (uint8_t i = 0; i < 6; ++i) {
        const std::string name = fmt::format("hdr.face.{}", i);

        VkBufferCreateInfo scratch_bci = {};
        scratch_bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        scratch_bci.size = sizeof(glm::mat4);
        scratch_bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        scratch_bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        Buffer scratch_ubo = fcx.cx.alloc.create_buffer(scratch_bci, VMA_MEMORY_USAGE_CPU_TO_GPU, true);
        fcx.bind(scratch_ubo);

        const glm::mat4 mat = capture_proj * capture_views[i];
        vk_mapped_write(fcx.cx.alloc, scratch_ubo, &mat, sizeof(glm::mat4));

        DescriptorSetInfo set_info;
        set_info.bind_buffer(scratch_ubo, VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        set_info.bind_texture(equirectangular_hdr, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

        DescriptorKey dk;
        const DescriptorSet set = fcx.cx.descriptor_cache.get_set(dk, set_info);

        if (!fcx.cx.pipeline_cache.contains("ibl.equirectangular_to_cubemap")) {
            VkPipelineRasterizationStateCreateInfo prsci = vk_rasterization_state_create_info(VK_POLYGON_MODE_FILL);
            prsci.cullMode = VK_CULL_MODE_NONE;

            SimplePipelineBuilder builder = SimplePipelineBuilder::begin(fcx.cx.dev, nullptr, fcx.cx.descriptor_cache, fcx.cx.pipeline_cache);
            builder.add_shader(fcx.cx.shader_cache.get("cubemap.vs"), VK_SHADER_STAGE_VERTEX_BIT);
            builder.add_shader(fcx.cx.shader_cache.get("equirectangular_to_cubemap.fs"), VK_SHADER_STAGE_FRAGMENT_BIT);
            builder.add_attachment(vk_color_blend_attachment_state());
            builder.set_rasterization_state(prsci);
            builder.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            builder.vertex_input<Vertex>(Vertex::Position);
            builder.set_samples(VK_SAMPLE_COUNT_1_BIT);
            builder.push_desc_set(set_info);

            fcx.cx.pipeline_cache.add("ibl.equirectangular_to_cubemap", builder.info());
        }

        VkImageViewCreateInfo ivci = {};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = hdr.image.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.components = vk_no_swizzle();
        ivci.format = hdr.image.format;
        ivci.subresourceRange = vk_subresource_range(i, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);

        Texture hdr_face = create_texture(fcx.cx.dev, hdr.image, ivci);
        fcx.bind([hdr_face, dev = fcx.cx.dev]() { vkDestroyImageView(dev, hdr_face.view, nullptr); });

        PassAttachment face;
        face.tex = hdr_face;
        face.subresource = vk_subresource_range(i, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
        rg.push_attachment({name}, face);

        RenderPass pass;
        pass.width = 512;
        pass.height = 512;
        pass.layers = 1;
        pass.push_color_output({name}, vk_clear_color({0.f, 0.f, 0.f, 1.f}));
        pass.push_dependent(
            {"hdr"}, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        pass.set_exec([=](FrameContext& fcx, const RenderGraph& rg, VkRenderPass pass) {
            const Pipeline pipeline = fcx.cx.pipeline_cache.get(pass, 0, "ibl.equirectangular_to_cubemap");

            vkCmdBindPipeline(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);

            const VkViewport viewport = vk_viewport(0.f, 0.f, 512.f, 512.f, 0.f, 1.f);
            const VkRect2D scissor = vk_rect(0, 0, 512, 512);

            vkCmdSetViewport(fcx.cmd, 0, 1, &viewport);
            vkCmdSetScissor(fcx.cmd, 0, 1, &scissor);

            vkCmdBindVertexBuffers(fcx.cmd, 0, 1, &cube_verts.buffer, &cube_verts.offset);
            vkCmdBindIndexBuffer(fcx.cmd, cube_inds.buffer, cube_inds.offset, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &set.set, 0, nullptr);

            vkCmdDrawIndexed(fcx.cmd, cube.indices.size(), 1, 0, 0, 0);
        });

        rg.push_pass(pass);
    }

    for (uint8_t i = 0; i < 6; ++i) {
        const std::string name = fmt::format("irrad.face.{}", i);

        VkBufferCreateInfo scratch_bci = {};
        scratch_bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        scratch_bci.size = sizeof(glm::mat4);
        scratch_bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        scratch_bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        Buffer scratch_ubo = fcx.cx.alloc.create_buffer(scratch_bci, VMA_MEMORY_USAGE_CPU_TO_GPU, true);
        fcx.bind(scratch_ubo);

        const glm::mat4 mat = capture_proj * capture_views[i];
        vk_mapped_write(fcx.cx.alloc, scratch_ubo, &mat, sizeof(glm::mat4));

        DescriptorSetInfo set_info;
        set_info.bind_buffer(scratch_ubo, VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        set_info.bind_texture(equirectangular_irrad, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

        DescriptorKey dk;
        const DescriptorSet set = fcx.cx.descriptor_cache.get_set(dk, set_info);

        if (!fcx.cx.pipeline_cache.contains("ibl.equirectangular_to_cubemap")) {
            VkPipelineRasterizationStateCreateInfo prsci = vk_rasterization_state_create_info(VK_POLYGON_MODE_FILL);
            prsci.cullMode = VK_CULL_MODE_NONE;

            SimplePipelineBuilder builder = SimplePipelineBuilder::begin(fcx.cx.dev, nullptr, fcx.cx.descriptor_cache, fcx.cx.pipeline_cache);
            builder.add_shader(fcx.cx.shader_cache.get("cubemap.vs"), VK_SHADER_STAGE_VERTEX_BIT);
            builder.add_shader(fcx.cx.shader_cache.get("equirectangular_to_cubemap.fs"), VK_SHADER_STAGE_FRAGMENT_BIT);
            builder.add_attachment(vk_color_blend_attachment_state());
            builder.set_rasterization_state(prsci);
            builder.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            builder.vertex_input<Vertex>(Vertex::Position);
            builder.set_samples(VK_SAMPLE_COUNT_1_BIT);
            builder.push_desc_set(set_info);

            fcx.cx.pipeline_cache.add("ibl.equirectangular_to_cubemap", builder.info());
        }

        VkImageViewCreateInfo ivci = {};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = irrad.image.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.components = vk_no_swizzle();
        ivci.format = irrad.image.format;
        ivci.subresourceRange = vk_subresource_range(i, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);

        Texture irrad_face = create_texture(fcx.cx.dev, irrad.image, ivci);
        fcx.bind([irrad_face, dev = fcx.cx.dev]() { vkDestroyImageView(dev, irrad_face.view, nullptr); });

        PassAttachment face;
        face.tex = irrad_face;
        face.subresource = vk_subresource_range(i, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
        rg.push_attachment({name}, face);

        RenderPass pass;
        pass.width = 128;
        pass.height = 128;
        pass.layers = 1;
        pass.push_color_output({name}, vk_clear_color({0.f, 0.f, 0.f, 1.f}));
        pass.push_dependent(
            {"irrad"}, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        pass.set_exec([=](FrameContext& fcx, const RenderGraph& rg, VkRenderPass pass) {
            const Pipeline pipeline = fcx.cx.pipeline_cache.get(pass, 0, "ibl.equirectangular_to_cubemap");

            vkCmdBindPipeline(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);

            const VkViewport viewport = vk_viewport(0.f, 0.f, 128.f, 128.f, 0.f, 1.f);
            const VkRect2D scissor = vk_rect(0, 0, 128, 128);

            vkCmdSetViewport(fcx.cmd, 0, 1, &viewport);
            vkCmdSetScissor(fcx.cmd, 0, 1, &scissor);

            vkCmdBindVertexBuffers(fcx.cmd, 0, 1, &cube_verts.buffer, &cube_verts.offset);
            vkCmdBindIndexBuffer(fcx.cmd, cube_inds.buffer, cube_inds.offset, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &set.set, 0, nullptr);

            vkCmdDrawIndexed(fcx.cmd, cube.indices.size(), 1, 0, 0, 0);
        });

        rg.push_pass(pass);
    }

    rg.set_output({"hdr"}, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    rg.exec(fcx, fcx.cx.rg_cache);

    VkImageMemoryBarrier irrad_barrier = {};
    irrad_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    irrad_barrier.image = irrad.image.image;
    irrad_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    irrad_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    irrad_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    irrad_barrier.dstAccessMask = 0;
    irrad_barrier.subresourceRange = vk_subresource_range(0, 6, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    irrad_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    irrad_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    vkCmdPipelineBarrier(
        fcx.cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &irrad_barrier);

    for (uint32_t i = 0; i < 6; ++i) {
        generate_mipmaps(fcx, hdr.image, hdr.image.format, hdr_desc.mips, i);
    }

    TextureDesc prefilter_desc;
    prefilter_desc.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    prefilter_desc.layers = 6;
    prefilter_desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    prefilter_desc.width = 512;
    prefilter_desc.height = 512;
    prefilter_desc.depth = 1;
    prefilter_desc.mips = 9;
    prefilter_desc.view_type = VK_IMAGE_VIEW_TYPE_CUBE;
    prefilter_desc.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    prefilter_desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    prefilter_desc.samples = VK_SAMPLE_COUNT_1_BIT;

    prefilter = create_texture(fcx.cx, prefilter_desc);

    VkSamplerCreateInfo hdr_sci = {};
    hdr_sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    hdr_sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    hdr_sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    hdr_sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    hdr_sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    hdr_sci.minFilter = VK_FILTER_LINEAR;
    hdr_sci.magFilter = VK_FILTER_LINEAR;
    hdr_sci.minLod = 0.f;
    hdr_sci.maxLod = static_cast<float>(hdr_desc.mips);
    hdr_sci.maxAnisotropy = 16.f;
    hdr_sci.anisotropyEnable = VK_TRUE;

    // Dispatch compute shaders to prefilter HDR for varying roughness levels

    VkImageMemoryBarrier prefilter_barrier = {};
    prefilter_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    prefilter_barrier.image = prefilter.image.image;
    prefilter_barrier.srcAccessMask = 0;
    prefilter_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    prefilter_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    prefilter_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    prefilter_barrier.subresourceRange = vk_subresource_range(0, 6, 0, prefilter_desc.mips, VK_IMAGE_ASPECT_COLOR_BIT);
    prefilter_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prefilter_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    vkCmdPipelineBarrier(fcx.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &prefilter_barrier);

    DescriptorSetInfo prefilter_set_layout_info;
    prefilter_set_layout_info.bind_texture({}, nullptr, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    prefilter_set_layout_info.bind_texture({}, nullptr, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);

    const VkDescriptorSetLayout prefilter_set_layout = fcx.cx.descriptor_cache.get_layout(prefilter_set_layout_info);

    VkPushConstantRange pcr = {};
    pcr.offset = 0;
    pcr.size = sizeof(float);
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo prefilter_plci = {};
    prefilter_plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    prefilter_plci.setLayoutCount = 1;
    prefilter_plci.pSetLayouts = &prefilter_set_layout;
    prefilter_plci.pPushConstantRanges = &pcr;
    prefilter_plci.pushConstantRangeCount = 1;

    VkPipelineLayout prefilter_layout;
    vk_log(vkCreatePipelineLayout(fcx.cx.dev, &prefilter_plci, nullptr, &prefilter_layout));
    fcx.bind([prefilter_layout, dev = fcx.cx.dev] { vkDestroyPipelineLayout(dev, prefilter_layout, nullptr); });

    VkShaderModule prefilter_shader = fcx.cx.shader_cache.get("prefilter.comp");

    VkComputePipelineCreateInfo prefilter_cpci = {};
    prefilter_cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    prefilter_cpci.layout = prefilter_layout;
    prefilter_cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    prefilter_cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    prefilter_cpci.stage.pName = "main";
    prefilter_cpci.stage.module = prefilter_shader;

    VkPipeline prefilter_pipeline;
    vk_log(vkCreateComputePipelines(fcx.cx.dev, nullptr, 1, &prefilter_cpci, nullptr, &prefilter_pipeline));
    fcx.bind([prefilter_pipeline, dev = fcx.cx.dev] { vkDestroyPipeline(dev, prefilter_pipeline, nullptr); });

    vkCmdBindPipeline(fcx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefilter_pipeline);

    uint32_t prefilter_width = prefilter.image.extent.width;
    uint32_t prefilter_height = prefilter.image.extent.height;

    for (uint32_t i = 0; i < prefilter_desc.mips; ++i) {
        VkImageViewCreateInfo ivci = {};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = prefilter.image.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        ivci.components = vk_no_swizzle();
        ivci.format = prefilter.image.format;
        ivci.subresourceRange = vk_subresource_range(0, 6, i, 1, VK_IMAGE_ASPECT_COLOR_BIT);

        const Texture mip_view = create_texture(fcx.cx.dev, prefilter.image, ivci);
        fcx.bind([mip_view, dev = fcx.cx.dev] { vkDestroyImageView(dev, mip_view.view, nullptr); });

        DescriptorSetInfo set_info;
        set_info.bind_texture(hdr, fcx.cx.sampler_cache.get(hdr_sci), VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        set_info.bind_texture(mip_view, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL);

        DescriptorKey dk;
        const DescriptorSet set = fcx.cx.descriptor_cache.get_set(dk, set_info);

        const float a = static_cast<float>(i) / 9.f;

        vkCmdBindDescriptorSets(fcx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefilter_layout, 0, 1, &set.set, 0, nullptr);
        vkCmdPushConstants(fcx.cmd, prefilter_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &a);

        vkCmdDispatch(fcx.cmd, prefilter_width, prefilter_height, 1);

        prefilter_width /= 2;
        prefilter_height /= 2;
    }

    prefilter_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    prefilter_barrier.dstAccessMask = 0;
    prefilter_barrier.oldLayout = prefilter_barrier.newLayout;
    prefilter_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    vkCmdPipelineBarrier(fcx.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &prefilter_barrier);

    uniforms.sun_dir = glm::normalize(glm::vec4{1.f, 2.f, -1.f, 0.f});
    uniforms.sun_radiant_flux = (glm::vec4{255.f, 255.f, 250.f, 255.f} / 255.f) * 50.f;
}

void PBRGraphicsPass::cleanup(FrameContext& fcx) {
    destroy_texture(fcx.cx, depth);
    destroy_texture(fcx.cx, out);
    destroy_texture(fcx.cx, ec_dfg_lut);
    destroy_texture(fcx.cx, ibl_dfg_lut);
    destroy_texture(fcx.cx, prefilter);
    destroy_texture(fcx.cx, irrad);
    fcx.cx.alloc.destroy(ubo);
}

void PBRGraphicsPass::add_resources(RenderGraph& rg) {
    PassAttachment out_attachment;
    out_attachment.tex = out;
    out_attachment.subresource = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT);

    rg.push_attachment({"pbr.out"}, out_attachment);

    PassAttachment depth_attachment;
    depth_attachment.tex = depth;
    depth_attachment.subresource = vk_subresource_range(0, 1, 0, 1, VK_IMAGE_ASPECT_DEPTH_BIT);

    rg.push_attachment({"pbr.depth"}, depth_attachment);
}

std::vector<RenderPass> PBRGraphicsPass::pass(FrameContext& fcx) {
    vk_mapped_write(fcx.cx.alloc, ubo, &uniforms, sizeof(SceneUniforms));

    RenderPass pass;

    pass.width = fcx.cx.width;
    pass.height = fcx.cx.height;
    pass.layers = 1;

    pass.push_color_output({"pbr.out"}, vk_clear_color(glm::vec4{2.f, 2.f, 2.f, 255.f} / 255.f));
    pass.set_depth_stencil({"pbr.depth"}, vk_clear_depth(1.f, 0));
    pass.push_texture_input({"shadow.map"});
    pass.set_exec([this](FrameContext& fcx, const RenderGraph& rg, VkRenderPass rp) { render(fcx, rg, rp); });

    return {pass};
}

void PBRGraphicsPass::render(FrameContext& fcx, const RenderGraph& rg, VkRenderPass rp) {
    const VkViewport viewport = vk_viewport(0.f, 0.f, static_cast<float>(fcx.cx.width), static_cast<float>(fcx.cx.height), 0.f, 1.f);
    const VkRect2D scissor = vk_rect(0, 0, fcx.cx.width, fcx.cx.height);

    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.anisotropyEnable = VK_TRUE;
    sci.maxAnisotropy = 16.f;
    sci.minLod = 0.f;
    sci.maxLod = 8.f;

    VkSamplerCreateInfo prefilter_sci = {};
    prefilter_sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    prefilter_sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    prefilter_sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    prefilter_sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    prefilter_sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    prefilter_sci.minFilter = VK_FILTER_LINEAR;
    prefilter_sci.magFilter = VK_FILTER_LINEAR;
    prefilter_sci.minLod = 0.f;
    prefilter_sci.maxLod = 8.f;

    VkSamplerCreateInfo shadow_sci = {};
    shadow_sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    shadow_sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadow_sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadow_sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadow_sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    shadow_sci.minFilter = VK_FILTER_LINEAR;
    shadow_sci.magFilter = VK_FILTER_LINEAR;
    shadow_sci.minLod = 0.f;
    shadow_sci.maxLod = 1.f;
    shadow_sci.maxAnisotropy = 1.f;
    shadow_sci.mipLodBias = 0.f;
    shadow_sci.compareEnable = VK_TRUE;
    shadow_sci.compareOp = VK_COMPARE_OP_LESS;

    DescriptorSetInfo set_info;
    set_info.bind_buffer(fcx.cx.scene.pass.instance_buffer(), VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set_info.bind_buffer(fcx.cx.scene.pass.instance_indices_buffer(), VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set_info.bind_buffer(fcx.cx.scene.storage.material_buffer(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set_info.bind_textures(
        fcx.cx.scene.storage.get_textures(), fcx.cx.sampler_cache.get(sci), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    set_info.bind_texture(ec_dfg_lut, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    set_info.bind_texture(ibl_dfg_lut, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    set_info.bind_texture(prefilter, fcx.cx.sampler_cache.get(prefilter_sci), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    set_info.bind_texture(irrad, fcx.cx.sampler_cache.basic(), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    set_info.bind_buffer(ubo, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    set_info.bind_texture(
        rg.attachment({"shadow.map"}).tex, fcx.cx.sampler_cache.get(shadow_sci), VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    set_info.bind_buffer(fcx.cx.renderer->shadow_pass.ubo, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    const DescriptorSet set = fcx.cx.descriptor_cache.get_set(desc_key, set_info);

    if (!fcx.cx.pipeline_cache.contains("pbr.pipeline")) {
        SimplePipelineBuilder builder = SimplePipelineBuilder::begin(fcx.cx.dev, nullptr, fcx.cx.descriptor_cache, fcx.cx.pipeline_cache);
        builder.add_shader(fcx.cx.shader_cache.get("pbr.vs"), VK_SHADER_STAGE_VERTEX_BIT);
        builder.add_shader(fcx.cx.shader_cache.get("pbr.fs"), VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.add_attachment(vk_color_blend_attachment_state());
        builder.set_depth_stencil_state(vk_depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL));
        builder.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        builder.vertex_input<Vertex>();
        builder.set_samples(VK_SAMPLE_COUNT_4_BIT);
        builder.push_desc_set(set_info);

        fcx.cx.pipeline_cache.add("pbr.pipeline", builder.info());
    }

    const Pipeline pipeline = fcx.cx.pipeline_cache.get(rp, 0, "pbr.pipeline");

    vkCmdBindPipeline(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    vkCmdBindDescriptorSets(fcx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &set.set, 0, nullptr);

    vkCmdSetViewport(fcx.cmd, 0, 1, &viewport);
    vkCmdSetScissor(fcx.cmd, 0, 1, &scissor);

    fcx.cx.scene.pass.execute(fcx.cmd, fcx.cx.scene.storage);
}

} // namespace gfx
