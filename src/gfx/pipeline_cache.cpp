#include "pipeline_cache.hpp"

#include "helpers.hpp"

namespace gfx {

void Pipeline::destroy(VkDevice device) {
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
}

VkPipeline PipelineBuilder::build(VkDevice device, VkRenderPass pass) {
    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;
    // viewport and scissor is dynamic state

    VkPipelineColorBlendStateCreateInfo color_blending = {};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = color_blend_attachments.size();
    color_blending.pAttachments = color_blend_attachments.data();

    VkGraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = shader_stages.size();
    pipeline_info.pStages = shader_stages.data();
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.layout = pipeline_layout;
    pipeline_info.renderPass = pass;
    pipeline_info.subpass = subpass;
    pipeline_info.basePipelineHandle = nullptr;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pDynamicState = &dynamic_state;

    VkPipeline pipe;
    vk_log(vkCreateGraphicsPipelines(device, cache, 1, &pipeline_info, nullptr, &pipe));

    return pipe;
}

SimplePipelineBuilder SimplePipelineBuilder::begin(VkDevice device, VkRenderPass pass, DescriptorCache& dc, PipelineCache& cache) {
    SimplePipelineBuilder builder;
    builder.dev = device;
    builder.pass = pass;
    builder.dc = &dc;
    builder.cache = &cache;
    builder.depth_stencil = vk_depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);
    builder.rasterization = vk_rasterization_state_create_info(VK_POLYGON_MODE_FILL);
    builder.color_blend_states = {};
    builder.subpass = 0;
    builder.samples = VK_SAMPLE_COUNT_1_BIT;
    return builder;
}

SimplePipelineBuilder& SimplePipelineBuilder::add_shader(VkShaderModule shader, VkShaderStageFlagBits stage) {
    shader_stages.push_back(vk_pipeline_shader_stage_create_info(stage, shader));
    return *this;
}

SimplePipelineBuilder& SimplePipelineBuilder::push_desc_set(DescriptorSetInfo set) {
    desc_sets.push_back(set);
    return *this;
}

SimplePipelineBuilder& SimplePipelineBuilder::push_constant(uint32_t offset, uint32_t size, VkShaderStageFlags stage) {
    VkPushConstantRange pcr = {};
    pcr.offset = offset;
    pcr.size = size;
    pcr.stageFlags = stage;
    push_consts.push_back(pcr);
    return *this;
}

SimplePipelineBuilder& SimplePipelineBuilder::set_vertex_input(VertexInputDescription vi) {
    this->vi = vi;
    return *this;
}

SimplePipelineBuilder& SimplePipelineBuilder::set_primitive_topology(VkPrimitiveTopology pt) {
    primitive_topology = pt;
    return *this;
}

SimplePipelineBuilder& SimplePipelineBuilder::set_depth_stencil_state(VkPipelineDepthStencilStateCreateInfo ds) {
    depth_stencil = ds;
    return *this;
}

SimplePipelineBuilder& SimplePipelineBuilder::set_rasterization_state(VkPipelineRasterizationStateCreateInfo rs) {
    rasterization = rs;
    return *this;
}

SimplePipelineBuilder& SimplePipelineBuilder::add_attachment(VkPipelineColorBlendAttachmentState pcbas) {
    color_blend_states.push_back(pcbas);
    return *this;
}

SimplePipelineBuilder& SimplePipelineBuilder::set_subpass(uint32_t sp) {
    subpass = sp;
    return *this;
}

SimplePipelineBuilder& SimplePipelineBuilder::set_samples(VkSampleCountFlagBits sam) {
    samples = sam;
    return *this;
}

Pipeline SimplePipelineBuilder::build() {
    std::vector<VkDescriptorSetLayout> desc_layouts;
    desc_layouts.reserve(desc_sets.size());
    for (const auto& set : desc_sets) {
        desc_layouts.push_back(dc->get_layout(set));
    }

    const VkDynamicState dynamic_states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = desc_layouts.size();
    plci.pSetLayouts = desc_layouts.data();
    plci.pushConstantRangeCount = push_consts.size();
    plci.pPushConstantRanges = push_consts.data();

    VkPipelineLayout pipeline_layout;
    vk_log(vkCreatePipelineLayout(dev, &plci, nullptr, &pipeline_layout));

    PipelineBuilder pb;

    pb.shader_stages = shader_stages;
    pb.vertex_input_info = {};
    pb.vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    pb.vertex_input_info.flags = vi.flags;
    pb.vertex_input_info.vertexAttributeDescriptionCount = vi.attributes.size();
    pb.vertex_input_info.pVertexAttributeDescriptions = &vi.attributes[0];
    pb.vertex_input_info.vertexBindingDescriptionCount = vi.bindings.size();
    pb.vertex_input_info.pVertexBindingDescriptions = &vi.bindings[0];
    pb.input_assembly = vk_input_assembly_create_info(primitive_topology);
    pb.rasterizer = rasterization;
    pb.multisampling = vk_multisampling_state_create_info(samples);
    pb.color_blend_attachments = color_blend_states;
    pb.pipeline_layout = pipeline_layout;
    pb.depth_stencil = depth_stencil;
    pb.cache = cache->cache;
    pb.dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    pb.dynamic_state.dynamicStateCount = 2;
    pb.dynamic_state.pDynamicStates = dynamic_states;
    pb.subpass = subpass;

    auto pl = pb.build(dev, pass);

    Pipeline pipeline;
    pipeline.pipeline = pl;
    pipeline.layout = pipeline_layout;

    return pipeline;
}

PipelineInfo SimplePipelineBuilder::info() const {
    static const VkDynamicState dynamic_states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    PipelineInfo info;

    info.desc_sets = desc_sets;
    info.shader_stages = shader_stages;
    info.push_consts = push_consts;
    info.vertex_input = vi;
    info.depth_stencil = depth_stencil;
    info.rasterization = rasterization;
    info.color_blend_states = color_blend_states;
    info.input_assembly = vk_input_assembly_create_info(primitive_topology);
    info.dynamic_state = {};
    info.dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    info.dynamic_state.dynamicStateCount = 2;
    info.dynamic_state.pDynamicStates = dynamic_states;
    info.samples = samples;

    return info;
}

PipelineCache::PipelineCache() {
}

void PipelineCache::init(VkDevice device, DescriptorCache& dc) {
    dev = device;
    VkPipelineCacheCreateInfo pcci = {};
    pcci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vk_log(vkCreatePipelineCache(device, &pcci, nullptr, &cache));

    this->dc = &dc;
}

void PipelineCache::cleanup() {
    for (const auto& [_, pipe] : pipelines) {
        vkDestroyPipeline(dev, pipe.pipeline, nullptr);
        vkDestroyPipelineLayout(dev, pipe.layout, nullptr);
    }

    vkDestroyPipelineCache(dev, cache, nullptr);
}

PipelineHandle PipelineCache::add(std::string_view name, const PipelineInfo& pi) {
    return add(PipelineHandle{std::hash<std::string_view>{}(name)}, pi);
}

PipelineHandle PipelineCache::add(PipelineHandle handle, const PipelineInfo& pi) {
    pipeline_infos[handle.hash] = pi;
    return handle;
}

Pipeline PipelineCache::get(VkRenderPass pass, uint32_t subpass, std::string_view name) {
    return get(pass, subpass, PipelineHandle{std::hash<std::string_view>{}(name)});
}

Pipeline PipelineCache::get(VkRenderPass pass, uint32_t subpass, PipelineHandle handle) {
    std::size_t hash = 0;
    hash_combine(hash, reinterpret_cast<void*>(pass), handle.hash);

    const PipelineInfo& info = pipeline_infos.at(handle.hash);

    if (pipelines.count(hash) == 0) {
        std::vector<VkDescriptorSetLayout> desc_layouts;
        desc_layouts.reserve(info.desc_sets.size());

        for (const auto& set : info.desc_sets) {
            desc_layouts.push_back(dc->get_layout(set));
        }

        VkPipelineLayoutCreateInfo plci = {};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = desc_layouts.size();
        plci.pSetLayouts = desc_layouts.data();
        plci.pushConstantRangeCount = info.push_consts.size();
        plci.pPushConstantRanges = info.push_consts.data();

        VkPipelineLayout pipeline_layout;
        vk_log(vkCreatePipelineLayout(dev, &plci, nullptr, &pipeline_layout));

        PipelineBuilder pb;

        pb.shader_stages = info.shader_stages;

        pb.vertex_input_info = {};
        pb.vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        pb.vertex_input_info.vertexBindingDescriptionCount = info.vertex_input.bindings.size();
        pb.vertex_input_info.pVertexBindingDescriptions = info.vertex_input.bindings.data();
        pb.vertex_input_info.vertexAttributeDescriptionCount = info.vertex_input.attributes.size();
        pb.vertex_input_info.pVertexAttributeDescriptions = info.vertex_input.attributes.data();

        pb.input_assembly = info.input_assembly;
        pb.rasterizer = info.rasterization;
        pb.color_blend_attachments = info.color_blend_states;
        pb.multisampling = vk_multisampling_state_create_info(info.samples);
        pb.pipeline_layout = pipeline_layout;
        pb.depth_stencil = info.depth_stencil;
        pb.dynamic_state = info.dynamic_state;
        pb.cache = cache;
        pb.subpass = subpass;

        Pipeline pipe;
        pipe.pipeline = pb.build(dev, pass);
        pipe.layout = pipeline_layout;

        pipelines[hash] = pipe;
    }

    return pipelines.at(hash);
}

PipelineInfo PipelineCache::info(std::string_view name) {
    return info(PipelineHandle{std::hash<std::string_view>{}(name)});
}

PipelineInfo PipelineCache::info(PipelineHandle handle) {
    return pipeline_infos.at(handle.hash);
}

bool PipelineCache::contains(std::string_view name) const {
    return contains({std::hash<std::string_view>{}(name)});
}

bool PipelineCache::contains(PipelineHandle handle) const {
    return pipeline_infos.count(handle.hash) == 1;
}

} // namespace gfx
