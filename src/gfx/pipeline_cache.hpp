#pragma once

#include "descriptor_cache.hpp"
#include "mesh.hpp"
#include "vk_helpers.hpp"

#include <volk.h>
#include <vector>
#include <string_view>
#include <unordered_map>

namespace gfx {

class DescriptorCache;

struct Pipeline final {
    void destroy(VkDevice device);

    VkPipeline pipeline;
    VkPipelineLayout layout;
};

class PipelineBuilder final {
  public:
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
    VkPipelineVertexInputStateCreateInfo vertex_input_info;
    VkPipelineInputAssemblyStateCreateInfo input_assembly;
    VkPipelineRasterizationStateCreateInfo rasterizer;
    std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments;
    VkPipelineMultisampleStateCreateInfo multisampling;
    VkPipelineLayout pipeline_layout;
    VkPipelineDepthStencilStateCreateInfo depth_stencil;
    VkPipelineCache cache;
    VkPipelineDynamicStateCreateInfo dynamic_state;
    uint32_t subpass;

    VkPipeline build(VkDevice device, VkRenderPass pass);
};

struct PipelineInfo final {
    std::vector<DescriptorSetInfo> desc_sets;
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
    std::vector<VkPushConstantRange> push_consts;
    VertexInputDescription vertex_input;
    VkPipelineDepthStencilStateCreateInfo depth_stencil;
    VkPipelineRasterizationStateCreateInfo rasterization;
    std::vector<VkPipelineColorBlendAttachmentState> color_blend_states;
    VkPipelineInputAssemblyStateCreateInfo input_assembly;
    VkPipelineDynamicStateCreateInfo dynamic_state;
    VkSampleCountFlagBits samples;

    // viewport, scissor, and primitive toplogy are decided at retrieval time
};

class SimplePipelineBuilder final {
  public:
    static SimplePipelineBuilder begin(VkDevice dev, VkRenderPass pass, DescriptorCache& dc, class PipelineCache& cache);

    SimplePipelineBuilder& add_shader(VkShaderModule shader, VkShaderStageFlagBits stage);
    SimplePipelineBuilder& push_desc_set(DescriptorSetInfo set);
    SimplePipelineBuilder& push_constant(uint32_t offset, uint32_t size, VkShaderStageFlags stage);
    SimplePipelineBuilder& set_vertex_input(VertexInputDescription vi);
    SimplePipelineBuilder& set_primitive_topology(VkPrimitiveTopology primitive_topology);
    SimplePipelineBuilder& set_depth_stencil_state(VkPipelineDepthStencilStateCreateInfo depth_stencil);
    SimplePipelineBuilder& set_rasterization_state(VkPipelineRasterizationStateCreateInfo rasterization);
    SimplePipelineBuilder& add_attachment(VkPipelineColorBlendAttachmentState pcbas);
    SimplePipelineBuilder& set_subpass(uint32_t subpass);
    SimplePipelineBuilder& set_samples(VkSampleCountFlagBits samples);

    template <typename VertexT, typename... Args>
    SimplePipelineBuilder& vertex_input(Args&&... arg) {
        return set_vertex_input(VertexT::description(std::forward<Args>(arg)...));
    }

    Pipeline build();
    PipelineInfo info() const;

  private:
    VkDevice dev;
    VkRenderPass pass;
    DescriptorCache* dc;
    class PipelineCache* cache;

    std::vector<DescriptorSetInfo> desc_sets;
    std::vector<VkPushConstantRange> push_consts;
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
    VertexInputDescription vi;
    VkPrimitiveTopology primitive_topology;
    VkPipelineDepthStencilStateCreateInfo depth_stencil;
    VkPipelineRasterizationStateCreateInfo rasterization;
    std::vector<VkPipelineColorBlendAttachmentState> color_blend_states;
    uint32_t subpass;
    VkSampleCountFlagBits samples;
};

struct PipelineHandle final {
    const std::size_t hash;
};

class PipelineCache final {
  public:
    PipelineCache();
    PipelineCache(const PipelineCache&) = delete;

    PipelineCache& operator=(const PipelineCache&) = delete;

    void init(VkDevice dev, DescriptorCache& dc);
    void cleanup();

    PipelineHandle add(std::string_view name, const PipelineInfo& pi);
    PipelineHandle add(PipelineHandle handle, const PipelineInfo& pi);

    Pipeline get(VkRenderPass pass, uint32_t subpass, std::string_view name);
    Pipeline get(VkRenderPass pass, uint32_t subpass, PipelineHandle handle);

    PipelineInfo info(std::string_view name);
    PipelineInfo info(PipelineHandle handle);

    bool contains(std::string_view name) const;
    bool contains(PipelineHandle handle) const;

    VkPipelineCache cache;

  private:
    DescriptorCache* dc;
    VkDevice dev;
    std::unordered_map<std::size_t, PipelineInfo> pipeline_infos;
    std::unordered_map<std::size_t, Pipeline> pipelines;
};

} // namespace gfx
