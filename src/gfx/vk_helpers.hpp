#pragma once

#include "types.hpp"
#include "mesh.hpp"

#include <volk.h>
#include <utility>
#include <string>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace gfx {

struct Context;
class FrameContext;
class Allocator;

struct TextureDesc final {
    VkImageCreateFlags flags = {};
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t width;
    uint32_t height;
    uint32_t depth = 1;
    uint32_t layers = 1;
    uint32_t mips = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageUsageFlags usage;
    VkFormat format;
};

void vk_log(VkResult result);

struct ImageLoadInfo final {
    const uint8_t* data = nullptr;
    std::size_t data_size = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool generate_mipmaps = false;
    int32_t dchans = 4;
    bool loadf = false;
    uint32_t bytes_per_pixel = 4;
    bool flip = false;
};

struct LoadedMesh final {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    glm::vec3 min;
    glm::vec3 max;
};

Image load_image(FrameContext& fcx, const ImageLoadInfo& info);
void generate_mipmaps(FrameContext& fcx, Image img, VkFormat format, uint32_t mip_levels, uint32_t layer);
Texture create_texture(VkDevice device, Image image, const VkImageViewCreateInfo& ivci);
Texture create_texture(Context& cx, const TextureDesc& desc);
void destroy_texture(Context& cx, Texture tex);
void load_shader(class ShaderCache& sc, const std::string& name, VkShaderStageFlags stage);
LoadedMesh load_mesh(const std::string& file);
void vk_mapped_write(Allocator& alloc, Buffer buf, const void* data, std::size_t size);

VkRect2D vk_rect(int32_t x, int32_t y, uint32_t width, uint32_t height);
VkComponentMapping vk_no_swizzle();
VkSemaphore vk_create_semaphore(VkDevice device);
VkFence vk_create_fence(VkDevice device, bool signalled = false);
VkImageSubresourceRange vk_subresource_range(uint32_t base_layer, uint32_t layer_count, uint32_t base_mip, uint32_t mip_count, VkImageAspectFlags aspect);
VkImageSubresourceLayers vk_subresource_layers(uint32_t base_layer, uint32_t layer_count, uint32_t mip, VkImageAspectFlags aspect);
VkClearColorValue vk_clear_color(float r, float g, float b, float a);
VkClearColorValue vk_clear_color(glm::vec4 rgba);
VkClearDepthStencilValue vk_clear_depth(float depth, uint32_t stencil);
VkViewport vk_viewport(float x, float y, float width, float height, float near, float far);
VkPipelineShaderStageCreateInfo vk_pipeline_shader_stage_create_info(VkShaderStageFlagBits stage, VkShaderModule shader_module);
VkPipelineVertexInputStateCreateInfo vk_vertex_input_state_create_info();
VkPipelineInputAssemblyStateCreateInfo vk_input_assembly_create_info(VkPrimitiveTopology topology);
VkPipelineRasterizationStateCreateInfo vk_rasterization_state_create_info(VkPolygonMode polygon_mode);
VkPipelineMultisampleStateCreateInfo vk_multisampling_state_create_info(VkSampleCountFlagBits samples);
VkPipelineColorBlendAttachmentState vk_color_blend_attachment_state();
VkPipelineDepthStencilStateCreateInfo vk_depth_stencil_create_info(bool depth_test, bool depth_write, VkCompareOp compare_op);
VkImageCreateInfo image_desc();
VkBufferMemoryBarrier vk_buffer_barrier(Buffer buffer);

} // namespace gfx

bool operator==(const VkRenderPassCreateInfo& lhs, const VkRenderPassCreateInfo& rhs);
bool operator==(const VkFramebufferCreateInfo& lhs, const VkFramebufferCreateInfo& rhs);
bool operator==(const VkSubpassDescription& lhs, const VkSubpassDescription& rhs);
bool operator==(const VkAttachmentDescription& lhs, const VkAttachmentDescription& rhs);
bool operator==(const VkAttachmentReference& lhs, const VkAttachmentReference& rhs);
bool operator==(const VkSubpassDependency& lhs, const VkSubpassDependency& rhs);
bool operator==(const VkDescriptorSetLayoutCreateInfo& lhs, const VkDescriptorSetLayoutCreateInfo& rhs);
bool operator==(const VkDescriptorSetLayoutBinding& lhs, const VkDescriptorSetLayoutBinding& rhs);
bool operator==(const VkSamplerCreateInfo& lhs, const VkSamplerCreateInfo& rhs);

namespace std {

template <>
struct hash<VkRenderPassCreateInfo> {
    std::size_t operator()(const VkRenderPassCreateInfo& rpci) const;
};

template <>
struct hash<VkFramebufferCreateInfo> {
    std::size_t operator()(const VkFramebufferCreateInfo& fbci) const;
};

template <>
struct hash<VkSubpassDescription> {
    std::size_t operator()(const VkSubpassDescription& desc) const;
};

template <>
struct hash<VkAttachmentDescription> {
    std::size_t operator()(const VkAttachmentDescription& desc) const;
};

template <>
struct hash<VkAttachmentReference> {
    std::size_t operator()(const VkAttachmentReference& desc) const;
};

template <>
struct hash<VkSubpassDependency> {
    std::size_t operator()(const VkSubpassDependency& dep) const;
};

template <>
struct hash<VkDescriptorSetLayoutCreateInfo> {
    std::size_t operator()(const VkDescriptorSetLayoutCreateInfo& dslci) const;
};

template <>
struct hash<VkDescriptorSetLayoutBinding> {
    std::size_t operator()(const VkDescriptorSetLayoutBinding& binding) const;
};

template <>
struct hash<VkSamplerCreateInfo> {
    std::size_t operator()(const VkSamplerCreateInfo& sci) const;
};

} // namespace std
