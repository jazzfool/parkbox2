#include "vk_helpers.hpp"

#include "def.hpp"
#include "helpers.hpp"
#include "context.hpp"
#include "frame_context.hpp"

#include <spdlog/spdlog.h>
#include <span>
#include <stb_image.h>
#include <fstream>
#include <tiny_obj_loader.h>

namespace gfx {

void vk_log(VkResult result) {
#if !defined(NDEBUG)
    if (result != VK_SUCCESS) {
        spdlog::error("vulkan error, result code {}", result);
    }
#endif
}

Image load_image(FrameContext& fcx, const ImageLoadInfo& info) {
    int32_t width = 0, height = 0, chans = 0;
    stbi_set_flip_vertically_on_load(info.flip);

    void* pixels = nullptr;
    if (info.loadf) {
        pixels = stbi_loadf_from_memory(info.data, info.data_size, &width, &height, &chans, info.dchans);
    } else {
        pixels = stbi_load_from_memory(info.data, info.data_size, &width, &height, &chans, info.dchans);
    }
    stbi_set_flip_vertically_on_load(false);

    const uint32_t mip_levels = info.generate_mipmaps ? static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1 : 1;

    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.arrayLayers = 1;
    ici.mipLevels = mip_levels;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.format = info.format;
    ici.extent.width = width;
    ici.extent.height = height;
    ici.extent.depth = 1;

    if (info.generate_mipmaps)
        ici.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    Image img = fcx.cx.alloc.create_image(ici, VMA_MEMORY_USAGE_GPU_ONLY);

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bci.size = info.bytes_per_pixel * width * height;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    Buffer staging = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_ONLY, true);
    fcx.bind(staging);

    ::memcpy(staging.pmap, pixels, bci.size);
    vk_log(vmaFlushAllocation(fcx.cx.alloc.allocator, staging.allocation, staging.offset, bci.size));

    fcx.copy_to_image(staging, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, info.bytes_per_pixel, vk_subresource_layers(0, 1, 0, VK_IMAGE_ASPECT_COLOR_BIT));

    if (info.generate_mipmaps) {
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = img.image;
        barrier.subresourceRange = vk_subresource_range(0, 1, 0, mip_levels, VK_IMAGE_ASPECT_COLOR_BIT);

        vkCmdPipelineBarrier(fcx.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        generate_mipmaps(fcx, img, info.format, mip_levels, 0);
    }

    return img;
}

void generate_mipmaps(FrameContext& fcx, Image img, VkFormat format, uint32_t mip_levels, uint32_t layer) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = img.image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = layer;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mip_width = img.extent.width;
    int32_t mip_height = img.extent.height;

    for (uint32_t i = 1; i < mip_levels; ++i) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(fcx.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkImageBlit blit = {};
        blit.srcOffsets[0] = VkOffset3D{0, 0, 0};
        blit.srcOffsets[1] = VkOffset3D{mip_width, mip_height, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = layer;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = VkOffset3D{0, 0, 0};
        blit.dstOffsets[1] = VkOffset3D{mip_width > 1 ? mip_width / 2 : 1, mip_height > 1 ? mip_height / 2 : 1, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = layer;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(fcx.cmd, img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = 0;

        vkCmdPipelineBarrier(fcx.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        if (mip_width > 1.f)
            mip_width /= 2.f;
        if (mip_height > 1.f)
            mip_height /= 2.f;
    }

    barrier.subresourceRange.baseMipLevel = mip_levels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;

    vkCmdPipelineBarrier(fcx.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

Texture create_texture(VkDevice device, Image image, const VkImageViewCreateInfo& ivci) {
    PK_ASSERT(ivci.image == image.image);

    Texture tex;
    tex.image = image;

    vk_log(vkCreateImageView(device, &ivci, nullptr, &tex.view));

    return tex;
}

Texture create_texture(Context& cx, const TextureDesc& desc) {
    VkImageCreateInfo ici = image_desc();
    ici.flags = desc.flags;
    ici.imageType = desc.type;
    ici.extent.width = desc.width;
    ici.extent.height = desc.height;
    ici.extent.depth = desc.depth;
    ici.arrayLayers = desc.layers;
    ici.mipLevels = desc.mips;
    ici.usage = desc.usage;
    ici.samples = desc.samples;
    ici.usage = desc.usage;
    ici.format = desc.format;

    Image img = cx.alloc.create_image(ici, VMA_MEMORY_USAGE_GPU_ONLY);

    VkImageViewCreateInfo ivci = {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = img.image;
    ivci.viewType = desc.view_type;
    ivci.components = vk_no_swizzle();
    ivci.format = ici.format;
    ivci.subresourceRange = vk_subresource_range(0, desc.layers, 0, desc.mips, desc.aspect);

    return create_texture(cx.dev, img, ivci);
}

void destroy_texture(Context& cx, Texture tex) {
    vkDestroyImageView(cx.dev, tex.view, nullptr);
    cx.alloc.destroy(tex.image);
}

void load_shader(ShaderCache& sc, const std::string& name, VkShaderStageFlags stage) {
    if (!sc.contains(name))
        sc.load(name, stage);
}

LoadedMesh load_mesh(const std::string& file) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;

    std::ifstream f{file};
    tinyobj::LoadObj(&attrib, &shapes, nullptr, nullptr, nullptr, &f, nullptr);
    f.close();

    std::unordered_map<gfx::Vertex, uint32_t> unique_verts;

    LoadedMesh out;
    out.min = glm::vec3{INFINITY, INFINITY, INFINITY};
    out.max = -glm::vec3{INFINITY, INFINITY, INFINITY};

    for (const tinyobj::shape_t& shape : shapes) {
        for (const tinyobj::index_t& index : shape.mesh.indices) {
            gfx::Vertex vert;

            vert.position.x = attrib.vertices[3 * index.vertex_index];
            vert.position.y = attrib.vertices[3 * index.vertex_index + 1];
            vert.position.z = attrib.vertices[3 * index.vertex_index + 2];

            vert.normal.x = attrib.normals[3 * index.normal_index];
            vert.normal.y = attrib.normals[3 * index.normal_index + 1];
            vert.normal.z = attrib.normals[3 * index.normal_index + 2];

            vert.tex_coord.x = attrib.texcoords[2 * index.texcoord_index];
            vert.tex_coord.y = 1.f - attrib.texcoords[2 * index.texcoord_index + 1];

            if (unique_verts.count(vert) == 0) {
                unique_verts[vert] = static_cast<uint32_t>(out.vertices.size());
                out.vertices.push_back(vert);
            }

            out.indices.push_back(unique_verts[vert]);

            out.min = glm::min(out.min, vert.position);
            out.max = glm::max(out.max, vert.position);
        }
    }

    return out;
}

void vk_mapped_write(Allocator& alloc, Buffer buf, const void* data, std::size_t size) {
    ::memcpy(static_cast<uint8_t*>(buf.pmap) + buf.offset, data, size);
    vk_log(vmaFlushAllocation(alloc.allocator, buf.allocation, buf.offset, size));
}

VkRect2D vk_rect(int32_t x, int32_t y, uint32_t width, uint32_t height) {
    VkRect2D rect;
    rect.offset.x = x;
    rect.offset.y = y;
    rect.extent.width = width;
    rect.extent.height = height;
    return rect;
}

VkComponentMapping vk_no_swizzle() {
    VkComponentMapping comps;
    comps.r = VK_COMPONENT_SWIZZLE_R;
    comps.g = VK_COMPONENT_SWIZZLE_G;
    comps.b = VK_COMPONENT_SWIZZLE_B;
    comps.a = VK_COMPONENT_SWIZZLE_A;
    return comps;
}

VkSemaphore vk_create_semaphore(VkDevice device) {
    VkSemaphoreCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphore sema = nullptr;
    vk_log(vkCreateSemaphore(device, &sci, nullptr, &sema));

    return sema;
}

VkFence vk_create_fence(VkDevice device, bool signalled) {
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = signalled ? VK_FENCE_CREATE_SIGNALED_BIT : 0;

    VkFence fence = nullptr;
    vk_log(vkCreateFence(device, &fci, nullptr, &fence));

    return fence;
}

VkImageSubresourceRange vk_subresource_range(uint32_t base_layer, uint32_t layer_count, uint32_t base_mip, uint32_t mip_count, VkImageAspectFlags aspect) {
    VkImageSubresourceRange ran;
    ran.baseArrayLayer = base_layer;
    ran.layerCount = layer_count;
    ran.baseMipLevel = base_mip;
    ran.levelCount = mip_count;
    ran.aspectMask = aspect;
    return ran;
}

VkImageSubresourceLayers vk_subresource_layers(uint32_t base_layer, uint32_t layer_count, uint32_t mip, VkImageAspectFlags aspect) {
    VkImageSubresourceLayers layers;
    layers.aspectMask = aspect;
    layers.baseArrayLayer = base_layer;
    layers.layerCount = layer_count;
    layers.mipLevel = mip;
    return layers;
}

VkClearColorValue vk_clear_color(float r, float g, float b, float a) {
    VkClearColorValue clear;
    clear.float32[0] = r;
    clear.float32[1] = g;
    clear.float32[2] = b;
    clear.float32[3] = a;
    return clear;
}

VkClearColorValue vk_clear_color(glm::vec4 rgba) {
    return vk_clear_color(rgba.x, rgba.y, rgba.z, rgba.w);
}

VkClearDepthStencilValue vk_clear_depth(float depth, uint32_t stencil) {
    VkClearDepthStencilValue clear;
    clear.depth = depth;
    clear.stencil = stencil;
    return clear;
}

VkViewport vk_viewport(float x, float y, float width, float height, float near, float far) {
    VkViewport vp;
    vp.x = x;
    vp.y = y;
    vp.width = width;
    vp.height = height;
    vp.minDepth = near;
    vp.maxDepth = far;
    return vp;
}

VkPipelineShaderStageCreateInfo vk_pipeline_shader_stage_create_info(VkShaderStageFlagBits stage, VkShaderModule shader_module) {
    VkPipelineShaderStageCreateInfo pssci = {};
    pssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pssci.stage = stage;
    pssci.module = shader_module;
    pssci.pName = "main";
    return pssci;
}

VkPipelineVertexInputStateCreateInfo vk_vertex_input_state_create_info() {
    VkPipelineVertexInputStateCreateInfo pvisci = {};
    pvisci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    pvisci.vertexBindingDescriptionCount = 0;
    pvisci.vertexAttributeDescriptionCount = 0;
    return pvisci;
}

VkPipelineInputAssemblyStateCreateInfo vk_input_assembly_create_info(VkPrimitiveTopology topology) {
    VkPipelineInputAssemblyStateCreateInfo piasci = {};
    piasci.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    piasci.topology = topology;
    piasci.primitiveRestartEnable = false;
    return piasci;
}

VkPipelineRasterizationStateCreateInfo vk_rasterization_state_create_info(VkPolygonMode polygon_mode) {
    VkPipelineRasterizationStateCreateInfo prsci = {};
    prsci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    prsci.depthClampEnable = VK_FALSE;
    prsci.rasterizerDiscardEnable = VK_FALSE;
    prsci.polygonMode = polygon_mode;
    prsci.lineWidth = 1.f;
    prsci.cullMode = VK_CULL_MODE_BACK_BIT;
    prsci.frontFace = VK_FRONT_FACE_CLOCKWISE;
    prsci.depthBiasEnable = VK_FALSE;
    prsci.depthBiasConstantFactor = 0.f;
    prsci.depthBiasClamp = 0.f;
    prsci.depthBiasSlopeFactor = 0.f;
    return prsci;
}

VkPipelineMultisampleStateCreateInfo vk_multisampling_state_create_info(VkSampleCountFlagBits samples) {
    VkPipelineMultisampleStateCreateInfo pmsci = {};
    pmsci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    pmsci.sampleShadingEnable = VK_FALSE;
    pmsci.rasterizationSamples = samples;
    pmsci.minSampleShading = 1.f;
    pmsci.pSampleMask = nullptr;
    pmsci.alphaToCoverageEnable = VK_FALSE;
    pmsci.alphaToOneEnable = VK_FALSE;
    return pmsci;
}

VkPipelineColorBlendAttachmentState vk_color_blend_attachment_state() {
    VkPipelineColorBlendAttachmentState pcbas = {};
    pcbas.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    pcbas.blendEnable = false;
    return pcbas;
}

VkPipelineDepthStencilStateCreateInfo vk_depth_stencil_create_info(bool depth_test, bool depth_write, VkCompareOp compare_op) {
    VkPipelineDepthStencilStateCreateInfo pdssci = {};
    pdssci.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    pdssci.depthTestEnable = depth_test ? VK_TRUE : VK_FALSE;
    pdssci.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE;
    pdssci.depthCompareOp = depth_test ? compare_op : VK_COMPARE_OP_ALWAYS;
    pdssci.depthBoundsTestEnable = VK_FALSE;
    pdssci.minDepthBounds = 0.f;
    pdssci.maxDepthBounds = 1.f;
    pdssci.stencilTestEnable = VK_FALSE;
    return pdssci;
}

VkImageCreateInfo image_desc() {
    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.arrayLayers = 1;
    ici.extent.depth = 1;
    ici.format = VK_FORMAT_UNDEFINED;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.mipLevels = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    return ici;
}

VkBufferMemoryBarrier vk_buffer_barrier(Buffer buffer) {
    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.buffer = buffer.buffer;
    barrier.offset = buffer.offset;
    barrier.size = buffer.size;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    return barrier;
}

} // namespace gfx

bool operator==(const VkRenderPassCreateInfo& lhs, const VkRenderPassCreateInfo& rhs) {
    return lhs.flags == rhs.flags && HashSpan{lhs.pSubpasses, lhs.subpassCount} == HashSpan{rhs.pSubpasses, rhs.subpassCount} &&
           HashSpan{lhs.pAttachments, lhs.attachmentCount} == HashSpan{rhs.pAttachments, rhs.attachmentCount} &&
           HashSpan{lhs.pDependencies, lhs.dependencyCount} == HashSpan{rhs.pDependencies, rhs.dependencyCount};
}

bool operator==(const VkFramebufferCreateInfo& lhs, const VkFramebufferCreateInfo& rhs) {
    return lhs.flags == rhs.flags && lhs.width == rhs.width && lhs.height == rhs.height && lhs.layers == rhs.layers && lhs.renderPass == rhs.renderPass &&
           HashSpan{lhs.pAttachments, lhs.attachmentCount} == HashSpan{rhs.pAttachments, rhs.attachmentCount};
}

bool operator==(const VkSubpassDescription& lhs, const VkSubpassDescription& rhs) {
    return lhs.flags == rhs.flags && lhs.pipelineBindPoint == rhs.pipelineBindPoint &&
           HashSpan{lhs.pColorAttachments, lhs.colorAttachmentCount} == HashSpan{rhs.pColorAttachments, rhs.colorAttachmentCount} &&
           HashSpan{lhs.pInputAttachments, lhs.inputAttachmentCount} == HashSpan{rhs.pInputAttachments, rhs.inputAttachmentCount} &&
           HashSpan{lhs.pPreserveAttachments, lhs.preserveAttachmentCount} == HashSpan{rhs.pPreserveAttachments, rhs.preserveAttachmentCount} &&
           ((lhs.pResolveAttachments != nullptr && rhs.pResolveAttachments != nullptr)
                   ? HashSpan{lhs.pResolveAttachments, lhs.colorAttachmentCount} == HashSpan{rhs.pResolveAttachments, rhs.colorAttachmentCount}
                   : false) &&
           ((lhs.pDepthStencilAttachment != nullptr && rhs.pDepthStencilAttachment != nullptr) ? (*lhs.pDepthStencilAttachment == *rhs.pDepthStencilAttachment)
                                                                                               : false);
}

bool operator==(const VkAttachmentDescription& lhs, const VkAttachmentDescription& rhs) {
    return lhs.flags == rhs.flags && lhs.initialLayout == rhs.initialLayout && lhs.finalLayout == rhs.finalLayout && lhs.format == rhs.format &&
           lhs.loadOp == rhs.loadOp && lhs.stencilLoadOp == rhs.stencilLoadOp && lhs.storeOp == rhs.storeOp && lhs.stencilStoreOp == rhs.stencilStoreOp &&
           lhs.samples == rhs.samples;
}

bool operator==(const VkAttachmentReference& lhs, const VkAttachmentReference& rhs) {
    return lhs.attachment == rhs.attachment && lhs.layout == rhs.layout;
}

bool operator==(const VkSubpassDependency& lhs, const VkSubpassDependency& rhs) {
    return lhs.dependencyFlags == rhs.dependencyFlags && lhs.srcSubpass == rhs.srcSubpass && lhs.dstSubpass == rhs.dstSubpass &&
           lhs.srcAccessMask == rhs.srcAccessMask && lhs.dstAccessMask == rhs.dstAccessMask && lhs.srcStageMask == rhs.srcStageMask &&
           lhs.dstStageMask == rhs.dstStageMask;
}

bool operator==(const VkDescriptorSetLayoutCreateInfo& lhs, const VkDescriptorSetLayoutCreateInfo& rhs) {
    return lhs.flags == rhs.flags && HashSpan{lhs.pBindings, lhs.bindingCount} == HashSpan{rhs.pBindings, rhs.bindingCount};
}

bool operator==(const VkDescriptorSetLayoutBinding& lhs, const VkDescriptorSetLayoutBinding& rhs) {
    return lhs.stageFlags == rhs.stageFlags && lhs.binding == rhs.binding && lhs.descriptorCount == rhs.descriptorCount &&
           lhs.descriptorType == rhs.descriptorType;
}

bool operator==(const VkSamplerCreateInfo& lhs, const VkSamplerCreateInfo& rhs) {
    return lhs.flags == rhs.flags && lhs.magFilter == rhs.magFilter && lhs.minFilter == rhs.minFilter && lhs.mipmapMode == rhs.mipmapMode &&
           lhs.addressModeU == rhs.addressModeU && lhs.addressModeV == rhs.addressModeV && lhs.addressModeW == rhs.addressModeW &&
           float_cmp(lhs.mipLodBias, rhs.mipLodBias) && lhs.anisotropyEnable == rhs.anisotropyEnable && float_cmp(lhs.maxAnisotropy, rhs.maxAnisotropy) &&
           lhs.compareEnable == rhs.compareEnable && lhs.compareOp == rhs.compareOp && float_cmp(lhs.minLod, rhs.minLod) && float_cmp(lhs.maxLod, rhs.maxLod) &&
           lhs.borderColor == rhs.borderColor && lhs.unnormalizedCoordinates == rhs.unnormalizedCoordinates;
}

std::size_t std::hash<VkRenderPassCreateInfo>::operator()(const VkRenderPassCreateInfo& rpci) const {
    std::size_t h = 0;
    hash_combine(h, rpci.flags, HashSpan{rpci.pSubpasses, rpci.subpassCount}, HashSpan{rpci.pAttachments, rpci.attachmentCount},
        HashSpan{rpci.pDependencies, rpci.dependencyCount});
    return h;
}

std::size_t std::hash<VkFramebufferCreateInfo>::operator()(const VkFramebufferCreateInfo& fbci) const {
    std::size_t h = 0;
    hash_combine(h, fbci.flags, fbci.width, fbci.height, fbci.layers, fbci.renderPass, HashSpan{fbci.pAttachments, fbci.attachmentCount});
    return h;
}

std::size_t std::hash<VkSubpassDescription>::operator()(const VkSubpassDescription& desc) const {
    std::size_t h = 0;
    hash_combine(h, desc.flags, desc.pipelineBindPoint, HashSpan{desc.pColorAttachments, desc.colorAttachmentCount},
        HashSpan{desc.pInputAttachments, desc.inputAttachmentCount}, HashSpan{desc.pPreserveAttachments, desc.preserveAttachmentCount});
    if (desc.pResolveAttachments != nullptr) {
        hash_combine(h, HashSpan{desc.pResolveAttachments, desc.colorAttachmentCount});
    }
    if (desc.pDepthStencilAttachment != nullptr) {
        hash_combine(h, *desc.pDepthStencilAttachment);
    }
    return h;
}

std::size_t std::hash<VkAttachmentDescription>::operator()(const VkAttachmentDescription& desc) const {
    std::size_t h = 0;
    hash_combine(
        h, desc.flags, desc.initialLayout, desc.finalLayout, desc.format, desc.loadOp, desc.stencilLoadOp, desc.storeOp, desc.stencilStoreOp, desc.samples);
    return h;
}

std::size_t std::hash<VkAttachmentReference>::operator()(const VkAttachmentReference& desc) const {
    std::size_t h = 0;
    hash_combine(h, desc.attachment, desc.layout);
    return h;
}

std::size_t std::hash<VkSubpassDependency>::operator()(const VkSubpassDependency& dep) const {
    std::size_t h = 0;
    hash_combine(h, dep.srcSubpass, dep.dstSubpass, dep.dependencyFlags, dep.srcAccessMask, dep.dstAccessMask, dep.srcStageMask, dep.dstStageMask);
    return h;
}

std::size_t std::hash<VkDescriptorSetLayoutCreateInfo>::operator()(const VkDescriptorSetLayoutCreateInfo& dslci) const {
    std::size_t h = 0;
    hash_combine(h, dslci.flags, HashSpan{dslci.pBindings, dslci.bindingCount});
    return h;
}

std::size_t std::hash<VkDescriptorSetLayoutBinding>::operator()(const VkDescriptorSetLayoutBinding& binding) const {
    std::size_t h = 0;
    hash_combine(h, binding.stageFlags, binding.binding, binding.descriptorCount, binding.descriptorType);
    return h;
}

std::size_t std::hash<VkSamplerCreateInfo>::operator()(const VkSamplerCreateInfo& sci) const {
    std::size_t h = 0;
    hash_combine(h, sci.flags, sci.magFilter, sci.minFilter, sci.mipmapMode, sci.addressModeU, sci.addressModeV, sci.addressModeW, sci.mipLodBias,
        sci.anisotropyEnable, sci.maxAnisotropy, sci.compareEnable, sci.compareOp, sci.minLod, sci.maxLod, sci.borderColor, sci.unnormalizedCoordinates);
    return h;
}
