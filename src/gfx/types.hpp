#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

namespace gfx {

struct Buffer {
    Buffer slice(VkDeviceSize off, VkDeviceSize size) const {
        return Buffer{buffer, offset + off, size, actual_size, allocation, pmap};
    }

    Buffer full() const {
        return Buffer{buffer, 0, actual_size, actual_size, allocation, pmap};
    }

    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
    VkDeviceSize actual_size;
    VmaAllocation allocation;

    void* pmap;
};

struct Image {
    Image() : image{nullptr}, allocation{nullptr} {
    }

    VkImage image;
    VkFormat format;
    VkSampleCountFlagBits samples;
    VmaAllocation allocation;
    VkExtent3D extent;
    uint32_t num_mips;
    uint32_t layers;
};

struct Texture {
    Texture() : view{nullptr} {
    }

    Image image;
    VkImageView view;
};

struct BufferCopy final {
    VkDeviceSize src_offset;
    VkDeviceSize dst_offset;
    VkDeviceSize size;
};

} // namespace gfx
