#include "allocator.hpp"

#include "def.hpp"
#include "vk_helpers.hpp"
#include "context.hpp"

#include <spdlog/spdlog.h>

namespace gfx {

Buffer& BufferAllocation::operator*() {
    return buffer;
}

const Buffer& BufferAllocation::operator*() const {
    return buffer;
}

Buffer* BufferAllocation::operator->() {
    return &buffer;
}

const Buffer* BufferAllocation::operator->() const {
    return &buffer;
}

FreeListAllocator::FreeListAllocator(VkDeviceSize size) : size{size} {
    // initially all the memory is free
    frees.emplace(size, 0);
}

bool FreeListAllocator::alloc(VkDeviceSize size, ContiguousAllocation& out) {
    // no more memory available at all
    if (frees.empty() || size == 0)
        throw false;
    VkDeviceSize base = 0;
    VkDeviceSize total_size = 0;
    // std::map orders by key ascending
    for (auto [s, b] : frees) {
        if (size <= s) {
            // find smallest memory block possible
            total_size = s;
            base = b;
            break;
        }
    }
    // no large enough contiguous memory available
    if (total_size == 0)
        return false;
    // erase if contiguous memory all used up
    if (size == total_size)
        frees.erase(total_size);
    // otherwise truncate contiguous memory
    else
        frees[total_size] += size;
    out = {base, size};
    return true;
}

void FreeListAllocator::free(ContiguousAllocation alloc) {
    // make this memory available for alloc
    frees.emplace(alloc.size, alloc.offset);
}

VkDeviceSize FreeListAllocator::size_hint() const {
    return size;
}

SlabAllocator::SlabAllocator(uint32_t num_blocks, VkDeviceSize slab_size) : num_blocks{num_blocks}, slab_size{slab_size} {
    slabs.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        slabs.push_back(i * slab_size);
    }
}

bool SlabAllocator::alloc(VkDeviceSize size, ContiguousAllocation& out) {
    if (slabs.empty())
        return false;
    out.offset = slabs.back();
    out.size = slab_size;
    slabs.pop_back();
    return true;
}

void SlabAllocator::free(ContiguousAllocation alloc) {
    slabs.push_back(alloc.offset);
}

VkDeviceSize SlabAllocator::size_hint() const {
    return num_blocks * slab_size;
}

void Allocator::init(Context& cx) {
    VmaVulkanFunctions vk_fns{};
    vk_fns.vkAllocateMemory = vkAllocateMemory;
    vk_fns.vkBindBufferMemory = vkBindBufferMemory;
    vk_fns.vkBindImageMemory = vkBindImageMemory;
    vk_fns.vkCreateBuffer = vkCreateBuffer;
    vk_fns.vkCreateImage = vkCreateImage;
    vk_fns.vkDestroyBuffer = vkDestroyBuffer;
    vk_fns.vkDestroyImage = vkDestroyImage;
    vk_fns.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
    vk_fns.vkFreeMemory = vkFreeMemory;
    vk_fns.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
    vk_fns.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2KHR;
    vk_fns.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
    vk_fns.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2KHR;
    vk_fns.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
    vk_fns.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    vk_fns.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
    vk_fns.vkMapMemory = vkMapMemory;
    vk_fns.vkUnmapMemory = vkUnmapMemory;
    vk_fns.vkCmdCopyBuffer = vkCmdCopyBuffer;

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice = cx.phys_dev;
    allocator_info.device = cx.dev;
    allocator_info.instance = cx.instance;
    allocator_info.pVulkanFunctions = &vk_fns;

    vk_log(vmaCreateAllocator(&allocator_info, &allocator));
}

void Allocator::cleanup() {
    vmaDestroyAllocator(allocator);
}

Buffer Allocator::create_buffer(const VkBufferCreateInfo& bci, VmaMemoryUsage usage, bool mapped) {
    PK_ASSERT(bci.size > 0);

    VmaAllocationCreateInfo aci = {};
    aci.usage = usage;
    aci.flags = mapped ? VMA_ALLOCATION_CREATE_MAPPED_BIT : 0;

    VkBuffer buffer;

    VmaAllocation alloc;
    VmaAllocationInfo alloc_info;

    vk_log(vmaCreateBuffer(allocator, &bci, &aci, &buffer, &alloc, &alloc_info));

    if (mapped && alloc_info.pMappedData == nullptr) {
        vk_log(vmaMapMemory(allocator, alloc, &alloc_info.pMappedData));
        PK_ASSERT(alloc_info.pMappedData != nullptr);
    }

    Buffer out;
    out.buffer = buffer;
    out.offset = 0;
    out.size = bci.size;
    out.actual_size = alloc_info.size;
    out.allocation = alloc;
    out.pmap = mapped ? alloc_info.pMappedData : nullptr;

    return out;
}

Image Allocator::create_image(const VkImageCreateInfo& ici, VmaMemoryUsage usage) {
    VmaAllocationCreateInfo aci = {};
    aci.usage = usage;

    VkImage image;

    VmaAllocation alloc;
    VmaAllocationInfo alloc_info;

    vk_log(vmaCreateImage(allocator, &ici, &aci, &image, &alloc, &alloc_info));

    Image out;
    out.image = image;
    out.format = ici.format;
    out.samples = ici.samples;
    out.allocation = alloc;
    out.extent = ici.extent;
    out.num_mips = ici.mipLevels;
    out.layers = ici.arrayLayers;

    return out;
}

void Allocator::destroy(Buffer buffer) {
    if (buffer.offset > 0) {
        spdlog::warn("destroying a buffer slice");
    }

    vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
}

void Allocator::destroy(Image image) {
    vmaDestroyImage(allocator, image.image, image.allocation);
}

} // namespace gfx
