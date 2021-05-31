#include "frame_context.hpp"

#include "context.hpp"
#include "vk_helpers.hpp"
#include "def.hpp"

namespace gfx {

void wait_fence(VkDevice dev, VkCommandBuffer cmd, FrameContext fcx) {
    vk_log(vkWaitForFences(fcx.cx.dev, 1, &fcx.fence, VK_TRUE, UINT64_MAX));
    fcx.cx.frame_pool.replace(cmd);
    fcx.cleanup();
}

FrameContext::FrameContext(Context& cx) : cx{cx}, owned_fence{false} {
    cmd = cx.frame_pool.take();
}

void FrameContext::copy(Buffer src, Buffer dst) {
    PK_ASSERT(src.size <= dst.size);

    VkBufferCopy info;
    info.srcOffset = src.offset;
    info.dstOffset = dst.offset;
    info.size = src.size;

    vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &info);
}

void FrameContext::copy_to_image(Buffer src, Image dst, VkImageLayout layout, uint32_t bytes_per_pixel, VkImageSubresourceLayers subresource) {
    VkBufferImageCopy info;
    info.imageSubresource = subresource;
    info.imageOffset = {0, 0, 0};
    info.imageExtent = dst.extent;
    info.bufferOffset = src.offset;
    info.bufferRowLength = 0;
    info.bufferImageHeight = 0;

    VkImageMemoryBarrier image_barrier = {};
    image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_barrier.image = dst.image;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.subresourceRange.aspectMask = subresource.aspectMask;
    image_barrier.subresourceRange.baseMipLevel = subresource.mipLevel;
    image_barrier.subresourceRange.levelCount = 1;
    image_barrier.subresourceRange.baseArrayLayer = subresource.baseArrayLayer;
    image_barrier.subresourceRange.layerCount = subresource.layerCount;
    image_barrier.srcAccessMask = 0;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);

    vkCmdCopyBufferToImage(cmd, src.buffer, dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &info);

    image_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier.newLayout = layout;
    image_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier.dstAccessMask = 0;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);
}

void FrameContext::stage(Buffer dst, const void* data) {
    const VkDeviceSize size = dst.size;

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bci.size = size;

    Buffer staging = cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_ONLY, true);
    bind(staging);

    ::memcpy(staging.pmap, data, size);

    vk_log(vmaFlushAllocation(cx.alloc.allocator, staging.allocation, staging.offset, size));

    copy(staging, dst);
}

void FrameContext::bind(Buffer buffer) {
    buffer_binds.push_back(buffer);
}

void FrameContext::bind(Image image) {
    image_binds.push_back(image);
}

void FrameContext::bind(std::function<void()> fn) {
    fn_binds.push_back(fn);
}

void FrameContext::begin() {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vk_log(vkBeginCommandBuffer(cmd, &cbbi));
}

void FrameContext::end() {
    vk_log(vkEndCommandBuffer(cmd));
}

std::future<void> FrameContext::submit(VkQueue queue) && {
    VkFence fence;

    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    vk_log(vkCreateFence(cx.dev, &fci, nullptr, &fence));

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    vk_log(vkQueueSubmit(queue, 1, &si, fence));

    owned_fence = true;

    return std::move(*this).wait(fence);
}

std::future<void> FrameContext::wait(VkFence fence) && {
    this->fence = fence;
    owned_fence = false;
    return std::async(std::launch::async, wait_fence, cx.dev, cmd, std::move(*this));
}

void FrameContext::cleanup() {
    for (Buffer buffer : buffer_binds) {
        cx.alloc.destroy(buffer);
    }

    for (Image image : image_binds) {
        cx.alloc.destroy(image);
    }

    for (std::function<void()>& fn : fn_binds) {
        fn();
    }

    if (owned_fence)
        vkDestroyFence(cx.dev, fence, nullptr);
}

} // namespace gfx
