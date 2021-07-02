#pragma once

#include "types.hpp"
#include "allocator.hpp"

#include <future>
#include <vector>
#include <functional>
#include <span.hpp>

namespace gfx {

struct Context;

/*
  Abstraction for submitting GPU commands that use resources.
  The main attraction is the binding of resources to the frame context's lifetime.

  Frame contexts aren't necessarily only for frames-in-flight, e.g. they're also used for the initialization phase and inter-frame memory transfers.
*/
class FrameContext final {
  public:
    FrameContext(Context& cx);

    FrameContext(FrameContext&&) = default;
    FrameContext(const FrameContext&) = delete;

    FrameContext& operator=(FrameContext&&) = delete;
    FrameContext& operator=(const FrameContext&) = delete;

    void copy(Buffer src, Buffer dst);
    void multicopy(Buffer src, Buffer dst, tcb::span<BufferCopy> copies);
    void copy_to_image(Buffer src, Image dst, VkImageLayout layout, uint32_t bytes_per_pixel, VkImageSubresourceLayers subresource);
    void stage(Buffer dst, const void* data);

    void bind(Buffer buffer);
    void bind(Image image);
    void bind(std::function<void()> fn);

    void begin();
    void end();

    std::future<void> submit(VkQueue queue) &&;
    std::future<void> wait(VkFence fence) &&;

    Context& cx;
    VkCommandBuffer cmd;

  private:
    friend void wait_fence(VkDevice dev, VkCommandBuffer cmd, FrameContext fcx);

    void cleanup();

    VkFence fence;
    bool owned_fence;
    std::vector<Buffer> buffer_binds;
    std::vector<Image> image_binds;
    std::vector<std::function<void()>> fn_binds;
};

} // namespace gfx
