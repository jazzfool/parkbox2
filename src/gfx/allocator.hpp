#pragma once

#include "types.hpp"
#include "def.hpp"

#include <vk_mem_alloc.h>
#include <vector>
#include <map>

namespace gfx {

struct Context;

struct ContiguousAllocation final {
    VkDeviceSize offset;
    VkDeviceSize size;
};

struct BufferAllocation final {
    Buffer buffer;
    ContiguousAllocation alloc;

    Buffer& operator*();
    const Buffer& operator*() const;

    Buffer* operator->();
    const Buffer* operator->() const;
};

class FreeListAllocator final {
  public:
    FreeListAllocator(VkDeviceSize size);

    bool alloc(VkDeviceSize size, ContiguousAllocation& out);
    void free(ContiguousAllocation alloc);

    VkDeviceSize size_hint() const;

  private:
    VkDeviceSize size;
    // key = size, value = ptr
    std::map<uint64_t, uint64_t> frees;
};

class SlabAllocator final {
  public:
    SlabAllocator(uint32_t num_blocks, VkDeviceSize slab_size);

    bool alloc(VkDeviceSize size, ContiguousAllocation& out);
    void free(ContiguousAllocation alloc);

    VkDeviceSize size_hint() const;

  private:
    uint32_t num_blocks;
    VkDeviceSize slab_size;
    std::vector<VkDeviceSize> slabs;
};

template <typename Alloc>
class BufferArena final {
  public:
    BufferAllocation alloc(VkDeviceSize size) {
        PK_ASSERT(size > 0);

        ContiguousAllocation block;
        PK_ASSERT(allocator.alloc(size, block));

        Buffer buf = buffer;
        buf.offset += block.offset;
        buf.actual_size = block.size;
        buf.size = size;

        return BufferAllocation{buf, block};
    }

    void free(const BufferAllocation& allocation) {
        allocator.free(allocation.alloc);
    }

    Buffer buffer;

  private:
    friend class Allocator;

    BufferArena(Buffer buffer, Alloc alloc) : buffer{buffer}, allocator{alloc} {
        static uint64_t NEXT_ID = 0;
        id = ++NEXT_ID;
    }

    Alloc allocator;
    uint64_t id;
};

class Allocator final {
  public:
    void init(Context& cx);
    void cleanup();

    Buffer create_buffer(const VkBufferCreateInfo& bci, VmaMemoryUsage usage, bool mapped);
    Image create_image(const VkImageCreateInfo& ici, VmaMemoryUsage usage);

    template <typename Alloc>
    BufferArena<Alloc> create_arena(Alloc alloc, VkBufferCreateInfo bci, VmaMemoryUsage usage, bool mapped) {
        PK_ASSERT(bci.size >= alloc.size_hint());
        Buffer buffer = create_buffer(bci, usage, mapped);
        return BufferArena<Alloc>{buffer, std::move(alloc)};
    }

    void destroy(Buffer buffer);
    void destroy(Image image);

    template <typename Alloc>
    void destroy(BufferArena<Alloc> arena) {
        destroy(arena.buffer);
    }

    VmaAllocator allocator;
};

} // namespace gfx
