#pragma once

#include "types.hpp"
#include "vk_helpers.hpp"

#include <vector>
#include <list>
#include <volk.h>
#include <unordered_map>
#include <span.hpp>

namespace gfx {

struct PipelineInfo;

struct StoredDescriptorWrite final {
    VkWriteDescriptorSet write;
    std::vector<VkDescriptorImageInfo> images;
    std::vector<VkDescriptorBufferInfo> buffers;

    void rebind() {
        write.pImageInfo = images.data();
        write.pBufferInfo = buffers.data();
    }
};

class DescriptorSetInfo final {
  public:
    static constexpr inline std::size_t MAX_BINDINGS = 64;

    DescriptorSetInfo();

    void bind_texture(
        Texture texture, VkSampler sampler, VkShaderStageFlags stages, VkDescriptorType type, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    void bind_textures(const std::vector<Texture>& textures, VkSampler sampler, VkShaderStageFlags stages, VkDescriptorType type,
        VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    void bind_buffer(Buffer buffer, VkShaderStageFlags stages, VkDescriptorType type);

    VkDescriptorSetLayoutCreateInfo vk_layout() const;
    std::vector<StoredDescriptorWrite> write(VkDevice dev, VkDescriptorSet set) const;
    std::vector<StoredDescriptorWrite> write_diff(VkDevice dev, tcb::span<const StoredDescriptorWrite> prev_writes, VkDescriptorSet set) const;

  private:
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<StoredDescriptorWrite> writes;
};

struct DescriptorKey final {
  public:
    DescriptorKey();
    DescriptorKey(const DescriptorKey&) = delete;
    DescriptorKey(DescriptorKey&&) = default;

    DescriptorKey& operator=(const DescriptorKey&) = delete;
    DescriptorKey& operator=(DescriptorKey&&) = default;

  private:
    friend class DescriptorCache;

    uint64_t key;
};

template <typename... Ts>
struct DescriptorKeyList final {
  public:
    DescriptorKey& get(const Ts&... k) {
        std::size_t h = 0;
        hash_combine(h, k...);
        if (keys.count(h) == 0)
            keys.emplace(h, DescriptorKey{});
        return keys.at(h);
    }

  private:
    std::unordered_map<std::size_t, DescriptorKey> keys;
};

struct DescriptorSet final {
    VkDescriptorSet set;
    VkDescriptorSetLayout layout;
};

class DescriptorCache final {
  public:
    void init(VkDevice dev);
    void cleanup();

    VkDescriptorSetLayout get_layout(const DescriptorSetInfo& info);
    DescriptorSet get_set(DescriptorKey& key, const DescriptorSetInfo& info);

    void reset_pools();

  private:
    VkDescriptorPool get_pool();
    VkDescriptorSet allocate_set(VkDescriptorSetLayout layout);

    VkDevice dev;
    std::unordered_map<std::size_t, VkDescriptorSetLayout> layout_cache;
    std::unordered_map<uint64_t, std::pair<DescriptorSet, std::vector<StoredDescriptorWrite>>> set_cache;

    VkDescriptorPool active_pool;
    std::vector<VkDescriptorPool> used_pools;
    std::vector<VkDescriptorPool> free_pools;
};

} // namespace gfx
