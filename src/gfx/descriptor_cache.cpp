#include "descriptor_cache.hpp"

#include "def.hpp"
#include "vk_helpers.hpp"
#include "pipeline_cache.hpp"

#include <array>
#include <spdlog/spdlog.h>

namespace gfx {

bool cmp_write(const VkWriteDescriptorSet& lhs, const VkWriteDescriptorSet& rhs) {
    bool b = lhs.dstBinding == rhs.dstBinding && lhs.dstArrayElement == rhs.dstArrayElement && lhs.descriptorCount == rhs.descriptorCount &&
             lhs.descriptorType == rhs.descriptorType;

    if (lhs.pImageInfo && rhs.pImageInfo) {
        for (uint32_t i = 0; i < lhs.descriptorCount; ++i) {
            b = b && lhs.pImageInfo[i].imageLayout == rhs.pImageInfo[i].imageLayout;
            b = b && lhs.pImageInfo[i].imageView == rhs.pImageInfo[i].imageView;
            b = b && lhs.pImageInfo[i].sampler == rhs.pImageInfo[i].sampler;
        }
    } else if (lhs.pImageInfo != rhs.pImageInfo) {
        return false;
    }

    if (lhs.pBufferInfo && rhs.pBufferInfo) {
        for (uint32_t i = 0; i < lhs.descriptorCount; ++i) {
            b = b && lhs.pBufferInfo[i].buffer == rhs.pBufferInfo[i].buffer;
            b = b && lhs.pBufferInfo[i].offset == rhs.pBufferInfo[i].offset;
            b = b && lhs.pBufferInfo[i].range == rhs.pBufferInfo[i].range;
        }
    } else if (lhs.pBufferInfo != rhs.pBufferInfo) {
        return false;
    }

    return b;
}

DescriptorSetInfo::DescriptorSetInfo() {
    bindings.reserve(MAX_BINDINGS);
}

void DescriptorSetInfo::bind_texture(Texture texture, VkSampler sampler, VkShaderStageFlags stages, VkDescriptorType type, VkImageLayout layout) {
    PK_ASSERT(bindings.size() < MAX_BINDINGS);

    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = bindings.size();
    binding.descriptorCount = 1;
    binding.descriptorType = type;
    binding.stageFlags = stages;

    bindings.push_back(binding);

    StoredDescriptorWrite write = {};

    write.images.push_back({});
    write.images[0].imageLayout = layout;
    write.images[0].imageView = texture.view;
    write.images[0].sampler = sampler;

    write.write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.write.descriptorType = binding.descriptorType;
    write.write.descriptorCount = binding.descriptorCount;
    write.write.dstBinding = binding.binding;
    write.write.dstArrayElement = 0;
    write.write.pImageInfo = write.images.data();

    writes.push_back(write);
}

void DescriptorSetInfo::bind_textures(
    const std::vector<Texture>& textures, VkSampler sampler, VkShaderStageFlags stages, VkDescriptorType type, VkImageLayout layout) {
    PK_ASSERT(bindings.size() < MAX_BINDINGS);

    if (textures.empty())
        return;

    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = bindings.size();
    binding.descriptorCount = textures.size();
    binding.descriptorType = type;
    binding.stageFlags = stages;

    bindings.push_back(binding);

    StoredDescriptorWrite write = {};

    for (const Texture& texture : textures) {
        VkDescriptorImageInfo info = {};
        info.imageLayout = layout;
        info.imageView = texture.view;
        info.sampler = sampler;

        write.images.push_back(info);
    }

    write.write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.write.descriptorType = binding.descriptorType;
    write.write.descriptorCount = binding.descriptorCount;
    write.write.dstBinding = binding.binding;
    write.write.dstArrayElement = 0;
    write.write.pImageInfo = write.images.data();

    writes.push_back(write);
}

void DescriptorSetInfo::bind_buffer(Buffer buffer, VkShaderStageFlags stages, VkDescriptorType type) {
    PK_ASSERT(bindings.size() < MAX_BINDINGS);

    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = bindings.size();
    binding.descriptorCount = 1;
    binding.descriptorType = type;
    binding.stageFlags = stages;

    bindings.push_back(binding);

    StoredDescriptorWrite write = {};

    write.buffers.push_back({});
    write.buffers[0].buffer = buffer.buffer;
    write.buffers[0].offset = buffer.offset;
    write.buffers[0].range = buffer.size;

    write.write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.write.descriptorType = binding.descriptorType;
    write.write.descriptorCount = binding.descriptorCount;
    write.write.dstBinding = binding.binding;
    write.write.dstArrayElement = 0;
    write.write.pBufferInfo = write.buffers.data();

    writes.push_back(write);
}

VkDescriptorSetLayoutCreateInfo DescriptorSetInfo::vk_layout() const {
    VkDescriptorSetLayoutCreateInfo layout = {};
    layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = bindings.size();
    layout.pBindings = bindings.data();
    return layout;
}

std::vector<StoredDescriptorWrite> DescriptorSetInfo::write(VkDevice dev, VkDescriptorSet set) const {
    std::vector<VkWriteDescriptorSet> writes_cpy;
    writes_cpy.reserve(writes.size());
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        VkWriteDescriptorSet write = writes[i].write;
        write.dstSet = set;
        write.pImageInfo = writes[i].images.data();
        write.pBufferInfo = writes[i].buffers.data();
        writes_cpy.push_back(write);
    }
    vkUpdateDescriptorSets(dev, writes_cpy.size(), writes_cpy.data(), 0, nullptr);
    return writes;
}

std::vector<StoredDescriptorWrite> DescriptorSetInfo::write_diff(VkDevice dev, tcb::span<const StoredDescriptorWrite> prev_writes, VkDescriptorSet set) const {
    std::vector<VkWriteDescriptorSet> writes_cpy;
    writes_cpy.reserve(writes.size());
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        VkWriteDescriptorSet write = writes[i].write;
        write.dstSet = set;
        write.pImageInfo = writes[i].images.data();
        write.pBufferInfo = writes[i].buffers.data();

        VkWriteDescriptorSet prev_write = prev_writes[i].write;
        prev_write.pImageInfo = prev_writes[i].images.data();
        prev_write.pBufferInfo = prev_writes[i].buffers.data();

        if (cmp_write(write, prev_write))
            continue;

        writes_cpy.push_back(write);
    }
    vkUpdateDescriptorSets(dev, writes_cpy.size(), writes_cpy.data(), 0, nullptr);
    return writes;
}

DescriptorKey::DescriptorKey() {
    static uint64_t next = 0;
    key = next++;
}

void DescriptorCache::init(VkDevice dev) {
    this->dev = dev;
    active_pool = nullptr;
}

void DescriptorCache::cleanup() {
    for (VkDescriptorPool pool : used_pools) {
        vkDestroyDescriptorPool(dev, pool, nullptr);
    }

    for (VkDescriptorPool pool : free_pools) {
        vkDestroyDescriptorPool(dev, pool, nullptr);
    }

    for (const auto& [_, layout] : layout_cache) {
        vkDestroyDescriptorSetLayout(dev, layout, nullptr);
    }
}

VkDescriptorSetLayout DescriptorCache::get_layout(const DescriptorSetInfo& info) {
    const VkDescriptorSetLayoutCreateInfo layout_info = info.vk_layout();
    const std::size_t hash = std::hash<VkDescriptorSetLayoutCreateInfo>{}(layout_info);
    if (layout_cache.count(hash)) {
        return layout_cache.at(hash);
    }

    VkDescriptorSetLayout layout;
    vk_log(vkCreateDescriptorSetLayout(dev, &layout_info, nullptr, &layout));

    layout_cache.emplace(hash, layout);

    return layout;
}

DescriptorSet DescriptorCache::get_set(DescriptorKey& key, const DescriptorSetInfo& info) {
    const VkDescriptorSetLayout layout = get_layout(info);

    if (!set_cache.count(key.key)) {
        DescriptorSet set;
        set.layout = layout;
        set.set = allocate_set(layout);
        set_cache.emplace(key.key, std::make_pair(set, info.write(dev, set.set)));
        return set;
    } else {
        auto& [set, prev_writes] = set_cache.at(key.key);
        prev_writes = info.write_diff(dev, prev_writes, set.set);
        return set;
    }
}

void DescriptorCache::reset_pools() {
    for (VkDescriptorPool pool : used_pools) {
        vk_log(vkResetDescriptorPool(dev, pool, 0));
    }

    free_pools = used_pools;
    used_pools.clear();
    active_pool = nullptr;
}

VkDescriptorPool DescriptorCache::get_pool() {
    static constexpr std::pair<VkDescriptorType, float> pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 0.5f},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4.f},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4.f},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1.f},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1.f},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1.f},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2.f},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2.f},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1.f},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1.f},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 0.5f},
    };

    if (!free_pools.empty()) {
        VkDescriptorPool pool = free_pools.back();
        free_pools.pop_back();
        return pool;
    } else {
        std::vector<VkDescriptorPoolSize> sizes;
        sizes.reserve(sizeof(pool_sizes) / sizeof(pool_sizes[0]));
        for (const std::pair<VkDescriptorType, float>& size : pool_sizes) {
            sizes.push_back({size.first, static_cast<uint32_t>(size.second * 1000)});
        }

        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1000;
        dpci.poolSizeCount = sizes.size();
        dpci.pPoolSizes = sizes.data();

        VkDescriptorPool pool;
        vk_log(vkCreateDescriptorPool(dev, &dpci, nullptr, &pool));

        return pool;
    }
}

VkDescriptorSet DescriptorCache::allocate_set(VkDescriptorSetLayout layout) {
    if (active_pool == nullptr) {
        active_pool = get_pool();
        used_pools.push_back(active_pool);
    }

    VkDescriptorSetAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.pSetLayouts = &layout;
    alloc.descriptorPool = active_pool;
    alloc.descriptorSetCount = 1;

    VkDescriptorSet set;
    VkResult result = vkAllocateDescriptorSets(dev, &alloc, &set);
    bool realloc = false;

    switch (result) {
    case VK_SUCCESS:
        return set;
    case VK_ERROR_FRAGMENTED_POOL:
    case VK_ERROR_OUT_OF_POOL_MEMORY:
        realloc = true;
        break;
    default:
        spdlog::error("unable to allocate descriptor set");
        return nullptr;
    }

    if (realloc) {
        active_pool = get_pool();
        used_pools.push_back(active_pool);
        result = vkAllocateDescriptorSets(dev, &alloc, &set);
        if (result == VK_SUCCESS)
            return set;
    }

    return nullptr;
}

} // namespace gfx
