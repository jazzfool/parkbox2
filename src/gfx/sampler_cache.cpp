#include "sampler_cache.hpp"

namespace gfx {

void SamplerCache::init(VkDevice dev) {
    this->dev = dev;
}

void SamplerCache::cleanup() {
    for (const auto& [_, sampler] : cache) {
        vkDestroySampler(dev, sampler, nullptr);
    }
}

VkSampler SamplerCache::basic() {
    VkSamplerCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minLod = 0.f;
    sci.maxLod = 1.f;
    sci.maxAnisotropy = 1.f;
    sci.mipLodBias = 0.f;

    return get(sci);
}

VkSampler SamplerCache::get(const VkSamplerCreateInfo& sci) {
    if (!cache.count(sci)) {
        VkSampler sampler;
        vk_log(vkCreateSampler(dev, &sci, nullptr, &sampler));
        cache.emplace(sci, sampler);
    }

    return cache.at(sci);
}

} // namespace gfx
