#pragma once

#include "vk_helpers.hpp"

#include <volk.h>
#include <unordered_map>

namespace gfx {

class SamplerCache final {
  public:
    void init(VkDevice dev);
    void cleanup();

    VkSampler basic();
    VkSampler get(const VkSamplerCreateInfo& sci);

  private:
    VkDevice dev;
    std::unordered_map<VkSamplerCreateInfo, VkSampler> cache;
};

} // namespace gfx
