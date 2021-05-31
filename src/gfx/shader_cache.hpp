#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <volk.h>

namespace gfx {

struct Context;

class ShaderCache final {
  public:
    void init(Context& cx);
    void cleanup();

    void load(const std::string& name, VkShaderStageFlags stage);
    VkShaderModule get(const std::string& name);

    bool contains(const std::string& name) const;

  private:
    Context* cx;
    std::unordered_map<std::string, VkShaderModule> cache;
};

} // namespace gfx
