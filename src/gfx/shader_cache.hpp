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

    void load(std::string_view name, VkShaderStageFlags stage);
    void load_str(std::string shader, std::string_view name, VkShaderStageFlags stage);
    VkShaderModule get(std::string_view name);

    bool contains(std::string_view name) const;

  private:
    Context* cx;
    std::unordered_map<std::size_t, VkShaderModule> cache;
};

} // namespace gfx
