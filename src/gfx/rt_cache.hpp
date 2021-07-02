#pragma once

#include "types.hpp"

#include <volk.h>
#include <unordered_map>
#include <string>
#include <string_view>

namespace gfx {

struct Context;
struct TextureDesc;

class RenderTargetCache final {
  public:
    void init(Context& cx);
    void cleanup();

    Texture get(std::string_view name, const TextureDesc& desc);
    void remove(std::string_view name);
    void reset();

  private:
    Context* cx;
    std::unordered_map<uint64_t, Texture> cache;
};

} // namespace gfx
