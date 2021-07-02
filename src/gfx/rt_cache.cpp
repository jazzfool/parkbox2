#include "rt_cache.hpp"

#include "vk_helpers.hpp"
#include "def.hpp"

namespace gfx {

void RenderTargetCache::init(Context& cx) {
    this->cx = &cx;
}

void RenderTargetCache::cleanup() {
    reset();
}

Texture RenderTargetCache::get(std::string_view name, const TextureDesc& desc) {
    const uint64_t key = std::hash<std::string_view>{}(name);
    if (cache.count(key) == 0) {
        const Texture tex = create_texture(*cx, desc);
        cache.emplace(key, tex);
        return tex;
    } else {
        return cache.at(key);
    }
}

void RenderTargetCache::remove(std::string_view name) {
    const uint64_t key = std::hash<std::string_view>{}(name);
    PK_ASSERT(cache.count(key) == 1);
    destroy_texture(*cx, cache.at(key));
    cache.erase(key);
}

void RenderTargetCache::reset() {
    for (const auto& [_, tex] : cache) {
        destroy_texture(*cx, tex);
    }
    cache.clear();
}

} // namespace gfx
