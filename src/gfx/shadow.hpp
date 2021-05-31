#pragma once

#include "gfx_pass.hpp"
#include "descriptor_cache.hpp"

#include <array>
#include <glm/mat4x4.hpp>

namespace gfx {

class ShadowPass final : public GFXPass {
  public:
    static constexpr inline uint8_t NUM_CASCADES = 4;
    static constexpr inline uint32_t DIM = 2048;

    void init(FrameContext& fcx) override;
    void cleanup(FrameContext& fcx) override;

    void add_resources(RenderGraph& rg) override;
    std::vector<RenderPass> pass(FrameContext& fcx) override;

    Buffer ubo;

  private:
    struct Uniforms final {
        glm::mat4 views[NUM_CASCADES];
        glm::mat4 projs[NUM_CASCADES];
        glm::vec4 cascade_splits;
    };

    static Uniforms compute_cascades(Context& cx);
    void render(uint32_t cascade, FrameContext& fcx, const RenderGraph& rg, VkRenderPass pass);

    DescriptorKey desc_key;
    std::array<Texture, NUM_CASCADES> depth_views;
    Texture depths;
};

} // namespace gfx
