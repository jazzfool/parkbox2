#pragma once

#include "gfx_pass.hpp"
#include "types.hpp"
#include "descriptor_cache.hpp"

#include <glm/mat4x4.hpp>

namespace gfx {

class SSAOPass final : public GFXPass {
  public:
    static constexpr inline float RESOLUTION = 0.5f;

    void init(FrameContext& fcx) override;
    void cleanup(FrameContext& fcx) override;

    void add_resources(FrameContext& fcx, RenderGraph& rg) override;
    std::vector<RenderPass> pass(FrameContext& fcx) override;

  private:
    void render(FrameContext& fcx, const RenderGraph& rg, VkRenderPass pass);
    void resize(int32_t, int32_t);

    DescriptorKey desc_key;
    bool use_a;
    bool first;
    Buffer ubo;
    glm::mat4 prev_vp;
};

} // namespace gfx
