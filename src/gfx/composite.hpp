#pragma once

#include "gfx_pass.hpp"
#include "descriptor_cache.hpp"

namespace gfx {

class CompositePass final : public GFXPass {
  public:
    void init(FrameContext& fcx) override;
    void cleanup(FrameContext& fcx) override;

    void add_resources(RenderGraph& rg) override;
    std::vector<RenderPass> pass(FrameContext& fcx) override;

    void render(FrameContext& fcx, const RenderGraph& rg, VkRenderPass rp);

    Texture in;

  private:
    DescriptorKey key;
};

} // namespace gfx
