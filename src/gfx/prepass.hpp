#pragma once

#include "gfx_pass.hpp"
#include "types.hpp"
#include "descriptor_cache.hpp"

namespace gfx {

class PrepassPass final : public GFXPass {
  public:
    void init(FrameContext& fcx) override;
    void cleanup(FrameContext& fcx) override;

    void add_resources(FrameContext& fcx, RenderGraph& rg) override;
    std::vector<RenderPass> pass(FrameContext& fcx) override;

  private:
    void render(FrameContext& fcx, const RenderGraph& rg, VkRenderPass pass);

    DescriptorKey desc_key;
};

} // namespace gfx
