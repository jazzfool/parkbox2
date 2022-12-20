#pragma once

#include "gfx_pass.hpp"
#include "types.hpp"
#include "descriptor_cache.hpp"
#include "indirect.hpp"

#include <glm/mat4x4.hpp>

namespace gfx {

class UIRenderer;

class PBRGraphicsPass final : public GFXPass {
  public:
    void init(FrameContext& fcx) override;
    void cleanup(FrameContext& fcx) override;

    void add_resources(FrameContext& fcx, RenderGraph& rg) override;
    std::vector<RenderPass> pass(FrameContext& fcx) override;

    void render(FrameContext& fcx, const RenderGraph& rg, VkRenderPass rp);

  private:
    Texture ec_dfg_lut;
    Texture ibl_dfg_lut;
    Texture prefilter;
    Texture irrad;

    DescriptorKeyList<IndirectMeshPass*> desc_keys;
};

} // namespace gfx
