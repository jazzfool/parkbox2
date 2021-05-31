#pragma once

#include <stdint.h>
#include <volk.h>
#include <vector>

namespace gfx {

class FrameContext;
class RenderGraph;
class RenderPass;

class GFXPass {
  public:
    virtual void init(FrameContext& fcx) = 0;
    virtual void cleanup(FrameContext& fcx) = 0;

    virtual void add_resources(RenderGraph& rg) = 0;
    virtual std::vector<RenderPass> pass(FrameContext& fcx) = 0;
};

} // namespace gfx
