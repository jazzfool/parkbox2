#pragma once

#include <volk.h>

namespace gfx {

struct Context;
class FrameContext;

class UIRenderer final {
  public:
    void late_init(FrameContext& fcx, VkRenderPass rp);
    void cleanup(Context& cx);

    bool begin();
    void end(FrameContext& fcx);

  private:
    bool initialized = false;
    bool begun = false;
    VkDescriptorPool pool;
};

} // namespace gfx
