#pragma once

#include "pbr.hpp"
#include "composite.hpp"
#include "resolve.hpp"
#include "world/world.hpp"
#include "shadow.hpp"

#include <vector>
#include <array>

namespace gfx {

struct Context;
class GFXPass;

class Renderer final {
  public:
    void init(Context& cx);
    void cleanup();

    void run();

    PBRGraphicsPass pbr_pass;
    CompositePass composite_pass;
    ResolvePass resolve_pass;
    ShadowPass shadow_pass;

  private:
    void render();

    Context* cx;

    VkSemaphore present_semaphore;
    VkSemaphore render_semaphore;
    VkFence render_fence;

    world::World world;
};

} // namespace gfx