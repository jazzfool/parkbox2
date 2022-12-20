#pragma once

#include "pbr.hpp"
#include "composite.hpp"
#include "resolve.hpp"
#include "world/world.hpp"
#include "shadow.hpp"
#include "ssao.hpp"
#include "prepass.hpp"
#include "ui.hpp"

#include <vector>
#include <array>

namespace gfx {

struct Context;
class GFXPass;

class Renderer final {
  public:
    static constexpr uint64_t FRAMES_IN_FLIGHT = 2;

    void init(Context& cx);
    void cleanup();

    void run();

    PBRGraphicsPass pbr_pass;
    CompositePass composite_pass;
    ResolvePass resolve_pass;
    ShadowPass shadow_pass;
    SSAOPass ssao_pass;
    PrepassPass prepass_pass;

    UIRenderer ui;

  private:
    struct FrameData final {
        VkSemaphore present_semaphore;
        VkSemaphore render_semaphore;
        VkFence render_fence;
    };

    void render();
    FrameData& current_frame();

    Context* cx;
    std::vector<FrameData> frame_data;
    world::World world;
    uint64_t frame_num;

    double time;
    double curr_time;
};

} // namespace gfx
