#pragma once

#include <entt/entt.hpp>

namespace gfx {
class FrameContext;
}

namespace world {

class World;

entt::entity spawn_grass(gfx::FrameContext& fcx, World& w);

void passive_system(gfx::FrameContext& fcx, World& w);

} // namespace world
