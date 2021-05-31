#include "passive.hpp"

#include "gfx/frame_context.hpp"
#include "world.hpp"
#include "mesh.hpp"

namespace world {

entt::entity spawn_grass(gfx::FrameContext& fcx, World& w) {
    entt::entity e = w.reg.create();

    MeshComponent mesh;
    mesh.mesh = w.static_mesh("sphere");
    mesh.material = w.material("metal");
    mesh.uv_scale = {1.f, 1.f};

    w.add_object(fcx.cx, mesh);

    w.reg.emplace<MeshComponent>(e, mesh);
    w.reg.emplace<TransformComponent>(e);

    return e;
}

void passive_system(gfx::FrameContext& fcx, World& w) {
}

} // namespace world
