#include "mesh.hpp"

#include "world.hpp"
#include "gfx/frame_context.hpp"
#include "gfx/context.hpp"
#include "camera.hpp"

namespace world {

void gpu_mesh_update(World& w, entt::entity e) {
    const MeshComponent& mesh = w.reg.get<MeshComponent>(e);
    const TransformComponent& transform = w.reg.get<TransformComponent>(e);

    gfx::IndirectObject& obj = w.cx->scene.pass.object(mesh.gpu_object);

    obj.transform = transform.mat();
    obj.material = mesh.material;
    obj.mesh = mesh.mesh;
    obj.uv_scale = mesh.uv_scale;

    w.cx->scene.pass.update_object(mesh.gpu_object);
}

} // namespace world