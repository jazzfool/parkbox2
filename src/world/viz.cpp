#include "viz.hpp"

#include "gfx/indirect.hpp"
#include "world.hpp"
#include "gfx/context.hpp"
#include "gfx/frame_context.hpp"
#include "plant.hpp"
#include "meshlib.hpp"

#include <vector>

namespace world {

void build_viz_mesh(std::vector<gfx::Vertex>& verts, std::vector<uint32_t>& inds, const PlantNode& node, glm::vec3 pos) {
    append_uv_sphere_mesh(verts, inds, pos, node.radius, 16, 32);

    for (const PlantNode& branch : node.branches) {
        build_viz_mesh(verts, inds, branch, pos + glm::normalize(branch.direction) * (branch.radius + node.radius) * 0.9f);
    }
}

PlantMesh viz_plant_mesh(gfx::FrameContext& fcx, const PlantNode& root, glm::vec3 origin) {
    std::vector<gfx::Vertex> vertices;
    std::vector<uint32_t> indices;

    build_viz_mesh(vertices, indices, root, origin + glm::normalize(root.direction) * root.radius);

    const gfx::BufferAllocation verts = fcx.cx.scene.storage.allocate_vertices(vertices.size());
    const gfx::BufferAllocation inds = fcx.cx.scene.storage.allocate_indices(indices.size());

    fcx.stage(*verts, vertices.data());
    fcx.stage(*inds, indices.data());

    const gfx::IndirectMeshKey mesh_key = gfx::indirect_mesh_key(verts, inds);

    return PlantMesh{verts, inds, mesh_key};
}

void cleanup_plant_mesh(gfx::FrameContext& fcx, const PlantMesh& mesh) {
    fcx.cx.scene.storage.free_vertices(mesh.verts);
    fcx.cx.scene.storage.free_indices(mesh.inds);
}

} // namespace world
