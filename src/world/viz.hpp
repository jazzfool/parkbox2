#pragma once

#include "gfx/indirect.hpp"

namespace gfx {
class FrameContext;
} // namespace gfx

namespace world {

struct PlantNode;

struct PlantMesh final {
    gfx::BufferAllocation verts;
    gfx::BufferAllocation inds;
    gfx::IndirectMeshKey mesh_key;
};

PlantMesh viz_plant_mesh(gfx::FrameContext& fcx, const PlantNode& root, glm::vec3 origin);
void cleanup_plant_mesh(gfx::FrameContext& fcx, const PlantMesh& mesh);

} // namespace world
