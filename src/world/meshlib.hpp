#pragma once

#include <vector>
#include <glm/vec3.hpp>

namespace gfx {
struct Vertex;
}

namespace world {

void append_uv_sphere_mesh(std::vector<gfx::Vertex>& verts, std::vector<uint32_t>& inds, glm::vec3 origin, float radius, uint32_t slices, uint32_t stacks);

}
