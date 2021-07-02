#include "meshlib.hpp"

#include "gfx/mesh.hpp"

#include <glm/gtc/constants.hpp>
#include <cmath>

namespace world {

uint32_t push_vertex(std::vector<gfx::Vertex>& verts, const gfx::Vertex& v) {
    verts.push_back(v);
    return verts.size() - 1;
}

void push_tri(std::vector<uint32_t>& inds, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t base = 0) {
    inds.push_back(base + v0);
    inds.push_back(base + v1);
    inds.push_back(base + v2);
}

// http://www.songho.ca/opengl/gl_sphere.html
void append_uv_sphere_mesh(std::vector<gfx::Vertex>& verts, std::vector<uint32_t>& inds, glm::vec3 origin, float radius, uint32_t slices, uint32_t stacks) {
    const uint32_t base_idx = verts.size();

    float x, y, z, xy;
    float nx, ny, nz, lengthInv = 1.f / radius;
    float s, t;

    const float sector_step = 2.f * glm::pi<float>() / static_cast<float>(slices);
    const float stack_step = glm::pi<float>() / static_cast<float>(stacks);
    float sector_angle, stack_angle;

    for (int i = 0; i <= stacks; ++i) {
        stack_angle = glm::pi<float>() / 2.f - static_cast<float>(i) * stack_step;
        xy = radius * std::cosf(stack_angle);
        z = radius * std::sinf(stack_angle);

        for (int j = 0; j <= slices; ++j) {
            gfx::Vertex vert;

            sector_angle = static_cast<float>(j) * sector_step;

            x = xy * std::cosf(sector_angle);
            y = xy * std::sinf(sector_angle);

            vert.position = glm::vec3{x, y, z} + origin;

            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;

            vert.normal = {nx, ny, nz};

            s = static_cast<float>(j) / static_cast<float>(slices);
            t = static_cast<float>(i) / static_cast<float>(stacks);

            vert.tex_coord = {s, t};

            verts.push_back(vert);
        }
    }

    uint32_t k1, k2;
    for (uint32_t i = 0; i < stacks; ++i) {
        k1 = i * (slices + 1);
        k2 = k1 + slices + 1;

        for (uint32_t j = 0; j < slices; ++j, ++k1, ++k2) {
            if (i != 0) {
                push_tri(inds, k1, k2, k1 + 1, base_idx);
            }

            if (i != (stacks - 1)) {
                push_tri(inds, k1 + 1, k2, k2 + 1, base_idx);
            }
        }
    }
}

} // namespace world
