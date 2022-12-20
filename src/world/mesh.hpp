#pragma once

#include "gfx/indirect.hpp"

#include <entt/entt.hpp>
#include <glm/gtc/quaternion.hpp>

namespace world {

class World;

struct MeshComponent final {
    gfx::IndirectObjectHandle gpu_object;
    gfx::IndirectMeshKey mesh;
    uint32_t material;
    glm::vec2 uv_scale;
    std::string shader_type;
    std::string shader;
};

struct TransformComponent final {
    glm::vec3 pos;
    glm::quat rot;
    glm::vec3 scale;

    TransformComponent() : pos{0.f, 0.f, 0.f}, rot{0.f, 0.f, 0.f, 1.f}, scale{1.f, 1.f, 1.f} {
    }

    inline glm::mat4 mat() const {
        return glm::translate(glm::identity<glm::mat4>(), pos) * glm::mat4_cast(rot) * glm::scale(glm::identity<glm::mat4>(), scale);
    }
};

void gpu_mesh_update(World& w, entt::entity e);

} // namespace world
