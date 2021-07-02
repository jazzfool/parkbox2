#pragma once

#include "viz.hpp"

#include <vector>
#include <glm/vec3.hpp>
#include <entt/entt.hpp>

namespace gfx {
class FrameContext;
}

namespace world {

class World;

struct PlantNode final {
    glm::vec3 growth_vector;
    glm::vec3 direction;
    float radius;

    std::vector<PlantNode> branches;
};

struct PlantData final {
    PlantNode root;
};

struct PlantEnvironment final {
    glm::vec3 sun_dir;
    glm::vec3 gravity_up;
    float growth;
};

struct PlantComponent final {
    PlantMesh mesh;
    PlantNode root;
    bool reset;
};

void plant_step(PlantNode& root, PlantEnvironment& env, bool reset, std::vector<PlantNode*>& candidates);

entt::entity spawn_plant(gfx::FrameContext& fcx, World& world);
void plant_system(gfx::FrameContext& fcx, World& world, PlantEnvironment& env);

} // namespace world
