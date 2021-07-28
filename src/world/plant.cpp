#include "plant.hpp"

#include "world.hpp"
#include "gfx/frame_context.hpp"
#include "gfx/context.hpp"
#include "mesh.hpp"

#include <glm/gtx/euler_angles.hpp>
#include <glm/geometric.hpp>
#include <random>
#include <vector>

namespace world {

void node_step(PlantNode& node, PlantEnvironment& env, bool reset, std::vector<PlantNode*>& candidates) {
    static constexpr float GROWTH_SCALE = 0.00001f;
    static constexpr float GROWTH_BIAS = 0.00005f;

    static std::random_device rd;
    static std::uniform_real_distribution<float> dist11{-1.f, 1.f};
    static std::uniform_real_distribution<float> dist01{0.f, 1.f};
    static std::mt19937_64 mt{rd()};

    for (PlantNode& branch : node.branches) {
        node_step(branch, env, reset, candidates);
    }

    if (reset) {
        node.growth_vector = glm::normalize(node.growth_vector) * 0.000001f;
    }

    const float growth = std::min((GROWTH_SCALE * node.radius) + GROWTH_BIAS, env.growth);
    node.growth_vector += glm::normalize(node.growth_vector + env.sun_dir * 0.00001f + env.gravity_up * 0.00001f) * growth;

    const float len = glm::length(node.growth_vector);
    node.growth_vector = glm::normalize(node.growth_vector);
    node.growth_vector = glm::clamp(node.growth_vector, node.direction - glm::vec3{0.1f, 0.1f, 0.1f}, node.direction + glm::vec3{0.1f, 0.1f, 0.1f});
    node.growth_vector *= len;

    if (len > 10.f) {
        candidates.push_back(&node);
    }
}

void plant_step(PlantNode& root, PlantEnvironment& env, bool reset, std::vector<PlantNode*>& candidates) {
    node_step(root, env, reset, candidates);
}

entt::entity spawn_plant(gfx::FrameContext& fcx, World& world) {
    entt::entity e = world.reg.create();

    PlantComponent plant;

    plant.root.growth_vector = glm::vec3{0.f, -1.f, 0.f};
    plant.root.direction = glm::vec3{0.f, -1.f, 0.f};
    plant.root.radius = 0.5f;

    plant.mesh = viz_plant_mesh(fcx, plant.root, glm::vec3{0.f});
    fcx.cx.scene.pass.push_mesh(plant.mesh.mesh_key, glm::vec3{0.f}, 50.f);

    MeshComponent mesh;
    mesh.mesh = plant.mesh.mesh_key;
    mesh.material = world.material("purple");
    mesh.uv_scale = {1.f, 1.f};

    world.add_object(fcx.cx, mesh);

    world.reg.emplace<PlantComponent>(e, std::move(plant));
    world.reg.emplace<MeshComponent>(e, std::move(mesh));
    world.reg.emplace<TransformComponent>(e);

    return e;
}

void plant_system(gfx::FrameContext& fcx, World& world, PlantEnvironment& env) {
    static std::random_device rd;
    static std::uniform_real_distribution<float> dist11{-1.f, 1.f};
    static std::uniform_real_distribution<float> dist01{0.f, 1.f};
    static std::mt19937_64 mt{rd()};

    for (auto [e, plant, mesh] : world.reg.view<PlantComponent, MeshComponent>().each()) {
        std::vector<PlantNode*> candidates;
        for (uint32_t i = 0; i < 50; ++i)
            plant_step(plant.root, env, plant.reset, candidates);

        plant.reset = false;
        if (!candidates.empty()) {
            plant.reset = true;

            PlantNode* chosen = candidates.front();
            for (PlantNode* candidate : candidates) {
                if (candidate->branches.empty()) {
                    chosen = candidate;
                    break;
                }
            }

            PlantNode branch;
            branch.radius = chosen->radius * 0.85f;
            branch.growth_vector = glm::normalize(glm::vec3{dist11(mt), -dist01(mt), dist11(mt)}) * 0.000001f;
            branch.direction = glm::normalize(chosen->growth_vector);

            if (!chosen->branches.empty()) {
                branch.radius *= 0.9f;
                branch.growth_vector = glm::normalize(chosen->growth_vector) * 0.00001f;
            }

            chosen->growth_vector = glm::normalize(glm::vec3{dist11(mt), -dist01(mt), dist11(mt)}) * 0.000001f;

            chosen->branches.push_back(branch);

            const gfx::IndirectMeshKey old_mesh = plant.mesh.mesh_key;

            cleanup_plant_mesh(fcx, plant.mesh);
            plant.mesh = viz_plant_mesh(fcx, plant.root, glm::vec3{0.f});

            fcx.cx.scene.pass.update_mesh(old_mesh, plant.mesh.mesh_key, glm::vec3{0.f}, 50.f);

            mesh.mesh = plant.mesh.mesh_key;
            mesh.gpu_object.mesh = plant.mesh.mesh_key;
        }
    }
}

} // namespace world
