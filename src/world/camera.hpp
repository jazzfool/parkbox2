#pragma once

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

namespace gfx {
class FrameContext;
struct Context;
} // namespace gfx

namespace world {

class World;

struct CameraComponent {
    glm::vec3 pos;
    glm::vec3 forward;
    glm::vec3 up;
    glm::vec3 center;

    float length;

    float last_x;
    float last_y;

    float yaw;
    float pitch;
};

entt::entity spawn_camera(World& world);
void camera_system(gfx::FrameContext& fcx, World& world, float dt);
void camera_look(gfx::Context& cx, World& world, float x, float y);
void camera_zoom(gfx::Context& cx, World& world, float x, float y);

} // namespace world
