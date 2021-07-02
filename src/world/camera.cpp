#include "camera.hpp"

#include "world.hpp"
#include "gfx/frame_context.hpp"
#include "gfx/context.hpp"
#include "helpers.hpp"

#include <GLFW/glfw3.h>

namespace world {

entt::entity spawn_camera(World& world) {
    entt::entity e = world.reg.create();

    CameraComponent cam;
    cam.pos = {0.f, 0.f, 0.f};
    cam.forward = {0.f, 0.f, -1.f};
    cam.up = {0.f, 1.f, 0.f};
    cam.center = {0.f, 0.f, 0.f};

    cam.length = 10.f;

    cam.last_x = -1.f;
    cam.last_y = -1.f;

    cam.yaw = 0.f;
    cam.pitch = 0.f;

    world.reg.emplace<CameraComponent>(e, cam);
    return e;
}

void set_camera_position(CameraComponent& camera) {
    glm::vec3 dir = {};
    dir.x = std::cos(radians(camera.yaw)) * std::cos(radians(camera.pitch));
    dir.y = std::sin(radians(camera.pitch));
    dir.z = std::sin(radians(camera.yaw)) * std::cos(radians(camera.pitch));

    camera.pos = camera.center - dir * camera.length;
}

void camera_system(gfx::FrameContext& fcx, World& world, float dt) {
    static constexpr float MOVE_SPEED = 5.f;

    for (auto [e, camera] : world.reg.view<CameraComponent>().each()) {
        const float dz = (glfwGetKey(fcx.cx.window, GLFW_KEY_W) | glfwGetKey(fcx.cx.window, GLFW_KEY_UP) - glfwGetKey(fcx.cx.window, GLFW_KEY_S) |
                             glfwGetKey(fcx.cx.window, GLFW_KEY_DOWN)) *
                         MOVE_SPEED * dt;
        const float dx = (glfwGetKey(fcx.cx.window, GLFW_KEY_D) | glfwGetKey(fcx.cx.window, GLFW_KEY_RIGHT) - glfwGetKey(fcx.cx.window, GLFW_KEY_A) |
                             glfwGetKey(fcx.cx.window, GLFW_KEY_LEFT)) *
                         MOVE_SPEED * dt;

        camera.center += glm::vec3{std::cos(radians(camera.yaw)), 0.f, std::sin(radians(camera.yaw))} * dz;
        camera.center += glm::normalize(glm::cross(camera.forward, camera.up)) * dx;

        set_camera_position(camera);
    }
}

void camera_look(gfx::Context& cx, World& world, float x, float y) {
    static constexpr float SENSITIVITY = 0.1f;

    for (auto [e, camera] : world.reg.view<CameraComponent>().each()) {
        if (glfwGetMouseButton(cx.window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (float_cmp(camera.last_x, -1.f) || float_cmp(camera.last_y, -1.f)) {
                camera.last_x = x;
                camera.last_y = y;
            }

            const float dx = (x - camera.last_x) * SENSITIVITY;
            const float dy = (y - camera.last_y) * SENSITIVITY;

            camera.yaw += dx;
            camera.pitch += dy;

            camera.pitch = std::min(std::max(camera.pitch, -89.f), 89.f);

            glm::vec3 dir = {};
            dir.x = std::cos(radians(camera.yaw)) * std::cos(radians(camera.pitch));
            dir.y = std::sin(radians(camera.pitch));
            dir.z = std::sin(radians(camera.yaw)) * std::cos(radians(camera.pitch));

            camera.forward = glm::normalize(dir);

            set_camera_position(camera);
        }

        camera.last_x = x;
        camera.last_y = y;
    }
}

void camera_zoom(gfx::Context& cx, World& world, float x, float y) {
    static constexpr float SENSITIVITY = 0.25f;

    for (auto [e, camera] : world.reg.view<CameraComponent>().each()) {
        camera.length = clamp(camera.length - y * SENSITIVITY, 2.f, 20.f);
        set_camera_position(camera);
    }
}

} // namespace world
