#pragma once

#include "gfx/allocator.hpp"
#include "gfx/indirect.hpp"
#include "gfx/mesh.hpp"
#include "signal.hpp"
#include "plant.hpp"

#include <unordered_map>
#include <string>
#include <string_view>
#include <entt/entt.hpp>

namespace gfx {
struct Context;
class FrameContext;
struct IndirectMaterial;
} // namespace gfx

namespace world {

class World final {
  public:
    struct StaticMesh final {
        gfx::BufferAllocation vertices;
        gfx::BufferAllocation indices;
        glm::vec3 min;
        glm::vec3 max;
    };

    void begin(gfx::FrameContext& fcx);
    void end(gfx::FrameContext& fcx);

    gfx::IndirectObjectHandle add_object(gfx::Context& cx, uint32_t material, gfx::IndirectMeshKey mesh, glm::vec2 uv_scale);
    void add_object(gfx::Context& cx, struct MeshComponent& mesh);
    uint32_t add_texture(
        gfx::FrameContext& fcx, const std::string& name, std::string_view file, bool mipped = false, VkFormat format = VK_FORMAT_R8G8B8A8_SRGB);
    uint32_t add_material(gfx::Context& cx, const std::string& name, gfx::IndirectMaterial mat);
    gfx::IndirectMeshKey add_static_mesh(gfx::FrameContext& fcx, const std::string& name, std::string_view file);

    uint32_t texture(const std::string& name) const;
    uint32_t material(const std::string& name) const;
    gfx::IndirectMeshKey static_mesh(const std::string& name) const;

    void ui();
    void update(gfx::FrameContext& fcx, float dt);

    entt::registry reg;

    entt::entity main_camera;
    glm::mat4 perspective;

    gfx::Context* cx;

  private:
    void mouse_move(double x, double y);
    void scroll(double x, double y);

    void set_perspective(int32_t w, int32_t h);

    std::unordered_map<std::string, uint32_t> textures;
    std::unordered_map<std::string, uint32_t> materials;
    std::unordered_map<std::string, StaticMesh> static_meshes;

    ScopedSignalListener<double, double> on_mouse_move;
    ScopedSignalListener<double, double> on_scroll;

    PlantEnvironment env;
};

} // namespace world
