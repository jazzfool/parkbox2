#include "world.hpp"

#include "gfx/context.hpp"
#include "gfx/frame_context.hpp"
#include "gfx/vk_helpers.hpp"
#include "helpers.hpp"
#include "mesh.hpp"
#include "camera.hpp"
#include "gfx/renderer.hpp"

#include <fstream>
#include <spdlog/fmt/fmt.h>
#include <tiny_obj_loader.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/norm.hpp>
#include <imgui.h>

namespace world {

glm::vec4 v3norm(glm::vec4 p) {
    return p / glm::length(glm::vec3{p});
}

gfx::Texture load_local_texture(gfx::FrameContext& fcx, std::string_view file, bool mipped, VkFormat format) {
    std::ifstream f{fmt::format("{}/textures/{}", PK_RESOURCE_DIR, file), std::ios::binary};
    const std::vector<uint8_t> buf{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    f.close();

    gfx::ImageLoadInfo info;
    info.format = format;
    info.data = buf.data();
    info.data_size = buf.size();
    info.generate_mipmaps = mipped;

    const gfx::Image img = gfx::load_image(fcx, info);

    VkImageViewCreateInfo ivci = {};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = img.image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.components = gfx::vk_no_swizzle();
    ivci.format = img.format;
    ivci.subresourceRange = gfx::vk_subresource_range(0, 1, 0, img.num_mips, VK_IMAGE_ASPECT_COLOR_BIT);

    return gfx::create_texture(fcx.cx.dev, img, ivci);
}

void World::begin(gfx::FrameContext& fcx) {
    cx = &fcx.cx;

    fcx.cx.on_resize.connect_delegate(delegate<&World::set_perspective>(this));

    on_mouse_move = {fcx.cx.on_mouse_move, fcx.cx.on_mouse_move.connect_delegate<&World::mouse_move>(this)};
    on_scroll = {fcx.cx.on_scroll, fcx.cx.on_scroll.connect_delegate<&World::scroll>(this)};

    main_camera = spawn_camera(*this);
    set_perspective(fcx.cx.width, fcx.cx.height);

    fcx.cx.scene.passes.pass("pbr").insert(fcx, "pbr_textured");

    add_texture(fcx, "metal.albedo", "metal_albedo.png", true);
    add_texture(fcx, "metal.roughness", "metal_roughness.png", false, VK_FORMAT_R8G8B8A8_UNORM);
    add_texture(fcx, "metal.metallic", "metal_metallic.png", false, VK_FORMAT_R8G8B8A8_UNORM);
    add_texture(fcx, "metal.normal", "metal_normal.png", true, VK_FORMAT_R8G8B8A8_UNORM);
    add_texture(fcx, "metal.ao", "metal_ao.png", false, VK_FORMAT_R8G8B8A8_UNORM);
    add_texture(fcx, "floor", "floor.jpg", true);
    add_texture(fcx, "black", "black.jpg", false, VK_FORMAT_R8G8B8A8_UNORM);
    add_texture(fcx, "white", "white.jpg", false, VK_FORMAT_R8G8B8A8_UNORM);
    add_texture(fcx, "flat", "flat.jpg", false, VK_FORMAT_R8G8B8A8_UNORM);
    add_texture(fcx, "gray", "gray.jpg");
    add_texture(fcx, "purple", "purple.png");

    gfx::MaterialInstance mat = {};
    mat.textures[0] = texture("metal.albedo");
    mat.textures[1] = texture("metal.roughness");
    mat.textures[2] = texture("metal.metallic");
    mat.textures[3] = texture("metal.normal");
    mat.textures[4] = texture("metal.ao");

    gfx::MaterialInstance floor_mat = {};
    floor_mat.textures[0] = texture("floor");
    floor_mat.textures[1] = texture("white");
    floor_mat.textures[2] = texture("black");
    floor_mat.textures[3] = texture("flat");
    floor_mat.textures[4] = texture("white");

    gfx::MaterialInstance purple_mat;
    purple_mat.textures[0] = texture("purple");
    purple_mat.textures[1] = texture("white");
    purple_mat.textures[2] = texture("black");
    purple_mat.textures[3] = texture("flat");
    purple_mat.textures[4] = texture("white");

    add_material(fcx.cx, "metal", mat);
    add_material(fcx.cx, "floor", floor_mat);
    add_material(fcx.cx, "purple", purple_mat);
    add_static_mesh(fcx, "cube", "cube.obj");
    add_static_mesh(fcx, "plane", "plane.obj");
    add_static_mesh(fcx, "sphere", "sphere.obj");

    entt::entity floor = reg.create();

    MeshComponent mesh;
    mesh.uv_scale = {48.f, 48.f};
    mesh.material = material("floor");
    mesh.mesh = static_mesh("cube");
    mesh.shader_type = "pbr";
    mesh.shader = "pbr_textured";
    add_object(fcx.cx, "pbr", "pbr_textured", mesh);

    TransformComponent transform;
    transform.scale = {6.f, 0.1f, 6.f};

    reg.emplace<MeshComponent>(floor, mesh);
    reg.emplace<TransformComponent>(floor, transform);

    gpu_mesh_update(*this, floor);

    entt::entity ball = reg.create();

    MeshComponent ball_mesh;
    ball_mesh.uv_scale = {1.f, 1.f};
    ball_mesh.material = material("metal");
    ball_mesh.mesh = static_mesh("cube");
    ball_mesh.shader_type = "pbr";
    ball_mesh.shader = "pbr_textured";
    add_object(fcx.cx, "pbr", "pbr_textured", ball_mesh);

    TransformComponent ball_transform;
    ball_transform.scale = {1.f, 1.f, 1.f};
    ball_transform.pos = {0.f, -1.f, 0.f};

    reg.emplace<MeshComponent>(ball, ball_mesh);
    reg.emplace<TransformComponent>(ball, ball_transform);

    gpu_mesh_update(*this, ball);
}

void World::end(gfx::FrameContext& fcx) {
    for (auto& [name, sm] : static_meshes) {
        fcx.cx.scene.storage.free_vertices(sm.vertices);
        fcx.cx.scene.storage.free_indices(sm.indices);
    }
}

gfx::IndirectObjectHandle World::add_object(
    gfx::Context& cx, std::string_view shader_type, std::string_view shader, uint32_t material, gfx::IndirectMeshKey mesh, glm::vec2 uv_scale) {
    gfx::IndirectObject obj;
    obj.material = material;
    obj.transform = glm::identity<glm::mat4>();
    obj.mesh = mesh;
    obj.uv_scale = uv_scale;
    return cx.scene.passes.pass(shader_type).pass(shader).push_object(cx, obj);
}

void World::add_object(gfx::Context& cx, std::string_view shader_type, std::string_view shader, MeshComponent& mesh) {
    mesh.gpu_object = add_object(cx, shader_type, shader, mesh.material, mesh.mesh, mesh.uv_scale);
}

uint32_t World::add_texture(gfx::FrameContext& fcx, const std::string& name, std::string_view file, bool mipped, VkFormat format) {
    const gfx::Texture tex = load_local_texture(fcx, file, mipped, format);
    const uint32_t id = fcx.cx.scene.storage.push_texture(tex);

    textures.emplace(name, id);

    return id;
}

uint32_t World::add_material(gfx::Context& cx, const std::string& name, gfx::MaterialInstance mat) {
    const uint32_t id = cx.scene.storage.push_material(mat);
    materials.emplace(name, id);
    return id;
}

gfx::IndirectMeshKey World::add_static_mesh(gfx::FrameContext& fcx, const std::string& name, std::string_view file) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;

    std::ifstream f{fmt::format("{}/meshes/{}", PK_RESOURCE_DIR, file)};
    tinyobj::LoadObj(&attrib, &shapes, nullptr, nullptr, nullptr, &f, nullptr);
    f.close();

    glm::vec3 min{INFINITY, INFINITY, INFINITY};
    glm::vec3 max{-INFINITY, -INFINITY, -INFINITY};

    std::unordered_map<gfx::Vertex, uint32_t> unique_verts;

    std::vector<gfx::Vertex> vertices;
    std::vector<uint32_t> indices;

    for (const tinyobj::shape_t& shape : shapes) {
        for (const tinyobj::index_t& index : shape.mesh.indices) {
            gfx::Vertex vert;

            vert.position.x = attrib.vertices[3 * index.vertex_index];
            vert.position.y = attrib.vertices[3 * index.vertex_index + 1];
            vert.position.z = attrib.vertices[3 * index.vertex_index + 2];

            vert.normal.x = attrib.normals[3 * index.normal_index];
            vert.normal.y = attrib.normals[3 * index.normal_index + 1];
            vert.normal.z = attrib.normals[3 * index.normal_index + 2];

            vert.tex_coord.x = attrib.texcoords[2 * index.texcoord_index];
            vert.tex_coord.y = 1.f - attrib.texcoords[2 * index.texcoord_index + 1];

            if (unique_verts.count(vert) == 0) {
                unique_verts[vert] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vert);
            }

            indices.push_back(unique_verts[vert]);

            min = glm::min(min, vert.position);
            max = glm::max(max, vert.position);
        }
    }

    const glm::vec3 center = (min + max) / 2.f;
    const float radius = std::sqrt(std::max(glm::length2(center - min), glm::length2(center - max)));

    StaticMesh mesh{fcx.cx.scene.storage.allocate_vertices(vertices.size()), fcx.cx.scene.storage.allocate_indices(indices.size()), min, max};

    fcx.stage(mesh.vertices.buffer, vertices.data());
    fcx.stage(mesh.indices.buffer, indices.data());

    const gfx::IndirectMeshKey mk = gfx::indirect_mesh_key(mesh.vertices, mesh.indices);
    fcx.cx.scene.passes.pass("pbr").pass("pbr_textured").push_mesh(mk, center, radius);

    static_meshes.emplace(name, std::move(mesh));

    return mk;
}

uint32_t World::texture(const std::string& name) const {
    return textures.at(name);
}

uint32_t World::material(const std::string& name) const {
    return materials.at(name);
}

gfx::IndirectMeshKey World::static_mesh(const std::string& name) const {
    const StaticMesh& sm = static_meshes.at(name);
    return gfx::indirect_mesh_key(sm.vertices, sm.indices);
}

void World::ui() {
}

void World::update(gfx::FrameContext& fcx, float dt) {
    camera_system(fcx, *this, dt);

    const CameraComponent cam = reg.get<CameraComponent>(main_camera);
    fcx.cx.scene.uniforms.cam_pos = {cam.pos, 0.f};
    fcx.cx.scene.uniforms.cam_view = glm::lookAt(cam.pos, cam.pos + cam.forward, cam.up);
    fcx.cx.scene.uniforms.cam_proj = perspective * fcx.cx.scene.uniforms.cam_view;

    const glm::mat4 projtrans = glm::transpose(perspective);
    const glm::vec4 frustum_x = v3norm(projtrans[3] + projtrans[0]);
    const glm::vec4 frustum_y = v3norm(projtrans[3] + projtrans[1]);

    fcx.cx.scene.passes.uniforms.frustum[0] = frustum_x.x;
    fcx.cx.scene.passes.uniforms.frustum[1] = frustum_x.z;
    fcx.cx.scene.passes.uniforms.frustum[2] = frustum_y.y;
    fcx.cx.scene.passes.uniforms.frustum[3] = frustum_y.z;
    fcx.cx.scene.passes.uniforms.near_far = glm::vec2{0.1f, 100.f};
    fcx.cx.scene.passes.uniforms.view = fcx.cx.scene.uniforms.cam_view;
}

void World::mouse_move(double x, double y) {
    camera_look(*cx, *this, static_cast<float>(x), static_cast<float>(y));
}

void World::scroll(double x, double y) {
    camera_zoom(*cx, *this, static_cast<float>(x), static_cast<float>(y));
}

void World::set_perspective(int32_t w, int32_t h) {
    perspective = glm::perspective(glm::radians(60.f), static_cast<float>(w) / static_cast<float>(h), 0.1f, 100.f);
}

} // namespace world
