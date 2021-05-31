#pragma once

#include "types.hpp"
#include "cache.hpp"
#include "allocator.hpp"
#include "descriptor_cache.hpp"

#include <unordered_map>
#include <optional>
#include <glm/mat4x4.hpp>

namespace gfx {

struct IndirectMeshKey final {
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t num_indices;

    bool operator==(const IndirectMeshKey& other) const noexcept;
};

} // namespace gfx

namespace std {

template <>
struct hash<gfx::IndirectMeshKey> {
    std::size_t operator()(const gfx::IndirectMeshKey& key) const;
};

} // namespace std

namespace gfx {

class FrameContext;

struct IndirectMaterial final {
    uint32_t albedo;
    uint32_t roughness;
    uint32_t metallic;
    uint32_t normal;
    uint32_t ao;
};

struct IndirectObject final {
    glm::mat4 transform;
    uint32_t material;
    IndirectMeshKey mesh;
    glm::vec2 uv_scale;
};

uint32_t indirect_vertex_offset(const BufferAllocation& buf);
uint32_t indirect_index_offset(const BufferAllocation& buf);
uint32_t indirect_num_indices(const BufferAllocation& buf);

IndirectMeshKey indirect_mesh_key(const BufferAllocation& vertices, const BufferAllocation& indices);

class IndirectStorage final {
  public:
    static constexpr inline uint32_t MAX_MESHES = 1024;
    static constexpr inline uint32_t MAX_VERTICES_PER_MESH = 32000;
    static constexpr inline uint32_t MAX_INDICES_PER_MESH = 64000;
    static constexpr inline uint32_t MAX_MATERIALS = 512;

    void init(FrameContext& fcx);
    void cleanup(FrameContext& fcx);

    void update(FrameContext& fcx);

    uint32_t push_texture(Texture tex);
    uint32_t push_material(IndirectMaterial mat);

    BufferAllocation allocate_vertices(uint64_t num_verts);
    void free_vertices(BufferAllocation& alloc);

    BufferAllocation allocate_indices(uint64_t num_inds);
    void free_indices(BufferAllocation& alloc);

    Buffer vertex_buffer() const;
    Buffer index_buffer() const;
    Buffer material_buffer() const;

    const std::vector<Texture>& get_textures() const;

  private:
    VkDevice dev;

    std::optional<BufferArena<FreeListAllocator>> vx_arena;
    std::optional<BufferArena<FreeListAllocator>> ix_arena;

    Buffer material_buf;
    Buffer material_staging;

    std::vector<Texture> textures;
    std::vector<IndirectMaterial> mats;

    bool dirty;
};

struct IndirectObjectHandle final {
    Cache<std::pair<IndirectObject, std::size_t>>::Handle handle;
    IndirectMeshKey mesh;
};

class IndirectMeshPass final {
  public:
    static constexpr inline uint32_t MAX_OBJECTS = 65536;

    struct Uniforms final {
        glm::vec4 frustum;
        glm::vec2 near_far;
        glm::mat4 view;
    };

    void init(FrameContext& fcx);
    void cleanup(FrameContext& fcx);

    void push_mesh(IndirectMeshKey mesh, glm::vec4 sphere_bounds);

    IndirectObjectHandle push_object(IndirectObject obj);
    bool remove_object(IndirectObjectHandle h);
    void update_object(IndirectObjectHandle h);
    IndirectObject& object(IndirectObjectHandle h);
    const IndirectObject& object(IndirectObjectHandle h) const;

    void prepare(FrameContext& fcx);
    void execute(VkCommandBuffer cmd, const IndirectStorage& storage);

    Buffer instance_buffer() const;
    Buffer instance_indices_buffer() const;

    Uniforms uniforms;

  private:
    struct GPUInstance final {
        glm::mat4 transform;
        uint32_t material;
        int32_t batch_idx;
        glm::vec2 uv_scale;
        glm::vec4 bounds;
    };

    VkDescriptorSet set;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkEvent event;

    Buffer instance_buf;
    Buffer instance_staging;
    Buffer instance_indices_buf;
    Buffer draw_cmds;
    Buffer draw_staging;
    Buffer ubo;
    std::unordered_map<IndirectMeshKey, Cache<std::pair<IndirectObject, std::size_t>>> batches; // 1 batch = 1 draw = 1 mesh = n instances
    std::unordered_map<IndirectMeshKey, glm::vec4> mesh_bounds;
    std::vector<IndirectMeshKey> batch_list;
    std::vector<GPUInstance> instances;
};

} // namespace gfx
