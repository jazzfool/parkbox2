#include "indirect.hpp"

#include "mesh.hpp"
#include "frame_context.hpp"
#include "context.hpp"
#include "def.hpp"
#include "helpers.hpp"

#include <array>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/norm.hpp>

bool gfx::IndirectMeshKey::operator==(const IndirectMeshKey& other) const noexcept {
    return vertex_offset == other.vertex_offset && index_offset == other.index_offset && num_indices == other.num_indices;
}

std::size_t std::hash<gfx::IndirectMeshKey>::operator()(const gfx::IndirectMeshKey& key) const {
    std::size_t h = 0;
    hash_combine(h, key.vertex_offset, key.index_offset, key.num_indices);
    return h;
}

namespace gfx {

uint32_t indirect_vertex_offset(const BufferAllocation& buf) {
    return buf->offset / sizeof(Vertex);
}

uint32_t indirect_index_offset(const BufferAllocation& buf) {
    return buf->offset / sizeof(uint32_t);
}

uint32_t indirect_num_indices(const BufferAllocation& buf) {
    return buf->size / sizeof(uint32_t);
}

IndirectMeshKey indirect_mesh_key(const BufferAllocation& vertices, const BufferAllocation& indices) {
    IndirectMeshKey key;
    key.vertex_offset = indirect_vertex_offset(vertices);
    key.index_offset = indirect_index_offset(indices);
    key.num_indices = indirect_num_indices(indices);
    return key;
}

void IndirectStorage::init(FrameContext& fcx) {
    dev = fcx.cx.dev;
    dirty = true;

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    bci.size = MAX_MESHES * MAX_VERTICES_PER_MESH * sizeof(Vertex);
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vx_arena = fcx.cx.alloc.create_arena(FreeListAllocator{MAX_MESHES * MAX_VERTICES_PER_MESH * sizeof(Vertex)}, bci, VMA_MEMORY_USAGE_GPU_ONLY, false);

    bci.size = MAX_MESHES * MAX_INDICES_PER_MESH * sizeof(uint32_t);
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    ix_arena = fcx.cx.alloc.create_arena(FreeListAllocator{MAX_MESHES * MAX_INDICES_PER_MESH * sizeof(uint32_t)}, bci, VMA_MEMORY_USAGE_GPU_ONLY, false);

    bci.size = MAX_MATERIALS * sizeof(IndirectMaterial);
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    material_buf = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_GPU_ONLY, false);

    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    material_staging = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_ONLY, true);
}

void IndirectStorage::cleanup(FrameContext& fcx) {
    fcx.cx.alloc.destroy(*vx_arena);
    fcx.cx.alloc.destroy(*ix_arena);
    fcx.cx.alloc.destroy(material_buf);
    fcx.cx.alloc.destroy(material_staging);

    for (const Texture& tex : textures) {
        destroy_texture(fcx.cx, tex);
    }
}

void IndirectStorage::update(FrameContext& fcx) {
    if (dirty) {
        dirty = false;

        ::memcpy(material_staging.pmap, mats.data(), sizeof(IndirectMaterial) * mats.size());
        vk_log(vmaFlushAllocation(fcx.cx.alloc.allocator, material_staging.allocation, material_staging.offset, sizeof(IndirectMaterial) * mats.size()));

        fcx.copy(material_staging, material_buf);

        const VkBufferMemoryBarrier barrier = vk_buffer_barrier(material_buf);
        vkCmdPipelineBarrier(fcx.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    }
}

uint32_t IndirectStorage::push_texture(Texture tex) {
    textures.push_back(tex);
    return textures.size() - 1;
}

uint32_t IndirectStorage::push_material(IndirectMaterial mat) {
    dirty = true;
    mats.push_back(mat);
    return mats.size() - 1;
}

BufferAllocation IndirectStorage::allocate_vertices(uint64_t num_verts) {
    return vx_arena->alloc(num_verts * sizeof(Vertex));
}

void IndirectStorage::free_vertices(const BufferAllocation& alloc) {
    vx_arena->free(alloc);
}

BufferAllocation IndirectStorage::allocate_indices(uint64_t num_inds) {
    return ix_arena->alloc(num_inds * sizeof(uint32_t));
}

void IndirectStorage::free_indices(const BufferAllocation& alloc) {
    ix_arena->free(alloc);
}

Buffer IndirectStorage::vertex_buffer() const {
    return vx_arena->buffer;
}

Buffer IndirectStorage::index_buffer() const {
    return ix_arena->buffer;
}

Buffer IndirectStorage::material_buffer() const {
    return material_buf;
}

const std::vector<Texture>& IndirectStorage::get_textures() const {
    return textures;
}

void IndirectMeshPass::init(FrameContext& fcx) {
    load_shader(fcx.cx.shader_cache, "cull.comp", VK_SHADER_STAGE_COMPUTE_BIT);

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    bci.size = MAX_OBJECTS * sizeof(GPUInstance);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    instance_buf = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_GPU_ONLY, false);

    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    instance_staging = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_ONLY, true);

    bci.size = MAX_OBJECTS * sizeof(uint32_t);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    instance_indices_buf = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_GPU_ONLY, false);

    bci.size = IndirectStorage::MAX_MESHES * sizeof(VkDrawIndexedIndirectCommand);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    draw_cmds = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_GPU_ONLY, false);

    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    draw_staging = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_ONLY, true);

    bci.size = sizeof(Uniforms);
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    ubo = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_TO_GPU, true);

    DescriptorKey desc_key;

    DescriptorSetInfo set_info;
    set_info.bind_buffer(draw_cmds, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set_info.bind_buffer(instance_buf, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set_info.bind_buffer(instance_indices_buf, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    set_info.bind_buffer(ubo, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    DescriptorSet set = fcx.cx.descriptor_cache.get_set(desc_key, set_info);

    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set.layout;

    this->set = set.set;

    vk_log(vkCreatePipelineLayout(fcx.cx.dev, &plci, nullptr, &layout));

    VkShaderModule shader = fcx.cx.shader_cache.get("cull.comp");

    VkComputePipelineCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.layout = layout;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.pName = "main";
    cpci.stage.module = shader;

    vk_log(vkCreateComputePipelines(fcx.cx.dev, nullptr, 1, &cpci, nullptr, &pipeline));

    VkEventCreateInfo eci = {};
    eci.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;

    vk_log(vkCreateEvent(fcx.cx.dev, &eci, nullptr, &event));
}

void IndirectMeshPass::cleanup(FrameContext& fcx) {
    fcx.cx.alloc.destroy(instance_buf);
    fcx.cx.alloc.destroy(instance_staging);
    fcx.cx.alloc.destroy(instance_indices_buf);
    fcx.cx.alloc.destroy(draw_cmds);
    fcx.cx.alloc.destroy(draw_staging);
    fcx.cx.alloc.destroy(ubo);

    vkDestroyPipeline(fcx.cx.dev, pipeline, nullptr);
    vkDestroyPipelineLayout(fcx.cx.dev, layout, nullptr);
    vkDestroyEvent(fcx.cx.dev, event, nullptr);
}

void IndirectMeshPass::push_mesh(IndirectMeshKey mesh, glm::vec3 center, float radius) {
    batches.emplace(mesh, Cache<std::pair<IndirectObject, std::size_t>>{});
    batch_list.push_back(mesh);
    mesh_bounds.emplace(mesh, std::make_pair(center, radius));
}

void IndirectMeshPass::update_mesh(IndirectMeshKey old_mesh, IndirectMeshKey new_mesh, glm::vec3 center, float radius) {
    Cache<std::pair<IndirectObject, std::size_t>> val = batches.at(old_mesh);
    batches.erase(old_mesh);
    batches.emplace(new_mesh, val);

    *std::find(batch_list.begin(), batch_list.end(), old_mesh) = new_mesh;

    mesh_bounds.erase(old_mesh);
    mesh_bounds.emplace(new_mesh, std::make_pair(center, radius));
}

IndirectObjectHandle IndirectMeshPass::push_object(Context& cx, IndirectObject obj) {
    GPUInstance instance;
    instance.transform = obj.transform;
    instance.material = obj.material;
    instance.uv_scale = obj.uv_scale;
    instance.batch_idx = std::find(batch_list.begin(), batch_list.end(), obj.mesh) - batch_list.begin();
    instances.push_back(instance);

    const IndirectObjectHandle h = {batches.at(obj.mesh).push(std::make_pair(obj, instances.size() - 1)), obj.mesh};

    update_object(cx, h); // calculate bounds

    return h;
}

bool IndirectMeshPass::remove_object(IndirectObjectHandle h) {
    if (batches.count(h.mesh)) {
        // TODO(jazzfool): free list to reuse removed instances for new ones
        instances.at(batches.at(h.mesh).get(h.handle).second).batch_idx = -1;
        return batches.at(h.mesh).remove(h.handle);
    } else {
        return false;
    }
}

void IndirectMeshPass::update_object(Context& cx, IndirectObjectHandle h) {
    const auto& [instance, idx] = batches.at(h.mesh).get(h.handle);
    instances[idx].transform = instance.transform;
    instances[idx].material = instance.material;
    instances[idx].uv_scale = instance.uv_scale;

    auto [center, radius] = mesh_bounds.at(h.mesh);

    center = instance.transform * glm::vec4{center, 1.f};
    radius *= std::sqrt(glm::compMax(glm::vec3{
        glm::length2(glm::vec3{instance.transform[0]}),
        glm::length2(glm::vec3{instance.transform[1]}),
        glm::length2(glm::vec3{instance.transform[2]}),
    }));

    instances[idx].bounds = glm::vec4{center, radius};

    vk_mapped_write(cx.alloc, instance_staging.slice(sizeof(GPUInstance) * idx, sizeof(GPUInstance)), &instances[idx], sizeof(GPUInstance));

    if (instance_updates.count(idx) == 0) {
        instance_updates.insert(idx);

        BufferCopy copy;
        copy.src_offset = sizeof(GPUInstance) * idx;
        copy.dst_offset = copy.src_offset;
        copy.size = sizeof(GPUInstance);

        instance_writes.push_back(copy);
    }
}

IndirectObject& IndirectMeshPass::object(IndirectObjectHandle h) {
    return batches.at(h.mesh).get(h.handle).first;
}

const IndirectObject& IndirectMeshPass::object(IndirectObjectHandle h) const {
    return batches.at(h.mesh).get(h.handle).first;
}

void IndirectMeshPass::prepare(FrameContext& fcx) {
    std::vector<VkDrawIndexedIndirectCommand> draws;
    draws.reserve(batches.size());

    uint32_t instance_start = 0;
    for (const auto& batch : batch_list) {
        VkDrawIndexedIndirectCommand draw = {};
        draw.firstIndex = batch.index_offset;
        draw.firstInstance = instance_start;
        draw.instanceCount = 0;
        draw.vertexOffset = batch.vertex_offset;
        draw.indexCount = batch.num_indices;

        draws.push_back(draw);

        instance_start += batches[batch].all().size();
    }

    vk_mapped_write(fcx.cx.alloc, draw_staging, draws.data(), sizeof(VkDrawIndexedIndirectCommand) * draws.size());
    fcx.copy(draw_staging, draw_cmds);

    if (!instance_writes.empty()) {
        fcx.multicopy(instance_staging, instance_buf, instance_writes);
        instance_writes.clear();
        instance_updates.clear();
    }

    vk_mapped_write(fcx.cx.alloc, ubo, &uniforms, sizeof(Uniforms));

    const std::array<VkBufferMemoryBarrier, 2> barriers = {vk_buffer_barrier(draw_cmds), vk_buffer_barrier(instance_buf)};
    vkCmdPipelineBarrier(
        fcx.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, barriers.size(), barriers.data(), 0, nullptr);

    vkCmdBindPipeline(fcx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(fcx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(fcx.cmd, instances.size(), 1, 1);

    vkCmdSetEvent(fcx.cmd, event, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void IndirectMeshPass::execute(VkCommandBuffer cmd, const IndirectStorage& storage) {
    const std::array<VkBufferMemoryBarrier, 2> barriers = {vk_buffer_barrier(draw_cmds), vk_buffer_barrier(instance_indices_buf)};
    vkCmdWaitEvents(
        cmd, 1, &event, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, nullptr, barriers.size(), barriers.data(), 0, nullptr);

    const VkDeviceSize vx_offset = storage.vertex_buffer().offset;
    const Buffer vx_buffer = storage.vertex_buffer();

    vkCmdBindVertexBuffers(cmd, 0, 1, &vx_buffer.buffer, &vx_offset);
    vkCmdBindIndexBuffer(cmd, storage.index_buffer().buffer, storage.index_buffer().offset, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexedIndirect(cmd, draw_cmds.buffer, draw_cmds.offset, batches.size(), sizeof(VkDrawIndexedIndirectCommand));
}

Buffer IndirectMeshPass::instance_buffer() const {
    return instance_buf;
}

Buffer IndirectMeshPass::instance_indices_buffer() const {
    return instance_indices_buf;
}

} // namespace gfx
