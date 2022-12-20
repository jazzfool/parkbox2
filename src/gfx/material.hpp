#pragma once

#include "pipeline_cache.hpp"
#include "deletion_queue.hpp"
#include "indirect.hpp"

#include <string_view>
#include <string>
#include <vector>

namespace gfx {

struct MaterialInstance final {
    uint32_t textures[8];
    float scalars[4];
    glm::vec4 vectors[4];
};

class MaterialShadingPass final {
  public:
    struct PassInfo final {
        PipelineHandle pipeline;
        IndirectMeshPass pass;
    };

    void init(Buffer ubo, std::string shader_template, PipelineInfo base);

    void insert(FrameContext& fcx, std::string name);
    IndirectMeshPass& pass(std::string_view name);
    const IndirectMeshPass& pass(std::string_view name) const;

    std::vector<PassInfo*> all();

    void prepare(FrameContext& fcx);

  private:
    Buffer ubo;

    std::string shader_template;
    PipelineInfo base;
    std::unordered_map<std::size_t, PassInfo> passes;
};

class MaterialPass final {
  public:
    struct Uniforms final {
        glm::vec4 frustum;
        glm::vec2 near_far;
        glm::mat4 view;
    };

    void init(FrameContext& fcx);
    void cleanup(FrameContext& fcx);

    void prepare(FrameContext& fcx);

    void insert(FrameContext& fcx, std::string_view name, std::string shader_template, PipelineInfo base);
    MaterialShadingPass& pass(std::string_view name);
    const MaterialShadingPass& pass(std::string_view name) const;

    Uniforms uniforms;

  private:
    std::unordered_map<std::size_t, MaterialShadingPass> passes;
    Buffer ubo;
};

} // namespace gfx