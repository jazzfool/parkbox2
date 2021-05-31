#pragma once

#include "gfx_pass.hpp"
#include "types.hpp"
#include "descriptor_cache.hpp"

#include <glm/mat4x4.hpp>

namespace gfx {

class PBRGraphicsPass final : public GFXPass {
  public:
    struct SceneUniforms {
        glm::vec4 cam_pos;
        glm::vec4 sun_dir;
        glm::vec4 sun_radiant_flux;
        glm::mat4 cam_proj;
        glm::mat4 cam_view;
    };

    void init(FrameContext& fcx) override;
    void cleanup(FrameContext& fcx) override;

    void add_resources(RenderGraph& rg) override;
    std::vector<RenderPass> pass(FrameContext& fcx) override;

    void render(FrameContext& fcx, const RenderGraph& rg, VkRenderPass rp);

    Texture out;
    Texture depth;

    SceneUniforms uniforms;

  private:
    Buffer ubo;
    Texture ec_dfg_lut;
    Texture ibl_dfg_lut;
    Texture prefilter;
    Texture irrad;

    DescriptorKey desc_key;
};

} // namespace gfx
