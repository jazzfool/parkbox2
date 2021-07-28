#pragma once

#include "indirect.hpp"

namespace gfx {

struct Context;
class FrameContext;

struct Scene final {
    struct Uniforms final {
        glm::vec4 cam_pos;
        glm::vec4 sun_dir;
        glm::vec4 sun_radiant_flux;
        glm::mat4 cam_proj;
        glm::mat4 cam_view;
    };

    void init(FrameContext& fcx);
    void cleanup(FrameContext& fcx);

    void update(FrameContext& fcx);

    IndirectStorage storage;
    IndirectMeshPass pass;
    Uniforms uniforms;
    Buffer ubo;
};

} // namespace gfx
