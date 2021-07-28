#include "scene.hpp"

#include "context.hpp"
#include "frame_context.hpp"

namespace gfx {

void Scene::init(FrameContext& fcx) {
    pass.init(fcx);
    storage.init(fcx);

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.size = sizeof(Uniforms);
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    ubo = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_GPU_ONLY, false);
}

void Scene::cleanup(FrameContext& fcx) {
    fcx.cx.alloc.destroy(ubo);

    storage.cleanup(fcx);
    pass.cleanup(fcx);
}

void Scene::update(FrameContext& fcx) {
    storage.update(fcx);
    pass.prepare(fcx);
    fcx.stage(ubo, &uniforms);
}

} // namespace gfx
