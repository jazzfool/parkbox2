#include "material.hpp"

#include "frame_context.hpp"
#include "context.hpp"

#include <fstream>
#include <spdlog/fmt/fmt.h>

namespace gfx {

void MaterialShadingPass::init(Buffer ubo, std::string shader_template, PipelineInfo base) {
    this->ubo = ubo;
    this->shader_template = std::move(shader_template);
    this->base = std::move(base);
}

void MaterialShadingPass::insert(FrameContext& fcx, std::string name) {
    std::string filename = fmt::format("{}.glsl", name);

    std::string path = fmt::format("{}/shaders/{}", PK_RESOURCE_DIR, filename);
    std::ifstream f;
    f.open(path.data(), std::ios::in | std::ios::binary);
    std::string shader{std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}};
    f.close();

    std::string src = shader_template;
    const std::string::size_type where = src.find("{...}");
    src.replace(where, 5, shader);

    fcx.cx.shader_cache.load_str(src, name, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipelineShaderStageCreateInfo pssci = {};
    pssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pssci.module = fcx.cx.shader_cache.get(name);
    pssci.pName = "main";
    pssci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;

    PipelineInfo pi = base;
    pi.shader_stages.push_back(pssci);

    PassInfo info{fcx.cx.pipeline_cache.add(name, pi)};
    info.pass.init(fcx, ubo);

    passes.emplace(std::hash<std::string>{}(name), std::move(info));
}

IndirectMeshPass& MaterialShadingPass::pass(std::string_view name) {
    return passes.at(std::hash<std::string_view>{}(name)).pass;
}

const IndirectMeshPass& MaterialShadingPass::pass(std::string_view name) const {
    return passes.at(std::hash<std::string_view>{}(name)).pass;
}

std::vector<MaterialShadingPass::PassInfo*> MaterialShadingPass::all() {
    std::vector<PassInfo*> out;
    for (auto& [key, pass] : passes) {
        out.push_back(&pass);
    }
    return out;
}

void MaterialShadingPass::prepare(FrameContext& fcx) {
    for (auto& [key, pass] : passes) {
        pass.pass.prepare(fcx);
    }
}

void MaterialPass::init(FrameContext& fcx) {
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bci.size = sizeof(Uniforms);
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    ubo = fcx.cx.alloc.create_buffer(bci, VMA_MEMORY_USAGE_CPU_TO_GPU, true);
}

void MaterialPass::cleanup(FrameContext& fcx) {
    fcx.cx.alloc.destroy(ubo);
}

void MaterialPass::prepare(FrameContext& fcx) {
    vk_mapped_write(fcx.cx.alloc, ubo, &uniforms, sizeof(Uniforms));

    for (auto& [key, pass] : passes) {
        pass.prepare(fcx);
    }
}

void MaterialPass::insert(FrameContext& fcx, std::string_view name, std::string shader_template, PipelineInfo base) {
    MaterialShadingPass pass;
    pass.init(ubo, shader_template, base);
    passes.emplace(std::hash<std::string_view>{}(name), std::move(pass));
}

MaterialShadingPass& MaterialPass::pass(std::string_view name) {
    return passes.at(std::hash<std::string_view>{}(name));
}

const MaterialShadingPass& MaterialPass::pass(std::string_view name) const {
    return passes.at(std::hash<std::string_view>{}(name));
}

} // namespace gfx
