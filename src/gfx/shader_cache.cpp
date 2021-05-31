#include "shader_cache.hpp"

#include "vk_helpers.hpp"
#include "context.hpp"

#include <fstream>
#include <shaderc/shaderc.hpp>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

namespace gfx {

struct FileIncluder : public shaderc::CompileOptions::IncluderInterface {
    struct Data {
        std::string name;
        std::string content;
    };

    shaderc_include_result* GetInclude(const char* requested_source, shaderc_include_type type, const char* requesting_source, size_t include_depth) override {
        Data* data = new Data;
        data->name = fmt::format("{}/shaders/{}", PK_RESOURCE_DIR, requested_source);

        std::ifstream f;
        f.open(data->name);
        std::string src{std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}};
        f.close();

        data->content = std::move(src);

        shaderc_include_result* out = new shaderc_include_result;

        out->source_name = data->name.c_str();
        out->source_name_length = data->name.length();
        out->content = data->content.c_str();
        out->content_length = data->content.length();
        out->user_data = data;

        return out;
    }

    void ReleaseInclude(shaderc_include_result* data) override {
        delete static_cast<Data*>(data->user_data);
        delete data;
    }
};

shaderc_shader_kind vk_to_shaderc_stage(VkShaderStageFlags stage) {
    switch (stage) {
    case VK_SHADER_STAGE_VERTEX_BIT:
        return shaderc_glsl_vertex_shader;
    case VK_SHADER_STAGE_FRAGMENT_BIT:
        return shaderc_glsl_fragment_shader;
    case VK_SHADER_STAGE_COMPUTE_BIT:
        return shaderc_glsl_compute_shader;
    default:
        return (shaderc_shader_kind)0;
    }
}

void ShaderCache::init(Context& cx) {
    this->cx = &cx;
}

void ShaderCache::cleanup() {
    for (const auto& [_, shader] : cache) {
        vkDestroyShaderModule(cx->dev, shader, nullptr);
    }
}

void ShaderCache::load(const std::string& name, VkShaderStageFlags stage) {
    std::string path = fmt::format("{}/shaders/{}", PK_RESOURCE_DIR, name);

    static shaderc::Compiler compiler;

    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
    options.SetIncluder(std::make_unique<FileIncluder>());

    std::ifstream f;
    f.open(path.data(), std::ios::in | std::ios::binary);
    std::string src{std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}};
    f.close();

    auto out = compiler.CompileGlslToSpv(src, vk_to_shaderc_stage(stage), name.data(), options);

    if (out.GetCompilationStatus() != shaderc_compilation_status_success) {
        spdlog::error("failed to compile shader {}: {}", path, out.GetErrorMessage());
        return;
    }

    std::vector<uint32_t> spirv{out.cbegin(), out.cend()};

    VkShaderModuleCreateInfo smci = {};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.pCode = spirv.data();
    smci.codeSize = spirv.size() * sizeof(uint32_t);

    VkShaderModule shader;
    vk_log(vkCreateShaderModule(cx->dev, &smci, nullptr, &shader));

    cache.emplace(name, shader);
}

VkShaderModule ShaderCache::get(const std::string& name) {
    return cache.at(name);
}

bool ShaderCache::contains(const std::string& name) const {
    return cache.count(name);
}

} // namespace gfx
