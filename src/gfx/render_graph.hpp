#pragma once

#include "types.hpp"
#include "vk_helpers.hpp"

#include <vector>
#include <string>
#include <volk.h>
#include <variant>
#include <tuple>
#include <optional>
#include <functional>
#include <unordered_map>

namespace gfx {

class FrameContext;
struct Context;

struct PassAttachment final {
    Texture tex;
    VkImageSubresourceRange subresource;
};

struct PassBuffer final {
    Buffer buffer;
};

struct Name final {
    std::string name;
};

inline bool operator==(const Name& a, const Name& b) {
    return a.name == b.name;
}

template <typename T>
inline bool operator==(const std::pair<Name, T>& a, const Name& b) {
    return a.first == b;
}

} // namespace gfx

namespace std {

template <>
struct hash<gfx::Name> {
    std::size_t operator()(const gfx::Name& name) const;
};

} // namespace std

namespace gfx {

class RenderPass final {
  public:
    void set_depth_stencil(Name name, std::optional<VkClearDepthStencilValue> clear);
    void push_color_output(Name name, std::optional<VkClearColorValue> clear);
    void push_resolve_output(Name name, std::optional<VkClearColorValue> clear);
    void push_input_attachment(Name name, bool self, std::optional<VkClearColorValue> clear);
    void push_texture_input(Name name);
    void push_dependency(Name name, VkImageLayout layout, VkPipelineStageFlags stage, VkAccessFlags access);
    void push_dependent(Name name, VkImageLayout layout, VkPipelineStageFlags stage, VkAccessFlags access);
    void push_virtual_dependent(Name name);

    void set_exec(std::function<void(FrameContext&, const class RenderGraph&, VkRenderPass)> exec);

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t layers = 0;

  private:
    friend class RenderGraph;

    struct Dependency final {
        VkImageLayout layout;
        VkPipelineStageFlags stage;
        VkAccessFlags access;
    };

    std::optional<std::pair<Name, std::optional<VkClearDepthStencilValue>>> depth_stencil;
    std::vector<std::pair<Name, std::optional<VkClearColorValue>>> color_outputs;
    std::vector<std::pair<Name, std::optional<VkClearColorValue>>> resolve_outputs;
    std::vector<std::tuple<Name, bool, std::optional<VkClearColorValue>>> input_attachments;
    std::vector<Name> texture_inputs;
    std::vector<std::pair<Name, Dependency>> dependencies;
    std::vector<std::pair<Name, Dependency>> dependents;
    std::vector<Name> virtual_dependents;

    std::function<void(FrameContext&, const class RenderGraph&, VkRenderPass)> exec;
};

class RenderGraphCache {
  public:
    void init(VkDevice dev);
    void cleanup();

    void clear();

    VkRenderPass create_pass(const VkRenderPassCreateInfo& rpci);
    VkFramebuffer create_framebuffer(const VkFramebufferCreateInfo& fbci);

  private:
    friend class RenderGraph;

    VkDevice dev;

    std::unordered_map<std::size_t, VkRenderPass> passes;
    std::unordered_map<std::size_t, VkFramebuffer> framebuffers;
};

// Render graphs are a handy abstraction for automatically handling synchronization between render passes.
class RenderGraph final {
  public:
    void push_pass(RenderPass pass);

    void push_attachment(Name name, PassAttachment attachment);
    void push_buffer(Name name, PassBuffer buffer);
    void push_initial_layout(Name name, VkImageLayout layout);

    PassAttachment attachment(Name name) const;
    PassBuffer buffer(Name name) const;

    void set_output(Name name, VkImageLayout layout);

    void exec(FrameContext& fcx, RenderGraphCache& cache);

  private:
    std::vector<std::size_t> find_all(std::function<bool(const RenderPass&)> pred) const;
    void push_writers(std::vector<std::size_t>& writers, Name res) const;

    std::unordered_map<Name, PassAttachment> attachments;
    std::unordered_map<Name, PassBuffer> buffers;
    std::unordered_map<Name, VkImageLayout> initial_layouts;
    std::vector<RenderPass> passes;
    Name output;
    VkImageLayout output_layout;
};

} // namespace gfx
