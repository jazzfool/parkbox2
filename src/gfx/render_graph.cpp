#include "render_graph.hpp"

#include "def.hpp"
#include "helpers.hpp"
#include "frame_context.hpp"
#include "context.hpp"

#include <unordered_set>

std::size_t std::hash<gfx::Name>::operator()(const gfx::Name& name) const {
    std::size_t h = 0;
    hash_combine(h, name.name);
    return h;
}

namespace gfx {

void RenderPass::set_depth_stencil(Name name, std::optional<VkClearDepthStencilValue> clear) {
    depth_stencil = std::make_pair(name, clear);
}

void RenderPass::push_color_output(Name name, std::optional<VkClearColorValue> clear) {
    color_outputs.push_back(std::make_pair(name, clear));
}

void RenderPass::push_resolve_output(Name name, std::optional<VkClearColorValue> clear) {
    resolve_outputs.push_back(std::make_pair(name, clear));
}

void RenderPass::push_input_attachment(Name name, bool self, std::optional<VkClearColorValue> clear) {
    input_attachments.push_back(std::make_tuple(name, self, clear));
}

void RenderPass::push_texture_input(Name name) {
    texture_inputs.push_back(name);
}

void RenderPass::push_dependency(Name name, VkImageLayout layout, VkPipelineStageFlags stage, VkAccessFlags access, bool virt) {
    dependencies.push_back(std::make_pair(name, Dependency{layout, stage, access, virt}));
}

void RenderPass::push_dependent(Name name, VkImageLayout layout, VkPipelineStageFlags stage, VkAccessFlags access, bool virt) {
    dependents.push_back(std::make_pair(name, Dependency{layout, stage, access, virt}));
}

void RenderPass::set_pre_exec(std::function<void(FrameContext&, const RenderGraph&, VkRenderPass)> pre_exec) {
    this->pre_exec = std::move(pre_exec);
}

void RenderPass::set_exec(std::function<void(FrameContext&, const RenderGraph&, VkRenderPass)> exec) {
    this->exec = std::move(exec);
}

void RenderGraphCache::init(VkDevice dev) {
    this->dev = dev;
}

void RenderGraphCache::cleanup() {
    clear();
}

void RenderGraphCache::clear() {
    for (const auto& [hash, pass] : passes) {
        vkDestroyRenderPass(dev, pass, nullptr);
    }

    for (const auto& [hash, fb] : framebuffers) {
        vkDestroyFramebuffer(dev, fb, nullptr);
    }

    passes.clear();
    framebuffers.clear();
}

VkRenderPass RenderGraphCache::create_pass(const VkRenderPassCreateInfo& rpci) {
    const std::size_t h = std::hash<VkRenderPassCreateInfo>{}(rpci);

    if (passes.count(h)) {
        return passes.at(h);
    }

    VkRenderPass rp;
    vk_log(vkCreateRenderPass(dev, &rpci, nullptr, &rp));
    passes.emplace(h, rp);
    return rp;
}

VkFramebuffer RenderGraphCache::create_framebuffer(const VkFramebufferCreateInfo& fbci) {
    const std::size_t h = std::hash<VkFramebufferCreateInfo>{}(fbci);

    if (framebuffers.count(h)) {
        return framebuffers.at(h);
    }

    VkFramebuffer fb;
    vk_log(vkCreateFramebuffer(dev, &fbci, nullptr, &fb));
    framebuffers.emplace(h, fb);
    return fb;
}

void RenderGraph::push_pass(RenderPass pass) {
    passes.push_back(std::move(pass));
}

void RenderGraph::push_attachment(Name name, PassAttachment attachment) {
    attachments.emplace(name, attachment);
}

void RenderGraph::push_buffer(Name name, PassBuffer buffer) {
    buffers.emplace(name, buffer);
}

void RenderGraph::push_initial_layout(Name name, VkImageLayout layout) {
    initial_layouts.emplace(name, layout);
}

PassAttachment RenderGraph::attachment(Name name) const {
    return attachments.at(name);
}

PassBuffer RenderGraph::buffer(Name name) const {
    return buffers.at(name);
}

void RenderGraph::set_output(Name name, VkImageLayout layout) {
    output = name;
    output_layout = layout;
}

void RenderGraph::exec(FrameContext& fcx, RenderGraphCache& cache) {
    // Validation
    for (const RenderPass& pass : passes) {
        if (pass.depth_stencil.has_value())
            PK_ASSERT(attachments.count(pass.depth_stencil.value().first));

        for (const auto& [res, _] : pass.color_outputs)
            PK_ASSERT(attachments.count(res));

        for (const auto& [res, _] : pass.resolve_outputs)
            PK_ASSERT(attachments.count(res));

        for (const auto& [res, _a, _b] : pass.input_attachments)
            PK_ASSERT(attachments.count(res));

        for (const Name& res : pass.texture_inputs)
            PK_ASSERT(attachments.count(res));

        for (const auto& [res, _] : pass.dependencies)
            PK_ASSERT(attachments.count(res));

        for (const auto& [res, _] : pass.dependents)
            PK_ASSERT(attachments.count(res));
    }

    // Bottom-up dependency traversal
    std::vector<std::size_t> pass_list;
    push_writers(pass_list, output);

    // Reverse ordering
    std::reverse(pass_list.begin(), pass_list.end());

    // Prune duplicates
    std::unordered_set<std::size_t> seen_passes;
    pass_list.erase(std::remove_if(pass_list.begin(), pass_list.end(),
                        [&seen_passes](std::size_t i) {
                            if (seen_passes.count(i))
                                return true;
                            seen_passes.insert(i);
                            return false;
                        }),
        pass_list.end());

    // Find the passes that aren't in the list and push them onto the end
    // This is mostly fine since passes can only be left out if they don't write to anything (but presumably read with side effects).
    for (std::size_t i = 0; i < passes.size(); ++i) {
        if (!seen_passes.count(i))
            pass_list.push_back(i);
    }

    struct Attachment final {
        PassAttachment attachment;
        VkImageLayout layout;
        VkPipelineStageFlags stage;
        VkAccessFlags access;
    };

    // Track resources
    std::unordered_map<Name, Attachment> tracked_attachments;
    tracked_attachments.reserve(attachments.size());

    for (const auto& [name, attachment] : attachments) {
        Attachment tracked;
        tracked.attachment = attachment;
        tracked.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        tracked.stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        tracked.access = 0;

        if (initial_layouts.count(name)) {
            tracked.layout = initial_layouts.at(name);
        }

        tracked_attachments.emplace(name, tracked);
    }

    // Execute
    for (const size_t i : pass_list) {
        const RenderPass& pass = passes[i];

        std::vector<VkImageMemoryBarrier> barriers;

        VkPipelineStageFlags src_stages = 0;
        VkPipelineStageFlags dst_stages = 0;

        std::vector<VkAttachmentDescription> attachments;
        std::vector<VkImageView> framebuffer_attachments;
        std::vector<VkClearValue> clear_values;
        std::vector<VkAttachmentReference> input_attachments;
        std::vector<VkAttachmentReference> color_attachments;
        std::vector<VkAttachmentReference> resolve_attachments;
        std::optional<VkAttachmentReference> depth_stencil_attachment;

        static constexpr auto make_barrier = [](Attachment& attachment, VkAccessFlags access, VkImageLayout layout) -> VkImageMemoryBarrier {
            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.image = attachment.attachment.tex.image.image;
            barrier.srcAccessMask = attachment.access;
            barrier.dstAccessMask = access;
            barrier.oldLayout = attachment.layout;
            barrier.newLayout = layout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange = attachment.attachment.subresource;
            return barrier;
        };

        // Wait for any past usage
        for (const auto& [name, self, clear] : pass.input_attachments) {
            Attachment& attachment = tracked_attachments[name];

            VkAccessFlags access = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
            VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

            if (self) {
                access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                layout = VK_IMAGE_LAYOUT_GENERAL;
                dst_stage |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

                VkClearValue val;
                if (clear.has_value()) {
                    val.color = clear.value();
                } else {
                    val.color.float32[0] = 0.f;
                    val.color.float32[1] = 0.f;
                    val.color.float32[2] = 0.f;
                    val.color.float32[3] = 0.f;
                }

                clear_values.push_back(val);
            }

            src_stages |= attachment.stage;
            dst_stages |= dst_stage;

            barriers.push_back(make_barrier(attachment, access, layout));

            VkAttachmentReference attachment_ref = {};
            attachment_ref.layout = layout;
            attachment_ref.attachment = attachments.size();

            input_attachments.push_back(attachment_ref);

            if (self) {
                color_attachments.push_back(attachment_ref);
            }

            VkAttachmentDescription attachment_desc = {};
            attachment_desc.initialLayout = layout;
            attachment_desc.finalLayout = layout;
            attachment_desc.format = attachment.attachment.tex.image.format;
            attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment_desc.samples = attachment.attachment.tex.image.samples;

            if (self && clear.has_value()) {
                attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            }

            attachments.push_back(attachment_desc);
            framebuffer_attachments.push_back(attachment.attachment.tex.view);

            // Update attachment info for following passes wanting to synchronize
            attachment.access = access;
            attachment.layout = layout;
            attachment.stage = dst_stage;
        }

        for (const Name& name : pass.texture_inputs) {
            Attachment& attachment = tracked_attachments[name];

            VkAccessFlags access = VK_ACCESS_SHADER_READ_BIT;
            VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

            src_stages |= attachment.stage;
            dst_stages |= dst_stage;

            barriers.push_back(make_barrier(attachment, access, layout));

            // Update attachment info for following passes wanting to synchronize
            attachment.access = access;
            attachment.layout = layout;
            attachment.stage = dst_stage;
        }

        for (const auto& [name, dep] : pass.dependencies) {
            Attachment& attachment = tracked_attachments[name];

            const VkAccessFlags access = dep.access;
            const VkImageLayout layout = dep.layout;
            const VkPipelineStageFlags dst_stage = dep.stage;

            src_stages |= attachment.stage;
            dst_stages |= dst_stage;

            if (!dep.virt)
                barriers.push_back(make_barrier(attachment, access, layout));

            attachment.access = access;
            attachment.layout = layout;
            attachment.stage = dst_stage;
        }

        if (pass.depth_stencil.has_value()) {
            Attachment& attachment = tracked_attachments[pass.depth_stencil.value().first];

            VkAccessFlags access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            VkImageLayout layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            const VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

            if (pass.depth_stencil.value().second.has_value()) {
                access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }

            src_stages |= attachment.stage;
            dst_stages |= dst_stage;

            barriers.push_back(make_barrier(attachment, access, layout));

            VkAttachmentReference attachment_ref = {};
            attachment_ref.layout = layout;
            attachment_ref.attachment = attachments.size();

            depth_stencil_attachment = attachment_ref;

            VkAttachmentDescription attachment_desc = {};
            attachment_desc.initialLayout = layout;
            attachment_desc.finalLayout = layout;
            attachment_desc.format = attachment.attachment.tex.image.format;
            attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment_desc.samples = attachment.attachment.tex.image.samples;

            if (pass.depth_stencil.value().second.has_value()) {
                attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                VkClearValue val;
                val.depthStencil = pass.depth_stencil.value().second.value();
                clear_values.push_back(val);
            } else {
                VkClearValue val;
                val.depthStencil.depth = 0.f;
                val.depthStencil.stencil = 0;
                clear_values.push_back(val);
            }

            attachments.push_back(attachment_desc);
            framebuffer_attachments.push_back(attachment.attachment.tex.view);

            attachment.access = access;
            attachment.layout = layout;
            attachment.stage = dst_stage;
        }

        // Synchronize writes
        for (const auto& [name, clear] : pass.color_outputs) {
            Attachment& attachment = tracked_attachments[name];

            const VkAccessFlags access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
            const VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            const VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

            src_stages |= attachment.stage;
            dst_stages |= dst_stage;

            barriers.push_back(make_barrier(attachment, access, layout));

            VkAttachmentReference attachment_ref = {};
            attachment_ref.layout = layout;
            attachment_ref.attachment = attachments.size();

            color_attachments.push_back(attachment_ref);

            VkAttachmentDescription attachment_desc = {};
            attachment_desc.initialLayout = layout;
            attachment_desc.finalLayout = layout;
            attachment_desc.format = attachment.attachment.tex.image.format;
            attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment_desc.samples = attachment.attachment.tex.image.samples;

            if (clear.has_value()) {
                attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                VkClearValue val;
                val.color = clear.value();
                clear_values.push_back(val);
            } else {
                VkClearValue val;
                val.color.float32[0] = 0.f;
                val.color.float32[1] = 0.f;
                val.color.float32[2] = 0.f;
                val.color.float32[3] = 0.f;
                clear_values.push_back(val);
            }

            attachments.push_back(attachment_desc);
            framebuffer_attachments.push_back(attachment.attachment.tex.view);

            attachment.access = access;
            attachment.layout = layout;
            attachment.stage = dst_stage;
        }

        for (const auto& [name, clear] : pass.resolve_outputs) {
            Attachment& attachment = tracked_attachments[name];

            const VkAccessFlags access = 0;
            const VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            const VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

            src_stages |= attachment.stage;
            dst_stages |= dst_stage;

            barriers.push_back(make_barrier(attachment, access, layout));

            VkAttachmentReference attachment_ref = {};
            attachment_ref.layout = layout;
            attachment_ref.attachment = attachments.size();

            resolve_attachments.push_back(attachment_ref);

            VkAttachmentDescription attachment_desc = {};
            attachment_desc.initialLayout = layout;
            attachment_desc.finalLayout = layout;
            attachment_desc.format = attachment.attachment.tex.image.format;
            attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment_desc.samples = attachment.attachment.tex.image.samples;

            if (clear.has_value()) {
                attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                VkClearValue val;
                val.color = clear.value();
                clear_values.push_back(val);
            } else {
                VkClearValue val;
                val.color.uint32[0] = 0;
                val.color.uint32[1] = 0;
                val.color.uint32[2] = 0;
                val.color.uint32[3] = 0;
                clear_values.push_back(val);
            }

            attachments.push_back(attachment_desc);
            framebuffer_attachments.push_back(attachment.attachment.tex.view);

            attachment.access = access;
            attachment.layout = layout;
            attachment.stage = dst_stage;
        }

        for (const auto& [name, dep] : pass.dependents) {
            Attachment& attachment = tracked_attachments[name];

            const VkAccessFlags access = dep.access;
            const VkImageLayout layout = dep.layout;
            const VkPipelineStageFlags dst_stage = dep.stage;

            src_stages |= attachment.stage;
            dst_stages |= dst_stage;

            if (!dep.virt)
                barriers.push_back(make_barrier(attachment, access, layout));

            attachment.access = access;
            attachment.layout = layout;
            attachment.stage = dst_stage;
        }

        vkCmdPipelineBarrier(fcx.cmd, src_stages != 0 ? src_stages : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, dst_stages, 0, 0, nullptr, 0, nullptr, barriers.size(),
            barriers.data());

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = color_attachments.size();
        subpass.pColorAttachments = color_attachments.data();
        subpass.inputAttachmentCount = input_attachments.size();
        subpass.pInputAttachments = input_attachments.data();
        subpass.pResolveAttachments = resolve_attachments.data();
        if (depth_stencil_attachment.has_value()) {
            subpass.pDepthStencilAttachment = &depth_stencil_attachment.value();
        }

        VkRenderPassCreateInfo rpci = {};
        rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpci.subpassCount = 1;
        rpci.pSubpasses = &subpass;
        rpci.dependencyCount = 0;
        rpci.pDependencies = nullptr;
        rpci.attachmentCount = attachments.size();
        rpci.pAttachments = attachments.data();

        VkRenderPass rp = cache.create_pass(rpci);

        VkFramebufferCreateInfo fbci = {};
        fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbci.attachmentCount = framebuffer_attachments.size();
        fbci.pAttachments = framebuffer_attachments.data();
        fbci.renderPass = rp;
        fbci.width = pass.width;
        fbci.height = pass.height;
        fbci.layers = pass.layers;

        VkFramebuffer fb = cache.create_framebuffer(fbci);

        VkRenderPassBeginInfo rpbi = {};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = rp;
        rpbi.framebuffer = fb;
        rpbi.clearValueCount = clear_values.size();
        rpbi.pClearValues = clear_values.data();
        rpbi.renderArea = vk_rect(0, 0, pass.width, pass.height);

        if (pass.pre_exec)
            pass.pre_exec(fcx, *this, rp);

        vkCmdBeginRenderPass(fcx.cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        pass.exec(fcx, *this, rp);

        vkCmdEndRenderPass(fcx.cmd);
    }

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = attachments[output].tex.image.image;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = 0;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.oldLayout = tracked_attachments[output].layout;
    barrier.newLayout = output_layout;
    barrier.subresourceRange = attachments[output].subresource;

    vkCmdPipelineBarrier(fcx.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

std::vector<std::size_t> RenderGraph::find_all(std::function<bool(const RenderPass&)> pred) const {
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < passes.size(); ++i) {
        if (pred(passes[i]))
            indices.push_back(i);
    }
    return indices;
}

void RenderGraph::push_writers(std::vector<std::size_t>& all_writers, Name res) const {
    const std::vector<std::size_t> writers = find_all([res](const RenderPass& pass) -> bool {
        return std::find(pass.color_outputs.begin(), pass.color_outputs.end(), res) != pass.color_outputs.end() ||
               std::find(pass.resolve_outputs.begin(), pass.resolve_outputs.end(), res) != pass.resolve_outputs.end() ||
               std::find(pass.dependents.begin(), pass.dependents.end(), res) != pass.dependents.end() ||
               (pass.depth_stencil.has_value() ? pass.depth_stencil.value().second.has_value() && pass.depth_stencil.value().first == res : false);
    });

    list_append(all_writers, writers);

    for (const std::size_t i : writers) {
        const RenderPass& pass = passes[i];

        if (pass.depth_stencil.has_value())
            if (!pass.depth_stencil->second.has_value())
                push_writers(all_writers, pass.depth_stencil.value().first);

        for (const auto& [res, clear] : pass.color_outputs)
            if (!clear.has_value())
                push_writers(all_writers, res);

        for (const auto& [res, self, clear] : pass.input_attachments)
            if (!self)
                push_writers(all_writers, res);

        for (const Name& res : pass.texture_inputs)
            push_writers(all_writers, res);

        for (const auto& [res, _] : pass.dependencies)
            push_writers(all_writers, res);
    }
}

} // namespace gfx
