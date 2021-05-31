#include "cmd_pool.hpp"

#include "vk_helpers.hpp"
#include "context.hpp"

#include <array>
#include <spdlog/spdlog.h>

namespace gfx {

CommandPool::CommandPool() : total{0} {
}

void CommandPool::init(Context& cx) {
    dev = cx.dev;

    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = /*VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |*/ VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = cx.gfx_queue_idx;

    vk_log(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
}

void CommandPool::cleanup() {
    if (total != cmds.size()) {
        spdlog::warn("destructing cmd pool without replacing {} cmd bufs", static_cast<int32_t>(total) - static_cast<int32_t>(cmds.size()));
    }

    vkDestroyCommandPool(dev, pool, nullptr);
}

VkCommandBuffer CommandPool::take() {
    std::scoped_lock<std::mutex> lock{m};

    if (cmds.empty()) {
        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandBufferCount = 1;
        cbai.commandPool = pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VkCommandBuffer cmd;
        vk_log(vkAllocateCommandBuffers(dev, &cbai, &cmd));

        cmds.push_back(cmd);

        total++;
    }

    VkCommandBuffer cmd = cmds.back();
    cmds.pop_back();

    return cmd;
}

void CommandPool::replace(VkCommandBuffer cmd) {
    std::scoped_lock<std::mutex> lock{m};

    vk_log(vkResetCommandBuffer(cmd, 0));
    cmds.push_back(cmd);
}

} // namespace gfx
