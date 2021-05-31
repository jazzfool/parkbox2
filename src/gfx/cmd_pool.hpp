#pragma once

#include <volk.h>
#include <vector>
#include <mutex>

namespace gfx {

struct Context;

class CommandPool final {
  public:
    CommandPool();

    void init(Context& cx);
    void cleanup();

    VkCommandBuffer take();
    void replace(VkCommandBuffer cmd);

    VkCommandPool pool;

  private:
    VkDevice dev;
    std::vector<VkCommandBuffer> cmds;
    std::mutex m;
    std::size_t total;
};

} // namespace gfx
