#pragma once

#include <deque>
#include <functional>

namespace gfx {

class DeletionQueue final {
  public:
    void push(std::function<void()> f);
    void flush();

  private:
    std::deque<std::function<void()>> all;
};

} // namespace gfx
