#include "deletion_queue.hpp"

namespace gfx {

void DeletionQueue::push(std::function<void()> f) {
    all.push_back(std::move(f));
}

void DeletionQueue::flush() {
    // reverse iter
    for (auto it = all.rbegin(); it != all.rend(); ++it) {
        (*it)();
    }
    all.clear();
}

} // namespace gfx
