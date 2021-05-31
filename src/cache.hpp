#pragma once

#include <unordered_map>

template <typename T>
class Cache final {
  public:
    struct Handle final {
        std::size_t id;
    };

    Cache() : next_id{0} {
    }

    T& get(Handle h) {
        return cache.at(h.id);
    }

    const T& get(Handle h) const {
        return cache.at(h.id);
    }

    bool valid(Handle h) const {
        return cache.count(h.id);
    }

    Handle push(T v) {
        const std::size_t id = next_id;
        next_id++;
        cache.emplace(id, std::move(v));
        return {id};
    }

    bool remove(Handle h) {
        if (!valid(h)) {
            return false;
        } else {
            cache.erase(h.id);
            return true;
        }
    }

    template <typename... Ts>
    Handle emplace(Ts&&... arg) {
        return push(T{std::forward<Ts>(arg)...});
    }

    const std::unordered_map<std::size_t, T>& all() const noexcept {
        return cache;
    }

  private:
    std::size_t next_id;
    std::unordered_map<std::size_t, T> cache;
};
