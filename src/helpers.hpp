#pragma once

#include <iterator>
#include <cmath>
#include <limits>
#include <fstream>
#include <vector>
#include <glm/mat4x4.hpp>
#include <glm/gtx/hash.hpp>

static constexpr inline double PI = 3.1415926535897;
static constexpr inline double PI180 = PI / 180.;
static constexpr inline double C180PI = 180. / PI;

template <typename A, typename B>
void list_append(A& dst, const B& src) {
    dst.insert(std::end(dst), std::begin(src), std::end(src));
}

template <typename Float>
inline bool float_cmp(Float a, Float b) {
    static_assert(std::is_floating_point_v<Float>, "Float must be a floating point type");
    return std::abs(a - b) < std::numeric_limits<Float>::epsilon();
}

// absolutely iconic
inline void hash_combine(std::size_t& seed) {
}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    hash_combine(seed, rest...);
}

template <typename T>
struct HashSpan {
    HashSpan(T* ptr, std::size_t len) : ptr{ptr}, len{len} {
    }

    bool operator==(const HashSpan<T>& rhs) const {
        if (len != rhs.len)
            return false;
        for (std::size_t i = 0; i < len; ++i)
            if (!(ptr[i] == rhs.ptr[i]))
                return false;
        return true;
    }

    T* ptr;
    std::size_t len;
};

template <typename Float>
inline Float radians(Float deg) {
    static_assert(std::is_floating_point_v<Float>, "Float must be a floating point type");
    return deg * PI180;
}

template <typename Float>
inline Float degrees(Float deg) {
    static_assert(std::is_floating_point_v<Float>, "Float must be a floating point type");
    return deg * C180PI;
}

template <typename Number>
inline Number clamp(Number x, Number lo, Number hi) {
    static_assert(std::is_arithmetic_v<Number>, "Number must be an arithmentic type");
    return std::max(std::min(x, hi), lo);
}

template <typename Float>
inline Float saturate(Float x) {
    static_assert(std::is_floating_point_v<Float>, "Float must be a floating point type");
    return clamp(x, static_cast<Float>(0.0), static_cast<Float>(1.0));
}

inline std::vector<uint8_t> read_binary(const std::string& path) {
    std::ifstream f{path, std::ios::binary};
    f.unsetf(std::ios::skipws);
    std::streampos sz;
    f.seekg(0, std::ios::end);
    sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> v;
    v.reserve(sz);
    v.insert(v.begin(), std::istream_iterator<uint8_t>(f), std::istream_iterator<uint8_t>());
    return v;
}

inline std::string read_str(std::string_view path) {
    std::ifstream f;
    f.open(path.data(), std::ios::in | std::ios::binary);
    std::string s{std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}};
    f.close();
    return s;
}

template <typename T, typename R, typename... Ts>
auto mfbind(T* x, R (T::*f)(Ts...)) {
    return [=](Ts... p) { return (x->*f)(p...); };
}

namespace std {

template <typename T>
struct hash<HashSpan<T>> {
    std::size_t operator()(const HashSpan<T>& span) const {
        std::size_t h = 0;
        for (std::size_t i = 0; i < span.len; ++i) {
            hash_combine(h, span.ptr[i]);
        }
        return h;
    }
};

} // namespace std
