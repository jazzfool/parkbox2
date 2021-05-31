#pragma once

#include <vector>
#include <volk.h>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

namespace gfx {

struct VertexInputDescription final {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;

    VkPipelineVertexInputStateCreateFlags flags;
};

struct Vertex final {
    enum Mask {
        Position = 1 << 0,
        Normal = 1 << 1,
        TexCoord = 1 << 2,
    };

    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 tex_coord;

    static VertexInputDescription description(uint32_t mask = Position | Normal | TexCoord);

    Vertex& set_position(glm::vec3 p);
    Vertex& set_normal(glm::vec3 n);
    Vertex& set_tex_coord(glm::vec2 tc);

    bool operator==(const Vertex& rhs) const noexcept;
};

} // namespace gfx

namespace std {

template <>
struct hash<gfx::Vertex> {
    std::size_t operator()(const gfx::Vertex& vert) const;
};

} // namespace std
