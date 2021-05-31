#include "mesh.hpp"

#include "helpers.hpp"

namespace gfx {

VertexInputDescription Vertex::description(uint32_t mask) {
    VertexInputDescription description = {};

    VkVertexInputBindingDescription main_binding = {};
    main_binding.binding = 0;
    main_binding.stride = sizeof(Vertex);
    main_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    description.bindings.push_back(main_binding);

    uint8_t loc = 0;

    if (mask & Position) {
        VkVertexInputAttributeDescription position_attribute = {};
        position_attribute.binding = 0;
        position_attribute.location = loc;
        position_attribute.format = VK_FORMAT_R32G32B32_SFLOAT;
        position_attribute.offset = offsetof(Vertex, position);

        description.attributes.push_back(position_attribute);

        ++loc;
    }

    if (mask & Normal) {
        VkVertexInputAttributeDescription normal_attribute = {};
        normal_attribute.binding = 0;
        normal_attribute.location = loc;
        normal_attribute.format = VK_FORMAT_R32G32B32_SFLOAT;
        normal_attribute.offset = offsetof(Vertex, normal);

        description.attributes.push_back(normal_attribute);

        ++loc;
    }

    if (mask & TexCoord) {
        VkVertexInputAttributeDescription tex_coord_attribute = {};
        tex_coord_attribute.binding = 0;
        tex_coord_attribute.location = loc;
        tex_coord_attribute.format = VK_FORMAT_R32G32_SFLOAT;
        tex_coord_attribute.offset = offsetof(Vertex, tex_coord);

        description.attributes.push_back(tex_coord_attribute);

        ++loc;
    }

    return description;
}

Vertex& Vertex::set_position(glm::vec3 p) {
    position = p;
    return *this;
}

Vertex& Vertex::set_normal(glm::vec3 n) {
    normal = n;
    return *this;
}

Vertex& Vertex::set_tex_coord(glm::vec2 tc) {
    tex_coord = tc;
    return *this;
}

bool Vertex::operator==(const Vertex& rhs) const noexcept {
    return position == rhs.position && normal == rhs.normal && tex_coord == rhs.tex_coord;
}

} // namespace gfx

std::size_t std::hash<gfx::Vertex>::operator()(const gfx::Vertex& vert) const {
    std::size_t h = 0;
    hash_combine(h, vert.position, vert.normal, vert.tex_coord);
    return h;
}
