#version 450

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec4 out_depth_normal;

void main() {
    out_depth_normal.rgb = in_normal;
    out_depth_normal.w = in_position.z / in_position.w;
}
