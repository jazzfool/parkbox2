#version 450

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec4 out_position;
layout(location = 1) out vec3 out_normal;

struct Instance {
    mat4 transform;
    uint material;
    uint batch_idx;
    vec2 uv_scale;
    vec4 bounds;
};

layout(set = 0, binding = 0) readonly buffer InstanceBuffer {
    Instance instances[];
}
instance_buf;

layout(set = 0, binding = 1) readonly buffer InstanceIndexBuffer {
    uint indices[];
}
instance_index_buf;

layout(set = 0, binding = 2) uniform Uniforms {
    SceneUniforms uniforms;
};

void main() {
    Instance instance = instance_buf.instances[instance_index_buf.indices[gl_InstanceIndex]];

    vec3 pos = (instance.transform * vec4(in_position, 1)).xyz;
    out_normal = normalize(transpose(inverse(mat3(instance.transform))) * in_normal);

    gl_Position = uniforms.cam_proj * vec4(pos, 1);
    out_position = gl_Position;
}
