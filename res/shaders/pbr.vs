#version 450

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) flat out uint out_material;
layout(location = 1) out vec3 out_position;
layout(location = 2) out vec3 out_normal;
layout(location = 3) out vec2 out_uv;
layout(location = 4) out vec3 out_mv_position;

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

layout(set = 0, binding = 8) uniform Uniforms {
    SceneUniforms uniforms;
};

void main() {
    Instance instance = instance_buf.instances[instance_index_buf.indices[gl_InstanceIndex]];

    out_material = instance.material;
    out_position = (instance.transform * vec4(in_position, 1)).xyz;
    out_normal = normalize(transpose(inverse(mat3(instance.transform))) * in_normal);
    out_uv = in_uv * instance.uv_scale;
    out_mv_position = (uniforms.cam_view * vec4(out_position, 1)).xyz;

    gl_Position = uniforms.cam_proj * vec4(out_position, 1);
}
