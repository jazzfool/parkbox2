#version 450

layout(location = 0) in vec3 in_position;

#define NUM_CASCADES 4

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

layout(set = 0, binding = 2) uniform LightProjction {
    mat4 views[NUM_CASCADES];
    mat4 view_proj[NUM_CASCADES];
    vec4 cascade_splits;
};

layout(push_constant) uniform PC {
    uint cascade;
};

void main() {
    Instance instance = instance_buf.instances[instance_index_buf.indices[gl_InstanceIndex]];
    gl_Position = view_proj[cascade] * instance.transform * vec4(in_position, 1);
}
