#version 450

#define NUM_CASCADES 4

layout(location = 0) in vec2 in_uv;

layout(location = 0) out float out_shadow;

layout(set = 0, binding = 0) uniform LightProjection {
    mat4 views[NUM_CASCADES];
    mat4 view_proj[NUM_CASCADES];
    vec4 cascade_splits;
};

layout(set = 0, binding = 1) uniform InverseProjection {
    mat4 inv_view_proj;
    mat4 view;
    mat4 prev_view_proj;
};

layout(set = 0, binding = 2) uniform sampler2DArrayShadow shadow_map;
layout(set = 0, binding = 3) uniform sampler2D prev_shadow_buf;
layout(set = 0, binding = 4) uniform sampler2D depth_normal;

const mat4 bias_mat = mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);

const float dither[4][4] = {
    {0.0f, 0.5f, 0.125f, 0.625f}, {0.75f, 0.22f, 0.875f, 0.375f}, {0.1875f, 0.6875f, 0.0625f, 0.5625}, {0.9375f, 0.4375f, 0.8125f, 0.3125}};

float shadow_sample(vec2 base_uv, float u, float v, vec2 texel_size, uint cascade, float depth, vec2 rpdb) {
    vec2 uv = base_uv + vec2(u, v) * texel_size;
    float z = depth + dot(vec2(u, v) * texel_size, rpdb);
    vec4 s = textureGather(shadow_map, vec3(uv, cascade), z);
    return (s.x + s.y + s.z + s.w) * 0.25;
}

float shadow(vec3 pos, vec3 mv_pos, vec3 normal) {
    const float biases[4] = {0.01, 0.005, 0.002, 0.001};

    bool ranges[3] = {
        mv_pos.z < cascade_splits[0],
        mv_pos.z < cascade_splits[1],
        mv_pos.z < cascade_splits[2],
    };

    uint cascade = 0;
    cascade += 1 * uint(ranges[0] && !ranges[1] && !ranges[2]);
    cascade += 2 * uint(ranges[1] && !ranges[2]);
    cascade += 3 * uint(ranges[2]);

    vec4 sc = (bias_mat * view_proj[cascade]) * vec4(pos + normal * 0.006, 1.0);
    sc = sc / sc.w;

    float z = sc.z;
    vec3 dx = dFdx(sc.xyz);
    vec3 dy = dFdy(sc.xyz);

    vec2 size = textureSize(shadow_map, 0).xy;
    vec2 texel_size = 1.0 / size;

    vec2 rpdb = vec2(0.0);
    z -= biases[cascade];

    vec2 uv = sc.xy * size;

    vec2 base_uv;
    base_uv.x = floor(uv.x + 0.5);
    base_uv.y = floor(uv.y + 0.5);

    float s = (uv.x + 0.5 - base_uv.x);
    float t = (uv.y + 0.5 - base_uv.y);

    base_uv -= vec2(0.5);
    base_uv *= texel_size;

    return shadow_sample(base_uv, 0.0, 0.0, texel_size, cascade, z, rpdb);
}

vec3 reconstruct_world_pos(float depth, vec2 uv) {
    vec4 clip_space_pos = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 world_space_pos = inv_view_proj * clip_space_pos;
    world_space_pos /= world_space_pos.w;
    return world_space_pos.xyz;
}

void main() {
    vec2 uv = in_uv;
    uv.y = 1.0 - uv.y;

    vec4 dn = texture(depth_normal, uv);

    vec3 normal = dn.rgb;
    float depth = dn.a;

    vec3 pos = reconstruct_world_pos(depth, uv);
    vec3 view_pos = (view * vec4(pos, 1)).xyz;

    out_shadow = shadow(pos, view_pos, normal);

    vec4 prev_pos = prev_view_proj * vec4(pos, 1);
    prev_pos /= prev_pos.w;
    prev_pos.xy = prev_pos.xy * vec2(0.5) + vec2(0.5);

    float prev_shadow = texture(prev_shadow_buf, prev_pos.xy).r;
    float prev_depth = texture(depth_normal, prev_pos.xy).w;

    bool depth_reject = abs(prev_depth - depth) > 0.001;

    float dither_val = dither[int(gl_FragCoord.x) % 4][int(gl_FragCoord.y) % 4];

    float alpha = 0.05 * float(!depth_reject) * dither_val + float(depth_reject);
    out_shadow = out_shadow * alpha + (1.0 - alpha) * prev_shadow;
}
