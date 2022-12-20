#version 450

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec2 in_uv;

layout(location = 0) out float out_ao;

layout(set = 0, binding = 0) uniform sampler2D depth_normal;

layout(set = 0, binding = 1) uniform sampler2D prev_ssao;

layout(set = 0, binding = 2) uniform InverseProjection {
    mat4 inv_view_proj;
    mat4 view;
    mat4 prev_view_proj;
};

const uint SAMPLES = 8;
const float RADIUS = 64.0;
const uint STEPS = 4;

layout(push_constant) uniform PC {
    float jitter;
};

const float dither[4][4] = {
    {0.0f, 0.5f, 0.125f, 0.625f}, {0.75f, 0.22f, 0.875f, 0.375f}, {0.1875f, 0.6875f, 0.0625f, 0.5625}, {0.9375f, 0.4375f, 0.8125f, 0.3125}};

float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 rotate_dir(vec2 dir, vec2 cos_sin) {
    return vec2(dir.x * cos_sin.x - dir.y * cos_sin.y, dir.x * cos_sin.y + dir.y * cos_sin.x);
}

float falloff(float distance_sqr) {
    // 1 scalar mad instruction
    return distance_sqr * (-1.0 / (RADIUS * RADIUS)) + 1.0;
}

float compute_ao(vec3 P, vec3 N, vec3 s) {
    vec3 V = s - P;
    float VoV = dot(V, V);
    float NoV = dot(N, V) * 1.0 / sqrt(VoV);

    // Use saturate(x) instead of max(x,0.f) because that is faster on Kepler
    return clamp(NoV - 0.1, 0, 1) * clamp(falloff(VoV), 0, 1);
}

void main() {
    vec2 uv = in_uv;
    uv.y = 1.0 - uv.y;

    const vec4 dn = texture(depth_normal, uv);
    const vec3 view_pos = get_view_pos(uv, dn, inv_view_proj, view);
    const vec3 view_normal = get_view_normal(uv, dn, view);
    const vec3 view_dir = normalize(-view_pos);
    const vec2 texel_size = 1.0 / vec2(textureSize(depth_normal, 0).xy);

    float step_size_pixels = RADIUS / (STEPS + 1);

    const float alpha = 2.0 * PI / SAMPLES;
    float AO = 0;

    uvec2 xy = uvec2(jitter * in_uv * textureSize(prev_ssao, 0).xy);
    float noise = dither[xy.x % 4][xy.y % 4];

    for (float dir_idx = 0; dir_idx < SAMPLES; ++dir_idx) {
        float angle = alpha * dir_idx;

        // Compute normalized 2D direction
        vec2 dir = rotate_dir(vec2(cos(angle), sin(angle)), vec2(rand(vec2(jitter + uv)), -rand(vec2(jitter * jitter - uv))));

        // Jitter starting sample within the first step
        float ray_pixels = (noise * step_size_pixels + 1.0);

        for (float step_idx = 0; step_idx < STEPS; ++step_idx) {
            vec2 snapped_uv = round(ray_pixels * dir) * texel_size + uv;
            vec3 s = get_view_pos(snapped_uv, texture(depth_normal, snapped_uv), inv_view_proj, view);

            ray_pixels += step_size_pixels;

            AO += compute_ao(view_pos, view_normal, s);
        }
    }

    AO *= 3.0 / (SAMPLES * STEPS);
    out_ao = clamp(1.0 - AO * 2.0, 0, 1);

    const vec3 world_pos = reconstruct_world_pos(dn.a, uv, inv_view_proj);

    vec4 prev_pos = prev_view_proj * vec4(world_pos, 1);
    prev_pos /= prev_pos.w;
    prev_pos.xy = prev_pos.xy * vec2(0.5) + vec2(0.5);

    const float prev_ao = texture(prev_ssao, prev_pos.xy).r;
    const float prev_depth = texture(depth_normal, prev_pos.xy).a;

    const bool depth_reject = abs(prev_depth - dn.a) > 0.001;
    const float mix_alpha = 0.02 * float(!depth_reject) + float(depth_reject);
    out_ao = mix(prev_ao, out_ao, mix_alpha);
}