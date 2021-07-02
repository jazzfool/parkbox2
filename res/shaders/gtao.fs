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

// we only take one sample because we are integrating temporally
const uint SAMPLES = 1;
const float RADIUS = 64.0;
const uint STEPS = 4;

layout(push_constant) uniform PC {
    float jitter;
};

const float dither[4][4] = {
    {0.0f, 0.5f, 0.125f, 0.625f}, {0.75f, 0.22f, 0.875f, 0.375f}, {0.1875f, 0.6875f, 0.0625f, 0.5625}, {0.9375f, 0.4375f, 0.8125f, 0.3125}};

void main() {
    vec2 uv = in_uv;
    uv.y = 1.0 - uv.y;

    const vec4 dn = texture(depth_normal, uv);
    const vec3 view_pos = get_view_pos(uv, dn, inv_view_proj, view);
    const vec3 view_normal = get_view_normal(uv, dn, view);
    const vec3 view_dir = normalize(-view_pos);
    const vec2 texel_size = 1.0 / vec2(textureSize(depth_normal, 0).xy);

    // const float radius = min(RADIUS / length(view_pos), 150.0);

    const float proj_scale = textureSize(depth_normal, 0).y / (tan(1.0472 * 0.5) * 2) * 0.5;
    const float step_radius = max(min((RADIUS * proj_scale) / view_pos.b, 512), float(STEPS)) / (float(STEPS) + 1.0);

    float phi = 0.0;
    const float phi_step = PI / float(SAMPLES);

    float ao = 0.0;

    out_ao = 0.0;

    const float noise = dither[int(gl_FragCoord.x) % 4][int(gl_FragCoord.y) % 4];

    // numerically integrate the outer integral
    for (uint i = 0; i < SAMPLES; ++i) {
        // t hat is parameterized by phi
        const vec2 t = vec2(cos(noise + jitter), sin(noise + jitter));

        // find greatest horizon angles (theta_1 and theta_2)
        // these will be the lower and upper bounds of the analytical integration
        float theta_1 = -1.0;
        float theta_2 = -1.0;
        for (uint j = 1; j < STEPS + 1; ++j) {
            // const float s_fac = (float(j) / float(STEPS + 1)) * radius;
            const float s_fac = max(step_radius * float(j), j + 1);
            const vec2 s_uv_1 = uv + texel_size * t * s_fac;
            const vec2 s_uv_2 = uv - texel_size * t * s_fac;
            const vec4 dn_1 = texture(depth_normal, s_uv_1);
            const vec4 dn_2 = texture(depth_normal, s_uv_2);
            const vec3 s_1 = get_view_pos(s_uv_1, dn_1, inv_view_proj, view);
            const vec3 s_2 = get_view_pos(s_uv_2, dn_2, inv_view_proj, view);
            // although the paper calls for omega_s to be normalized, we need to keep the length to calculate distance fade
            const vec3 omega_s_1 = s_1 - view_pos;
            const vec3 omega_s_2 = s_2 - view_pos;
            const float len_1 = length(omega_s_1);
            const float len_2 = length(omega_s_2);
            const float SoV_1 = dot(view_dir, omega_s_1 / len_1); // so we normalize here instead
            const float SoV_2 = dot(view_dir, omega_s_2 / len_2);

            // calculate distance fade
            float dist_fac_1 = saturate(len_1 / RADIUS);
            dist_fac_1 *= dist_fac_1;
            float dist_fac_2 = saturate(len_2 / RADIUS);
            dist_fac_2 *= dist_fac_2;

            theta_1 = mix(max(theta_1, SoV_1), theta_1, dist_fac_1); // lerp by distance fade
            theta_2 = mix(max(theta_2, SoV_2), theta_2, dist_fac_2);
        }

        // we found the dot products of the normalized vectors, so just take inverse cos to get angle
        theta_1 = acos(theta_1);
        theta_2 = acos(theta_2);

        const vec3 plane_normal = normalize(cross(vec3(t, 0.0), view_dir));
        const vec3 tangent = cross(view_dir, plane_normal);
        const vec3 proj_n = view_normal - plane_normal * dot(view_normal, plane_normal);
        const float proj_n_len = length(proj_n);

        const float cos_n = clamp(dot(normalize(proj_n), view_dir), -1.0, 1.0);
        const float n = -sign(dot(proj_n, tangent)) * acos(cos_n);

        theta_1 = n + max(-theta_1 - n, -PI_HALF);
        theta_2 = n + min(theta_2 - n, PI_HALF);

        const float sin_n = sin(n);
        const float arc_1 = -cos(2.0 * theta_1 - n) + cos_n + 2.0 * theta_1 * sin_n;
        const float arc_2 = -cos(2.0 * theta_2 - n) + cos_n + 2.0 * theta_2 * sin_n;
        const float a = 0.25 * (arc_1 + arc_2);

        ao += proj_n_len * a;

        phi += phi_step;
    }

    ao /= float(SAMPLES);

    out_ao = ao;

    const vec3 world_pos = reconstruct_world_pos(dn.a, uv, inv_view_proj);

    vec4 prev_pos = prev_view_proj * vec4(world_pos, 1);
    prev_pos /= prev_pos.w;
    prev_pos.xy = prev_pos.xy * vec2(0.5) + vec2(0.5);

    const float prev_ao = texture(prev_ssao, prev_pos.xy).r;
    const float prev_depth = texture(depth_normal, prev_pos.xy).a;

    const bool depth_reject = abs(prev_depth - dn.a) > 0.001;
    const float alpha = 0.05 * float(!depth_reject) + float(depth_reject);
    out_ao = mix(prev_ao, out_ao, alpha);
}
