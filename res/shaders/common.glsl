#define PI 3.14159265358979323846
#define PI_2 6.28318530717958647692
#define PI_HALF 1.57079632679489661923
#define PI_INV 0.318309886183790671538

struct SceneUniforms {
    vec4 cam_pos;
    vec4 sun_dir;
    vec4 sun_radiant_flux;
    mat4 cam_proj;
    mat4 cam_view;
};

const float A = 0.15;
const float B = 0.50;
const float C = 0.10;
const float D = 0.20;
const float E = 0.02;
const float F = 0.30;
const float W = 11.2;

// Uncharted 2 filmic tonemap
vec3 filmic_tone_map(vec3 x) {
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 reconstruct_world_pos(float depth, vec2 uv, mat4 ivp) {
    vec4 clip_space_pos = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 world_space_pos = ivp * clip_space_pos;
    world_space_pos /= world_space_pos.w;
    return world_space_pos.xyz;
}

vec3 get_view_pos(vec2 uv, vec4 dn, mat4 ivp, mat4 view) {
    vec3 wp = reconstruct_world_pos(dn.w, uv, ivp);
    return (view * vec4(wp, 1)).xyz;
}

vec3 get_view_normal(vec2 uv, vec4 dn, mat4 view) {
    vec3 n = dn.xyz;
    return normalize((view * vec4(n, 0.0)).xyz);
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}
