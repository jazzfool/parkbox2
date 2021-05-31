#version 450

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2DMS pbr_out;

layout(push_constant) uniform Dim {
    vec2 dims;
};

// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/

float fmax3(float x, float y, float z) {
    return max(x, max(y, z));
}

vec3 tone_map(vec3 c, float w) {
    return c * (w * (1.0 / (fmax3(c.r, c.g, c.b) + 1.0)));
}

vec3 inv_tone_map(vec3 c) {
    return c * (1.0 / (1.0 - fmax3(c.r, c.g, c.b)));
}

void main() {
    ivec2 p = ivec2(in_uv * dims);

    vec3 s0 = texelFetch(pbr_out, p, 0).rgb;
    vec3 s1 = texelFetch(pbr_out, p, 1).rgb;
    vec3 s2 = texelFetch(pbr_out, p, 2).rgb;
    vec3 s3 = texelFetch(pbr_out, p, 3).rgb;

    out_color = vec4(inv_tone_map(tone_map(s0, 0.25) + tone_map(s1, 0.25) + tone_map(s2, 0.25) + tone_map(s3, 0.25)), 1);
}
