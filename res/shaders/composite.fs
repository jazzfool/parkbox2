#version 450

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D composite_in;

void main() {
    vec2 uv = in_uv;

    vec3 in_color = texture(composite_in, uv).rgb;

    out_color = vec4(filmic_tone_map(in_color), 1);
}
