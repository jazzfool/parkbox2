#version 450

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_world_pos;

layout(location = 0) out vec2 out_color;

#define NUM_CASCADES 4

layout(set = 0, binding = 3) uniform LightProjction {
    mat4 views[NUM_CASCADES];
    mat4 view_proj[NUM_CASCADES];
    vec4 cascade_splits;
};

layout(set = 0, binding = 4) uniform sampler2DArray src_map;

layout(push_constant) uniform PC {
    uint cascade;
    uint frames;
};

void main() {
    vec2 size = textureSize(src_map, 0).xy;

    vec4 pos = view_proj[cascade] * in_world_pos;
    pos /= pos.w;
    vec2 uv = pos.xy;
    uv = uv * 0.5 + 0.5;

    vec2 prev = texture(src_map, vec3(uv, cascade)).xy;

    float d = in_position.z / in_position.w;

    out_color.x = d;

    float dx = dFdx(d);
    float dy = dFdy(d);
    out_color.y = d * d + 0.25 * (dx * dx + dy * dy);

    float alpha = 0.005;
    out_color = out_color * alpha + prev * (1.0 - alpha);
}
