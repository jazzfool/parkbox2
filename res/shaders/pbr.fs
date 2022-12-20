#version 450

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "pbr.glsl"

#define NUM_CASCADES 4

struct Material {
    uint textures[8];
    float scalars[4];
    vec4 vectors[4];
};

layout(location = 0) flat in uint in_material;
layout(location = 1) in vec3 in_position;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec2 in_uv;
layout(location = 4) in vec3 in_mv_position;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 2) readonly buffer MaterialBuffer {
    Material materials[];
}
material_buf;

layout(set = 0, binding = 3) uniform sampler2D textures[];

layout(set = 0, binding = 4) uniform sampler2D ec_dfg_lut;
layout(set = 0, binding = 5) uniform sampler2D ibl_dfg_lut;
layout(set = 0, binding = 6) uniform samplerCube prefilter_map;
layout(set = 0, binding = 7) uniform samplerCube irradiance_map;

layout(set = 0, binding = 8) uniform Uniforms {
    SceneUniforms uniforms;
};

layout(set = 0, binding = 9) uniform sampler2D shadow_buf;

layout(set = 0, binding = 10) uniform ShadowUniforms {
    mat4 views[NUM_CASCADES];
    mat4 view_proj[NUM_CASCADES];
    vec4 cascade_splits;
}
shadow_uniforms;

layout(set = 0, binding = 11) uniform sampler2D ssao;

vec3 get_normal_from_map(vec3 base_normal, vec3 texture_normal) {
    vec3 tangent_normal = texture_normal * 2.0 - 1.0;

    vec3 Q1 = dFdx(in_position);
    vec3 Q2 = dFdy(in_position);
    vec2 st1 = dFdx(in_uv);
    vec2 st2 = dFdy(in_uv);

    vec3 N = normalize(base_normal);
    vec3 T = normalize(Q1 * st2.t - Q2 * st1.t);
    vec3 B = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangent_normal);
}

{...}

void main() {
    vec2 ibl_dfg = texture(ibl_dfg_lut, in_uv).rg;
    vec4 prefilter = texture(prefilter_map, vec3(0, 0, 0));

    float shadow = texture(shadow_buf, gl_FragCoord.xy / textureSize(shadow_buf, 0).xy).r;
    float ssao_fac = texture(ssao, gl_FragCoord.xy / textureSize(shadow_buf, 0).xy).r;

    Material params = material_buf.materials[in_material];
    PBRMaterial pbr_material = material(params);

    PBRComputed pbr_computed = pbr_compute(pbr_material, uniforms, ec_dfg_lut);

    out_color = vec4(0, 0, 0, 1);

    out_color.rgb += pbr_sun_light(pbr_material, pbr_computed, uniforms.sun_dir.xyz, uniforms.sun_radiant_flux.xyz) * shadow;
    out_color.rgb += pbr_ibl(pbr_material, pbr_computed, irradiance_map, prefilter_map, ibl_dfg_lut) * ssao_fac;
}
