#version 450

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "pbr.glsl"

#define NUM_CASCADES 4

struct Material {
    uint albedo;
    uint roughness;
    uint metallic;
    uint normal;
    uint ao;
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

layout(set = 0, binding = 9) uniform sampler2DArrayShadow shadow_map;

layout(set = 0, binding = 10) uniform ShadowUniforms {
    mat4 views[NUM_CASCADES];
    mat4 view_proj[NUM_CASCADES];
    vec4 cascade_splits;
}
shadow_uniforms;

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

const mat4 bias_mat = mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);

const float dither[4][4] = {
    {0.0f, 0.5f, 0.125f, 0.625f}, {0.75f, 0.22f, 0.875f, 0.375f}, {0.1875f, 0.6875f, 0.0625f, 0.5625}, {0.9375f, 0.4375f, 0.8125f, 0.3125}};

float shadow_sample(vec2 base_uv, float u, float v, vec2 texel_size, uint cascade, float depth, vec2 rpdb) {
    vec2 uv = base_uv + vec2(u, v) * texel_size;
    float z = depth + dot(vec2(u, v) * texel_size, rpdb);
    vec4 s = textureGather(shadow_map, vec3(uv, cascade), z);
    return (s.x + s.y + s.z + s.w) / 4.0;
}

float shadow() {
    const float biases[4] = {0.1, 0.04, 0.0075, 0.002};

    bool ranges[3] = {
        in_mv_position.z < shadow_uniforms.cascade_splits[0],
        in_mv_position.z < shadow_uniforms.cascade_splits[1],
        in_mv_position.z < shadow_uniforms.cascade_splits[2],
    };

    uint cascade = 0;
    cascade += 1 * uint(ranges[0] && !ranges[1] && !ranges[2]);
    cascade += 2 * uint(ranges[1] && !ranges[2]);
    cascade += 3 * uint(ranges[2]);

    vec4 sc =
        (bias_mat * shadow_uniforms.view_proj[cascade]) * vec4(in_position + in_normal * 0.006 * dither[int(gl_FragCoord.x) % 4][int(gl_FragCoord.y) % 4], 1.0);
    sc = sc / sc.w;

    float z = sc.z;
    vec3 dx = dFdx(sc.xyz);
    vec3 dy = dFdy(sc.xyz);

    vec2 size = textureSize(shadow_map, 0).xy;
    vec2 texel_size = 1.0 / size;

    vec2 rpdb = vec2(0.0);
    z -= biases[cascade] * 0.75;

    vec2 uv = sc.xy * size;

    vec2 base_uv;
    base_uv.x = floor(uv.x + 0.5);
    base_uv.y = floor(uv.y + 0.5);

    float s = (uv.x + 0.5 - base_uv.x);
    float t = (uv.y + 0.5 - base_uv.y);

    base_uv -= vec2(0.5);
    base_uv *= texel_size;

    float uw0 = (4 - 3 * s);
    float uw1 = 7;
    float uw2 = (1 + 3 * s);

    float u0 = (3 - 2 * s) / uw0 - 2;
    float u1 = (3 + s) / uw1;
    float u2 = s / uw2 + 2;

    float vw0 = (4 - 3 * t);
    float vw1 = 7;
    float vw2 = (1 + 3 * t);

    float v0 = (3 - 2 * t) / vw0 - 2;
    float v1 = (3 + t) / vw1;
    float v2 = t / vw2 + 2;

    float sum = 0.0;

    sum += uw0 * vw0 * shadow_sample(base_uv, u0, v0, texel_size, cascade, z, rpdb);
    sum += uw1 * vw0 * shadow_sample(base_uv, u1, v0, texel_size, cascade, z, rpdb);
    sum += uw2 * vw0 * shadow_sample(base_uv, u2, v0, texel_size, cascade, z, rpdb);

    sum += uw0 * vw1 * shadow_sample(base_uv, u0, v1, texel_size, cascade, z, rpdb);
    sum += uw1 * vw1 * shadow_sample(base_uv, u1, v1, texel_size, cascade, z, rpdb);
    sum += uw2 * vw1 * shadow_sample(base_uv, u2, v1, texel_size, cascade, z, rpdb);

    sum += uw0 * vw2 * shadow_sample(base_uv, u0, v2, texel_size, cascade, z, rpdb);
    sum += uw1 * vw2 * shadow_sample(base_uv, u1, v2, texel_size, cascade, z, rpdb);
    sum += uw2 * vw2 * shadow_sample(base_uv, u2, v2, texel_size, cascade, z, rpdb);

    return sum * 1.0f / 144;
}

void main() {
    vec2 ibl_dfg = texture(ibl_dfg_lut, in_uv).rg;
    vec4 prefilter = texture(prefilter_map, vec3(0, 0, 0));

    Material in_material = material_buf.materials[in_material];

    vec4 albedo = texture(textures[in_material.albedo], in_uv);
    vec4 roughness = texture(textures[in_material.roughness], in_uv);
    vec4 metallic = texture(textures[in_material.metallic], in_uv);
    vec3 normal = texture(textures[in_material.normal], in_uv).xyz;
    vec4 ao = texture(textures[in_material.ao], in_uv);

    albedo = pow(albedo, vec4(2.2));
    normal = get_normal_from_map(in_normal, normal);

    PBRMaterial material;
    material.albedo = albedo.rgb;
    material.roughness = roughness.r;
    material.metallic = metallic.r;
    material.normal = normal;
    material.ao = ao.r;
    material.world_pos = in_position;
    material.reflectance = 0.04;

    PBRComputed pbr_computed = pbr_compute(material, uniforms, ec_dfg_lut);

    out_color = vec4(0, 0, 0, 1);

    out_color.rgb += pbr_sun_light(material, pbr_computed, uniforms) * shadow();
    out_color.rgb += pbr_ibl(material, pbr_computed, irradiance_map, prefilter_map, ibl_dfg_lut);
}
