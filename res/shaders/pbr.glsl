#define PI 3.141592653589793238
#define INV_PI 0.31830988618
#define MEDIUMP_FLT_MAX 65504.0
#define SATURATE_MEDIUMP(x) min(x, MEDIUMP_FLT_MAX)

struct PBRMaterial {
    vec3 albedo;
    float metallic;
    float roughness;
    float reflectance;
    float ao;
    vec3 normal;
    vec3 world_pos;
};

struct PBRComputed {
    float NoV;
    vec3 Fd;
    vec3 diffuse;
    vec3 v;
    float roughness;
    vec3 f0;
    vec2 dfg;
};

float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

// For Cook-Torrance:

// Approximation 1: The normal distribution function, D.
// The distribution function used is called GGX.
// More optimized implementation.
float d_ggx(float NoH, float roughness) {
    float one_minus = 1.0 - NoH * NoH;
    float a = NoH * roughness;
    float k = roughness / (one_minus + a * a);
    float d = k * k * INV_PI;
    return clamp(d, 0.0, 1.0);
}

// Approximation 2: The shadowing function, G.
// Smith + GGX + height-correlated.
// Again, optimized.
float v_smith_ggx_correlated_fast(float NoV, float NoL, float roughness) {
    float a = roughness;
    float GGXV = NoL * (NoV * (1.0 - a) + a);
    float GGXL = NoV * (NoL * (1.0 - a) + a);
    return 0.5 / (GGXV + GGXL);
}

float v_smith_ggx_correlated(float NoV, float NoL, float roughness) {
    float a = roughness;
    float GGXV = NoL * (NoV * (1.0 - a) + a);
    float GGXL = NoV * (NoL * (1.0 - a) + a);
    return 0.5 / (GGXV + GGXL);
}

// Approximation 3: The fresnel function, F.
// Schlick approximation.
vec3 f_schlick(float u, vec3 f0) {
    float f90 = clamp(dot(f0, vec3(50.0 * 0.33)), 0.0, 1.0);
    return f0 + (f90 - f0) * pow5(1.0 - u);
}

// TODO(jazzfool): maybe use the Disney BRDF instead? it looks quite nice but is a bit slower.

float fd_lambert() {
    return INV_PI;
}

PBRComputed pbr_compute(PBRMaterial material, SceneUniforms uniforms, sampler2D dfg_lut) {
    PBRComputed computed;

    computed.v = normalize(uniforms.cam_pos.xyz - material.world_pos);
    computed.NoV = abs(dot(material.normal, computed.v));
    computed.diffuse = (1.0 - material.metallic) * material.albedo;
    // roughness = (perceptual roughness)^2
    computed.roughness = material.roughness * material.roughness;
    computed.Fd = computed.diffuse * fd_lambert();
    computed.f0 = 0.16 * material.reflectance * material.reflectance * (1.0 - material.metallic) + material.albedo * material.metallic;
    computed.dfg = texture(dfg_lut, vec2(dot(material.normal, computed.v), computed.roughness)).xy;

    return computed;
}

vec3 pbr_sun_light(PBRMaterial material, PBRComputed computed, SceneUniforms uniforms) {
    const vec3 n = material.normal;
    const vec3 l = normalize(-uniforms.sun_dir.xyz);
    const vec3 h = normalize(computed.v + l);
    const float NoL = clamp(dot(n, l), 0.0, 1.0);
    const float NoH = clamp(dot(n, h), 0.0, 1.0);
    const float LoH = clamp(dot(l, h), 0.0, 1.0);

    const float D = d_ggx(NoH, computed.roughness);
    const vec3 F = f_schlick(LoH, computed.f0);
    const float V = v_smith_ggx_correlated(computed.NoV, NoL, computed.roughness);

    vec3 Fr = (D * V) * F;

    const vec3 energy_compensation = 1.0 + computed.f0 * (1.0 / computed.dfg.y - 1.0);
    Fr *= energy_compensation;

    vec3 Lo = (computed.Fd + Fr) * uniforms.sun_radiant_flux.rgb * NoL;

    return Lo;
}

/*
vec3 pbr_point_light(PBRMaterial material, PBRComputed computed, SceneUniforms uniforms, PointLight light) {
    if (light.enabled == 0) {
        return vec3(0);
    }

    const vec3 to_light = light.position.xyz - material.world_pos;

    const vec3 n = material.normal;
    const vec3 l = normalize(to_light);
    const vec3 h = normalize(computed.v + l);
    const float NoL = clamp(dot(n, l), 0.0, 1.0);
    const float NoH = clamp(dot(n, h), 0.0, 1.0);
    const float LoH = clamp(dot(l, h), 0.0, 1.0);

    const float D = d_ggx(NoH, computed.roughness);
    const vec3 F = f_schlick(LoH, computed.f0);
    const float V = v_smith_ggx_correlated_fast(computed.NoV, NoL, computed.roughness);

    vec3 Fr = (D * V) * F;

    const vec3 energy_compensation = 1.0 + computed.f0 * (1.0 / computed.dfg.y - 1.0);
    Fr *= energy_compensation;

    const float dist = length(to_light);

    vec3 Lo = (computed.Fd + Fr) * (light.color.rgb / (dist * dist)) * NoL;

    return Lo;
}
*/

vec3 pbr_indirect(PBRComputed computed, SceneUniforms uniforms, vec3 color) {
    return computed.diffuse * color * fd_lambert();
}

vec3 pbr_ibl(PBRMaterial mat, PBRComputed comp, samplerCube irrad, samplerCube prefilter, sampler2D dfg_lut) {
    float f90 = clamp(dot(comp.f0, vec3(50.0 * 0.33)), 0.0, 1.0);
    vec3 r = reflect(-comp.v, mat.normal);
    vec3 Ld = texture(irrad, r).rgb * mat.albedo;
    float lod = mat.roughness * 8.0;
    vec3 Lld = textureLod(prefilter, r, lod).rgb;
    vec2 Ldfg = texture(dfg_lut, vec2(dot(mat.normal, comp.v), comp.roughness)).xy;
    vec3 Lr = (comp.f0 * Ldfg.x + f90 * Ldfg.y) * Lld;
    return Ld + Lr;
}
