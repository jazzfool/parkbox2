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
    vec3 f0;
    vec2 dfg;
};

float d_ggx(float NoH, float roughness) {
    float one_minus = 1.0 - NoH * NoH;
    float a = NoH * roughness;
    float k = roughness / (one_minus + a * a);
    float d = k * k * PI_INV;
    return clamp(d, 0.0, 1.0);
}

float v_smith_ggx_correlated(float NoV, float NoL, float roughness) {
    float a = roughness;
    float GGXV = NoL * (NoV * (1.0 - a) + a);
    float GGXL = NoV * (NoL * (1.0 - a) + a);
    return 0.5 / (GGXV + GGXL);
}

float f_schlick(float u, float f0, float f90) {
    return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

vec3 f_schlick2(float u, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(1.0 - u, 5.0);
}

float fd_burley(float NoV, float NoL, float LoH, float roughness) {
    float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
    float ls = f_schlick(NoL, 1.0, f90);
    float vs = f_schlick(NoV, 1.0, f90);
    return ls * vs * (1.0 / PI);
}

vec3 brdf(PBRMaterial mat, PBRComputed comp, vec3 l) {
    vec3 h = normalize(comp.v + l);

    float NoV = abs(dot(mat.normal, comp.v)) + 1e-5;
    float NoL = clamp(dot(mat.normal, l), 0.0, 1.0);
    float NoH = clamp(dot(mat.normal, h), 0.0, 1.0);
    float LoH = clamp(dot(l, h), 0.0, 1.0);

    float roughness = mat.roughness * mat.roughness;

    float D = d_ggx(NoH, roughness);
    vec3 F = f_schlick2(LoH, comp.f0);
    float V = v_smith_ggx_correlated(NoV, NoL, roughness);

    vec3 ec = 1.0 + comp.f0 * (1.0 / comp.dfg.y - 1.0);
    vec3 Fr = (D * V) * F * ec;

    vec3 Fd = comp.diffuse * fd_burley(NoV, NoL, LoH, roughness);

    return (Fd + Fr) * NoL;
}

PBRComputed pbr_compute(PBRMaterial material, SceneUniforms uniforms, sampler2D dfg_lut) {
    PBRComputed computed;

    computed.v = normalize(uniforms.cam_pos.xyz - material.world_pos);
    computed.NoV = abs(dot(material.normal, computed.v));
    computed.diffuse = (1.0 - material.metallic) * material.albedo;
    computed.f0 = 0.16 * material.reflectance * material.reflectance * (1.0 - material.metallic) + material.albedo * material.metallic;
    computed.dfg = texture(dfg_lut, vec2(dot(material.normal, computed.v), material.roughness)).xy;

    return computed;
}

vec3 pbr_sun_light(PBRMaterial mat, PBRComputed comp, vec3 dir, vec3 flux) {
    return brdf(mat, comp, -dir) * flux;
}

vec3 f_schlick_a(float cos_theta, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(max(1.0 - cos_theta, 0.0), 5.0);
}

vec3 pbr_ibl(PBRMaterial mat, PBRComputed comp, samplerCube irrad, samplerCube prefilter, sampler2D dfg_lut) {
    vec3 Ld = texture(irrad, mat.normal).rgb * mat.albedo;
    vec3 Lld = textureLod(prefilter, reflect(-comp.v, mat.normal), mat.roughness * 8.0).rgb;
    vec2 dfg = texture(dfg_lut, vec2(max(dot(mat.normal, comp.v), 0.0), mat.roughness)).rg;
    vec3 Lr = Lld * (f_schlick_a(max(dot(mat.normal, comp.v), 0.0), comp.f0, mat.roughness) * dfg.x + dfg.y);
    return Ld + Lr;
}
