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
