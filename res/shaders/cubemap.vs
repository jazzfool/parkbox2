#version 450

layout(location = 0) in vec3 in_pos;

layout(location = 0) out vec3 out_pos;

layout(binding = 0) uniform Transform {
    mat4 m;
};

void main() {
    out_pos = in_pos;
    gl_Position = m * vec4(out_pos, 1.0);
}
