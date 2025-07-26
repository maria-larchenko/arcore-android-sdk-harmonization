#version 300 es
precision mediump float;
layout(location = 0) in vec2 a_ScreenPos;   // [-1,+1] quad
layout(location = 1) in vec2 a_TexCoord;    // [0,1]  quad
out vec2 v_Tex;
void main() {
    v_Tex = a_TexCoord;
    gl_Position = vec4(a_ScreenPos, 0.0, 1.0);
}