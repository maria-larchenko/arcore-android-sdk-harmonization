#version 300 es
precision mediump float;
uniform sampler2D u_ColorTex;   // from virtualSceneFramebuffer
uniform sampler2D u_MaskTex;    // from virtualMaskFramebuffer
in  vec2 v_Tex;
out vec4 o_FragColor;
void main() {
    vec3 rgb  = texture(u_ColorTex, v_Tex).rgb;
    float a   = 0.8 * texture(u_MaskTex,  v_Tex).r + 0.2;   // red channel = mask
    o_FragColor = vec4(rgb, a);
}