s = open('engine.py').read()

s = s.replace('''    vec4 tx = texture(u_tex, v_uv);
    if(v_params.w > 0.5){
        col = tx.rgb * emis;
    } else {
        col += base * emis;
    }
    frag = vec4(col, 1.0);''','''    vec4 tx = texture(u_tex, v_uv);
    float alpha = 1.0;
    if(v_params.w > 0.5){
        col = tx.rgb * emis;
        alpha = tx.a;
    } else {
        col += base * emis;
    }
    frag = vec4(col, alpha);''')

s = s.replace('''uniform vec2 u_res;
in vec2 v_uv; out vec4 frag;

vec3 aces''','''uniform vec2 u_res;
uniform vec3 u_bg_in, u_bg_out;
in vec2 v_uv; out vec4 frag;

vec3 aces''')

s = s.replace('''void main(){
    vec3 c = texture(u_col, v_uv).rgb;
    c += texture(u_bloom, v_uv).rgb * u_bloom_amt;''','''void main(){
    vec4 sc = texture(u_col, v_uv);
    vec3 c = sc.rgb;
    vec2 q = (v_uv - vec2(0.5, 0.52)) * vec2(1.0, 1.15);
    float rad = length(q);
    float g = smoothstep(0.05, 0.80, rad);
    vec3 bgc = mix(u_bg_in, u_bg_out, g);
    bgc += u_bg_in * 0.35 * exp(-rad*rad*14.0);
    c = mix(bgc, c, clamp(sc.a, 0.0, 1.0));
    c += texture(u_bloom, v_uv).rgb * u_bloom_amt;''')

# set bg uniforms in composite
s = s.replace('''        f["u_bloom_amt"].value = float(scene.get("bloom", 0.55))''','''        f["u_bloom_amt"].value = float(scene.get("bloom", 0.55))
        bgc = scene.get("bg_colors", ((0.05, 0.06, 0.09), (0.005, 0.006, 0.012)))
        f["u_bg_in"].value = tuple(bgc[0])
        f["u_bg_out"].value = tuple(bgc[1])''')

# main pass clear to transparent
s = s.replace('''        bg = scene.get("bg", (0.02, 0.025, 0.035))
        self.ctx.clear(*bg, 1.0, depth=1.0)''','''        bg = scene.get("bg", (0.0, 0.0, 0.0))
        self.ctx.clear(bg[0], bg[1], bg[2], 0.0, depth=1.0)''')
open('engine.py','w').write(s)
print("ok")
