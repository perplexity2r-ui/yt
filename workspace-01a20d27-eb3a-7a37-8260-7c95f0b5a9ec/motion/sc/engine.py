"""
Tiny premium-looking 3D renderer (moderngl / OpenGL 3.3 core, llvmpipe-friendly).
Renders frames to numpy -> PNG. Designed for vertical 3D motion-graphics shorts.
"""
import os, math, json
import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = "/home/user/assets/fonts"

# ---------------------------------------------------------------- math helpers
def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

def ortho(l, r, b, t, n, f):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2 / (r - l); m[1, 1] = 2 / (t - b); m[2, 2] = -2 / (f - n)
    m[0, 3] = -(r + l) / (r - l); m[1, 3] = -(t + b) / (t - b); m[2, 3] = -(f + n) / (f - n)
    return m

def norm(v):
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v)
    return v / (n if n > 1e-9 else 1.0)

def look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, np.float64); target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    f = norm(target - eye)
    s = norm(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
    m[0, 3] = -s @ eye; m[1, 3] = -u @ eye; m[2, 3] = f @ eye
    return m

def T(x, y, z):
    m = np.eye(4, dtype=np.float32); m[0, 3] = x; m[1, 3] = y; m[2, 3] = z; return m

def S(x, y=None, z=None):
    y = x if y is None else y; z = x if z is None else z
    m = np.eye(4, dtype=np.float32); m[0, 0] = x; m[1, 1] = y; m[2, 2] = z; return m

def RX(a):
    c, s = math.cos(a), math.sin(a); m = np.eye(4, dtype=np.float32)
    m[1, 1] = c; m[1, 2] = -s; m[2, 1] = s; m[2, 2] = c; return m

def RY(a):
    c, s = math.cos(a), math.sin(a); m = np.eye(4, dtype=np.float32)
    m[0, 0] = c; m[0, 2] = s; m[2, 0] = -s; m[2, 2] = c; return m

def RZ(a):
    c, s = math.cos(a), math.sin(a); m = np.eye(4, dtype=np.float32)
    m[0, 0] = c; m[0, 1] = -s; m[1, 0] = s; m[1, 1] = c; return m

def ease(t, kind="smooth"):
    t = min(max(t, 0.0), 1.0)
    if kind == "linear": return t
    if kind == "smooth": return t * t * (3 - 2 * t)
    if kind == "out":    return 1 - (1 - t) ** 3
    if kind == "in":     return t ** 3
    if kind == "inout":  return 4 * t ** 3 if t < .5 else 1 - (-2 * t + 2) ** 3 / 2
    if kind == "back":   # overshoot
        c1, c3 = 1.70158, 2.70158
        return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
    return t

def seg(t, t0, t1, kind="smooth"):
    if t1 <= t0: return 1.0 if t >= t1 else 0.0
    return ease((t - t0) / (t1 - t0), kind)

# ---------------------------------------------------------------- mesh builders
def _fix_winding(pos, nrm, idx):
    """Guarantee CCW outward winding for every triangle (front faces)."""
    P = np.asarray(pos, np.float64); N = np.asarray(nrm, np.float64)
    I = np.asarray(idx).reshape(-1, 3).copy()
    a, b, c = P[I[:, 0]], P[I[:, 1]], P[I[:, 2]]
    fn = np.cross(b - a, c - a)
    vn = N[I[:, 0]] + N[I[:, 1]] + N[I[:, 2]]
    flip = np.einsum("ij,ij->i", fn, vn) < 0
    I[flip] = I[flip][:, [0, 2, 1]]
    return I.reshape(-1).astype(np.uint32)

def _interleave(pos, nrm, uv=None, idx=None):
    pos = np.asarray(pos, np.float32); nrm = np.asarray(nrm, np.float32)
    if uv is None: uv = np.zeros((len(pos), 2), np.float32)
    uv = np.asarray(uv, np.float32)
    data = np.hstack([pos, nrm, uv]).astype(np.float32)
    idx = _fix_winding(pos, nrm, idx)
    return data, idx

def mesh_box(size=(1, 1, 1)):
    sx, sy, sz = [s / 2 for s in size]
    p = [(-sx,-sy, sz),( sx,-sy, sz),( sx, sy, sz),(-sx, sy, sz),
         (-sx,-sy,-sz),(-sx, sy,-sz),( sx, sy,-sz),( sx,-sy,-sz),
         (-sx, sy,-sz),(-sx, sy, sz),( sx, sy, sz),( sx, sy,-sz),
         (-sx,-sy,-sz),( sx,-sy,-sz),( sx,-sy, sz),(-sx,-sy, sz),
         ( sx,-sy,-sz),( sx, sy,-sz),( sx, sy, sz),( sx,-sy, sz),
         (-sx,-sy,-sz),(-sx,-sy, sz),(-sx, sy, sz),(-sx, sy,-sz)]
    n = [(0,0,1)]*4 + [(0,0,-1)]*4 + [(0,1,0)]*4 + [(0,-1,0)]*4 + [(1,0,0)]*4 + [(-1,0,0)]*4
    idx = []
    for f in range(6):
        b = f * 4
        idx += [b, b+1, b+2, b, b+2, b+3]
    return _interleave(p, n, None, idx)

def mesh_rounded_box(size=(1, 1, 1), r=0.12, seg=6):
    """Rounded box built from a subdivided cube projected onto the rounded surface."""
    hx, hy, hz = [s / 2 for s in size]
    r = min(r, min(size) * 0.49)
    verts = {}
    pos, nrm, idx = [], [], []
    # build 6 faces grid
    N = seg * 2 + 2
    def fpos(u, v, axis, sign):
        # u,v in [0,1] over the face
        a = (u * 2 - 1); b = (v * 2 - 1)
        if axis == 0:
            p = (sign * hx, b * hy, a * hz)
        elif axis == 1:
            p = (a * hx, sign * hy, b * hz)
        else:
            p = (a * hx, b * hy, sign * hz)
        return np.array(p, np.float64)
    for axis in range(3):
        for sign in (1, -1):
            base = len(pos)
            for i in range(N + 1):
                for j in range(N + 1):
                    pb = fpos(i / N, j / N, axis, sign)
                    inner = np.array([max(-(hx - r), min(hx - r, pb[0])),
                                      max(-(hy - r), min(hy - r, pb[1])),
                                      max(-(hz - r), min(hz - r, pb[2]))])
                    d = pb - inner
                    ln = np.linalg.norm(d)
                    n = d / ln if ln > 1e-9 else np.array([0, 0, 1.0])
                    pos.append(inner + n * r)
                    nrm.append(n)
            for i in range(N):
                for j in range(N):
                    a0 = base + i * (N + 1) + j
                    a1 = a0 + 1
                    b0 = a0 + (N + 1)
                    b1 = b0 + 1
                    idx += [a0, a1, b1, a0, b1, b0]
    return _interleave(pos, nrm, None, idx)

def mesh_cylinder(r=0.5, h=1.0, segs=48, caps=True):
    pos, nrm, idx = [], [], []
    for i in range(segs + 1):
        a = 2 * math.pi * i / segs
        c, s = math.cos(a), math.sin(a)
        pos += [(r * c, -h / 2, r * s), (r * c, h / 2, r * s)]
        nrm += [(c, 0, s), (c, 0, s)]
    for i in range(segs):
        b = i * 2
        idx += [b, b + 2, b + 3, b, b + 3, b + 1]
    if caps:
        for sign in (1, -1):
            base = len(pos)
            pos.append((0, sign * h / 2, 0)); nrm.append((0, sign, 0))
            for i in range(segs + 1):
                a = 2 * math.pi * i / segs
                pos.append((r * math.cos(a), sign * h / 2, r * math.sin(a)))
                nrm.append((0, sign, 0))
            for i in range(segs):
                if sign > 0: idx += [base, base + 1 + i, base + 2 + i]
                else:        idx += [base, base + 2 + i, base + 1 + i]
    return _interleave(pos, nrm, None, idx)

def mesh_plane(w=1, h=1):
    pos = [(-w/2, -h/2, 0), (w/2, -h/2, 0), (w/2, h/2, 0), (-w/2, h/2, 0)]
    nrm = [(0, 0, 1)] * 4
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return _interleave(pos, nrm, uv, [0, 1, 2, 0, 2, 3])

def mesh_sphere(r=0.5, segs=48, rings=32):
    pos, nrm, idx = [], [], []
    for i in range(rings + 1):
        v = i / rings; phi = math.pi * v
        for j in range(segs + 1):
            u = j / segs; th = 2 * math.pi * u
            n = (math.sin(phi) * math.cos(th), math.cos(phi), math.sin(phi) * math.sin(th))
            pos.append((n[0] * r, n[1] * r, n[2] * r)); nrm.append(n)
    for i in range(rings):
        for j in range(segs):
            a = i * (segs + 1) + j; b = a + segs + 1
            idx += [a, b, b + 1, a, b + 1, a + 1]
    return _interleave(pos, nrm, None, idx)

def mesh_torus(R=0.5, r=0.15, segs=64, sides=24):
    pos, nrm, idx = [], [], []
    for i in range(segs + 1):
        u = 2 * math.pi * i / segs
        cu, su = math.cos(u), math.sin(u)
        for j in range(sides + 1):
            v = 2 * math.pi * j / sides
            cv, sv = math.cos(v), math.sin(v)
            n = (cu * cv, sv, su * cv)
            pos.append(((R + r * cv) * cu, r * sv, (R + r * cv) * su)); nrm.append(n)
    for i in range(segs):
        for j in range(sides):
            a = i * (sides + 1) + j; b = a + sides + 1
            idx += [a, b, b + 1, a, b + 1, a + 1]
    return _interleave(pos, nrm, None, idx)

# ---------------------------------------------------------------- shaders
VS_MAIN = """
#version 330
uniform mat4 u_view, u_proj;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
in mat4 i_model; in vec4 i_color; in vec4 i_params; // metal, rough, emissive, texflags
out vec3 v_wpos; out vec3 v_nrm; out vec2 v_uv; out vec4 v_color; out vec4 v_params;
void main(){
    vec4 wp = i_model * vec4(in_pos, 1.0);
    v_wpos = wp.xyz;
    v_nrm = normalize(transpose(inverse(mat3(i_model))) * in_nrm);
    v_uv = in_uv; v_color = i_color; v_params = i_params;
    gl_Position = u_proj * u_view * wp;
}
"""
FS_MAIN = """
#version 330
uniform vec3 u_campos;
uniform vec3 u_l0dir, u_l0col, u_l1dir, u_l1col, u_l2dir, u_l2col;
uniform vec3 u_amb_top, u_amb_bot, u_rimcol;
uniform mat4 u_lightVP;
uniform sampler2D u_shadow;
uniform float u_shadow_on;
uniform sampler2D u_tex;
uniform float u_time;
uniform vec3 u_fog_col;
uniform float u_fog_density;
in vec3 v_wpos; in vec3 v_nrm; in vec2 v_uv; in vec4 v_color; in vec4 v_params;
out vec4 frag;

float ggx(vec3 N, vec3 V, vec3 L, float rough){
    vec3 H = normalize(V + L);
    float a = max(rough*rough, 0.0025);
    float NdH = max(dot(N,H), 0.0), NdL = max(dot(N,L), 0.0), NdV = max(dot(N,V), 0.0);
    float d = NdH*NdH*(a*a-1.0)+1.0;
    float D = a*a/(3.14159*d*d);
    float k = a*0.5;
    float G = (NdL/(NdL*(1.0-k)+k)) * (NdV/(NdV*(1.0-k)+k));
    vec3 F0 = mix(vec3(0.04), v_color.rgb, v_params.x);
    vec3 F = F0 + (1.0-F0)*pow(1.0-max(dot(H,V),0.0), 5.0);
    return max(dot(D*G*F, vec3(1.0))/(4.0*NdL*NdV+0.0001), 0.0) * NdL;
}

float shadow(vec3 wp, vec3 n){
    if(u_shadow_on < 0.5) return 1.0;
    vec3 p = wp + n*0.012;
    vec4 sp = u_lightVP * vec4(p, 1.0);
    vec3 c = sp.xyz / sp.w * 0.5 + 0.5;
    if(c.x<0.001||c.x>0.999||c.y<0.001||c.y>0.999||c.z>1.0) return 1.0;
    float bias = 0.0015;
    float s = 0.0;
    vec2 texel = vec2(1.0/2048.0);
    for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++){
        float d = texture(u_shadow, c.xy + vec2(x,y)*texel*1.4).r;
        s += (c.z - bias > d) ? 0.0 : 1.0;
    }
    return s/9.0;
}

void main(){
    vec3 N = normalize(v_nrm);
    vec3 V = normalize(u_campos - v_wpos);
    if(dot(N, V) < 0.0 && v_params.x > 1.5) N = -N; // unused
    float rough = clamp(v_params.y, 0.03, 1.0);
    float metal = v_params.x;
    float emis = v_params.z;

    vec3 base = v_color.rgb;
    // subtle procedural variation (noise) to avoid plastic feel
    float n = fract(sin(dot(v_wpos.xz*7.3 + v_wpos.yy*3.1, vec2(12.9898,78.233)))*43758.5453);
    base *= 0.96 + 0.08*n;

    vec3 col = vec3(0.0);
    vec3 Ls[3]; Ls[0]=normalize(u_l0dir); Ls[1]=normalize(u_l1dir); Ls[2]=normalize(u_l2dir);
    vec3 Cs[3]; Cs[0]=u_l0col; Cs[1]=u_l1col; Cs[2]=u_l2col;
    float sh = shadow(v_wpos, N);
    for(int i=0;i<3;i++){
        vec3 L = Ls[i];
        float NdL = max(dot(N,L), 0.0);
        if(NdL <= 0.0 && i==0) continue;
        float s = (i==0) ? mix(0.15, 1.0, sh) : 1.0;
        vec3 diff = base * (1.0-metal) / 3.14159 * NdL;
        vec3 spec = ggx(N,V,L,rough) * Cs[i];
        col += (diff + spec) * Cs[i] * s;
    }
    // ambient / env approximation
    float up = N.y*0.5+0.5;
    vec3 amb = mix(u_amb_bot, u_amb_top, up);
    float ao = clamp(0.55 + 0.45*N.y, 0.0, 1.0);
    col += base * amb * ao * (1.0-metal*0.6);
    // fresnel rim for that premium edge glow
    float fres = pow(1.0 - max(dot(N,V),0.0), 4.0);
    col += u_rimcol * fres * (0.25 + 0.75*metal) * 0.5;
    // fake reflection of key light for metals
    vec3 R = reflect(-V, N);
    float envSpec = pow(max(dot(R, normalize(vec3(0.2,0.9,0.4))),0.0), 6.0);
    col += u_l0col * envSpec * (0.15 + 0.85*metal) * (1.0-rough*0.8) * 0.6;

    // emissive texture
    vec4 tx = texture(u_tex, v_uv);
    float alpha = 1.0;
    if(v_params.w > 0.5){
        col = tx.rgb * emis;
        alpha = tx.a;
    } else {
        col += base * emis;
    }
    // distance fog -> melts the floor into the backdrop
    float dist = length(u_campos - v_wpos);
    float fg = 1.0 - exp(-pow(max(dist - 3.0, 0.0) * u_fog_density, 2.0));
    col = mix(col, u_fog_col, clamp(fg, 0.0, 1.0));
    alpha = mix(alpha, 1.0, clamp(fg, 0.0, 1.0) * step(0.999, alpha));
    frag = vec4(col, alpha);
}
"""

VS_SHADOW = """
#version 330
uniform mat4 u_lightVP;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
in mat4 i_model; in vec4 i_color; in vec4 i_params;
void main(){ gl_Position = u_lightVP * i_model * vec4(in_pos,1.0); }
"""
FS_SHADOW = """
#version 330
void main(){ }
"""

VS_QUAD = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos,0.0,1.0); }
"""
FS_DOF = """
#version 330
uniform sampler2D u_col; uniform sampler2D u_dep;
uniform vec2 u_texel; uniform vec2 u_dir;
uniform float u_focus, u_aperture, u_near, u_far;
in vec2 v_uv; out vec4 frag;
float lin(float d){
    float z = d*2.0-1.0;
    return (2.0*u_near*u_far)/(u_far+u_near - z*(u_far-u_near));
}
void main(){
    float z = lin(texture(u_dep, v_uv).r);
    if(z > 40.0) z = 40.0;
    float coc = clamp(abs(z - u_focus) / max(z,0.001) * u_aperture, 0.0, 1.0);
    vec3 sum = vec3(0.0); float wsum = 0.0;
    const int N = 12;
    for(int i=0;i<N;i++){
        float t = (float(i)+0.5)/float(N) * 2.0 - 1.0;
        vec2 off = u_dir * u_texel * t * (1.0 + 7.0*coc);
        sum += texture(u_col, v_uv + off).rgb;
        wsum += 1.0;
    }
    frag = vec4(sum/wsum, texture(u_col, v_uv).a);
}
"""
FS_BRIGHT = """
#version 330
uniform sampler2D u_col; uniform float u_thresh;
in vec2 v_uv; out vec4 frag;
void main(){
    vec3 c = texture(u_col, v_uv).rgb;
    float l = max(max(c.r,c.g),c.b);
    float k = max(l - u_thresh, 0.0) / max(l, 0.0001);
    frag = vec4(c*k, 1.0);
}
"""
FS_BLUR = """
#version 330
uniform sampler2D u_col; uniform vec2 u_texel; uniform vec2 u_dir;
in vec2 v_uv; out vec4 frag;
void main(){
    vec3 s = vec3(0.0); float w = 0.0;
    for(int i=-6;i<=6;i++){
        float fi = float(i);
        float k = exp(-fi*fi/18.0);
        s += texture(u_col, v_uv + u_dir*u_texel*fi*1.6).rgb * k;
        w += k;
    }
    frag = vec4(s/w, 1.0);
}
"""
FS_FINAL = """
#version 330
uniform sampler2D u_col, u_bloom;
uniform float u_bloom_amt, u_exposure, u_vignette, u_grain, u_time, u_dof_amt;
uniform vec2 u_res;
uniform vec3 u_bg_in, u_bg_out, u_glow_col;
uniform vec2 u_glow_pos;
uniform float u_glow_int;
in vec2 v_uv; out vec4 frag;

vec3 aces(vec3 x){
    return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0);
}
void main(){
    vec4 sc = texture(u_col, v_uv);
    vec3 c = sc.rgb;
    vec2 q = (v_uv - vec2(0.5, 0.52)) * vec2(1.0, 1.15);
    float rad = length(q);
    float gr = smoothstep(0.05, 0.80, rad);
    vec3 bgc = mix(u_bg_in, u_bg_out, gr);
    vec2 gq = (v_uv - u_glow_pos) * vec2(1.0, 1.25);
    bgc += u_glow_col * u_glow_int * exp(-dot(gq, gq) * 9.0);
    bgc += u_glow_col * u_glow_int * 0.35 * exp(-dot(gq, gq) * 2.0);
    c = mix(bgc, c, clamp(sc.a, 0.0, 1.0));
    c += texture(u_bloom, v_uv).rgb * u_bloom_amt;
    c = aces(c * u_exposure);
    c = pow(c, vec3(1.0/2.2));
    // chromatic aberration near edges
    vec2 d = v_uv - 0.5;
    float ca = dot(d,d)*0.0035;
    float r = texture(u_col, v_uv + d*ca).r;
    c.r = mix(c.r, pow(aces(texture(u_col, v_uv + d*ca).rgb*u_exposure).r, 1.0/2.2), 0.7);
    // vignette
    float vig = smoothstep(0.95, 0.25, length(d*vec2(1.05,0.95)));
    c *= mix(1.0, vig, u_vignette);
    // grain
    float g = fract(sin(dot(v_uv*u_res + u_time*60.0, vec2(12.9898,78.233)))*43758.5453);
    c += (g-0.5)*u_grain;
    frag = vec4(c, 1.0);
}
"""

class Renderer:
    def __init__(self, W=1080, H=1920, shadow=2048):
        self.W, self.H = W, H
        self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.cull_face = "back"
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.prog = self.ctx.program(vertex_shader=VS_MAIN, fragment_shader=FS_MAIN)
        self.sp = self.ctx.program(vertex_shader=VS_SHADOW, fragment_shader=FS_SHADOW)
        self.dofp = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_DOF)
        self.bright = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_BRIGHT)
        self.blurp = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_BLUR)
        self.final = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_FINAL)

        quad = np.array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1], np.float32)
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        self.quad_vaos = {}
        for k, p in [("dof", self.dofp), ("bright", self.bright), ("blur", self.blurp), ("final", self.final)]:
            self.quad_vaos[k] = self.ctx.vertex_array(p, [(self.quad_vbo, "2f", "in_pos")])

        # meshes
        self.meshes = {
            "box": mesh_box((1, 1, 1)),
            "rbox": mesh_rounded_box((1, 1, 1), r=0.16, seg=5),
            "cyl": mesh_cylinder(0.5, 1.0, 40),
            "plane": mesh_plane(1, 1),
            "sphere": mesh_sphere(0.5, 40, 24),
            "torus": mesh_torus(0.4, 0.12, 48, 20),
        }
        self.gpu = {}
        self.inst_cache = {}
        for name, (data, idx) in self.meshes.items():
            vbo = self.ctx.buffer(data.tobytes())
            ibo = self.ctx.buffer(idx.tobytes())
            maxinst = 512
            self.inst_cache[name] = np.zeros((maxinst, 24), np.float32)
            inst = self.ctx.buffer(reserve=maxinst * 24 * 4)
            vao = self.ctx.vertex_array(self.prog, [
                (vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv"),
                (inst, "16f 4f 4f /i", "i_model", "i_color", "i_params"),
            ], index_buffer=ibo)
            svao = self.ctx.vertex_array(self.sp, [
                (vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv"),
                (inst, "16f 4f 4f /i", "i_model", "i_color", "i_params"),
            ], index_buffer=ibo, skip_errors=True)
            self.gpu[name] = dict(vbo=vbo, ibo=ibo, inst=inst, vao=vao, svao=svao,
                                  count=len(idx), maxinst=maxinst)

        # shadow map
        self.shadow_size = shadow
        self.shadow_tex = self.ctx.depth_texture((shadow, shadow))
        self.shadow_tex.compare_func = ""
        self.shadow_fbo = self.ctx.framebuffer(depth_attachment=self.shadow_tex)

        # HDR targets
        self.hdr = self.ctx.texture((W, H), 4, dtype="f2")
        self.hdr.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hdr.repeat_x = self.hdr.repeat_y = False
        self.depth_tex = self.ctx.depth_texture((W, H))
        self.fbo = self.ctx.framebuffer(color_attachments=[self.hdr], depth_attachment=self.depth_tex)

        hw, hh = W // 2, H // 2
        self.dof_a = self.ctx.texture((hw, hh), 4, dtype="f2"); self.dof_a.filter = (moderngl.LINEAR,)*2
        self.dof_b = self.ctx.texture((hw, hh), 4, dtype="f2"); self.dof_b.filter = (moderngl.LINEAR,)*2
        self.dof_fbo_a = self.ctx.framebuffer([self.dof_a])
        self.dof_fbo_b = self.ctx.framebuffer([self.dof_b])

        bw, bh = W // 4, H // 4
        self.bloom_tex = []
        self.bloom_fbo = []
        for i in range(3):
            w = max(bw >> i, 8); h = max(bh >> i, 8)
            t1 = self.ctx.texture((w, h), 4, dtype="f2"); t1.filter = (moderngl.LINEAR,)*2
            t2 = self.ctx.texture((w, h), 4, dtype="f2"); t2.filter = (moderngl.LINEAR,)*2
            self.bloom_tex.append((t1, t2))
            self.bloom_fbo.append((self.ctx.framebuffer([t1]), self.ctx.framebuffer([t2])))

        self.out_tex = self.ctx.texture((W, H), 3)
        self.out_fbo = self.ctx.framebuffer([self.out_tex])

        # default sampler bindings
        self._bind_samplers()

    def _bind_samplers(self):
        for p, names in [(self.prog, {"u_shadow": 0, "u_tex": 1}),
                         (self.dofp, {"u_col": 0, "u_dep": 1}),
                         (self.bright, {"u_col": 0}),
                         (self.blurp, {"u_col": 0}),
                         (self.final, {"u_col": 0, "u_bloom": 1})]:
            for k, v in names.items():
                if k in p:
                    p[k].value = v

    # -------------------------------------------------- textures from PIL images
    def texture_from_image(self, img):
        img = img.convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
        t = self.ctx.texture(img.size, 4, img.tobytes())
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
        return t

    # -------------------------------------------------- main render
    def render(self, scene):
        W, H = self.W, self.H
        cam = scene["camera"]
        eye = np.array(cam["pos"], np.float64)
        target = np.array(cam.get("target", (0, 0, 0)), np.float64)
        up = cam.get("up", (0, 1, 0))
        view = look_at(eye, target, up)
        view = RX_(view, cam.get("roll", 0.0))
        proj = perspective(cam.get("fov", 38), W / H, cam.get("near", 0.05), cam.get("far", 200))

        lights = scene.get("lights", {})
        L = lights.get("list", [
            ((0.5, 1.0, 0.55), (1.0, 0.97, 0.92)),
            ((-0.8, 0.4, 0.6), (0.35, 0.45, 0.7)),
            ((0.0, -0.5, -1.0), (0.25, 0.3, 0.45)),
        ])
        for i in range(3, 0, -1):
            pass
        l0dir = norm(L[0][0]); l0col = np.array(L[0][1], np.float32) * lights.get("key_int", 3.0)
        l1dir = norm(L[1][0]); l1col = np.array(L[1][1], np.float32) * lights.get("fill_int", 1.1)
        l2dir = norm(L[2][0]); l2col = np.array(L[2][1], np.float32) * lights.get("rim_int", 1.2)

        # light space for shadows (directional along -l0dir)
        center = np.array(scene.get("shadow_center", (0, 0, 0)), np.float64)
        extent = scene.get("shadow_extent", 9.0)
        eyeL = center - l0dir * 20.0
        lview = look_at(eyeL, center, (0, 1, 0))
        lproj = ortho(-extent, extent, -extent, extent, 1.0, 45.0)
        lightVP = lproj @ lview

        objects = scene["objects"]
        # build instance data per mesh type
        per_mesh = {}
        textured = []
        for o in objects:
            if o.get("hidden"): continue
            mesh = o.get("mesh", "rbox")
            if mesh == "tex":
                textured.append(o); continue
            M = o["model"] if "model" in o else _compose(o)
            if "scale" in o and "model" not in o:
                pass
            tint = np.array(list(o.get("color", (0.8, 0.8, 0.8))) + [1.0], np.float32)
            params = np.array([o.get("metal", 0.0), o.get("rough", 0.45), o.get("emissive", 0.0),
                               0.0], np.float32)
            per_mesh.setdefault(mesh, []).append((M, tint, params))

        # shadow pass
        self.shadow_fbo.use()
        self.ctx.clear(depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.sp["u_lightVP"].write(lightVP.T.tobytes())
        for mesh, items in per_mesh.items():
            g = self.gpu[mesh]
            if len(items) > g["maxinst"]: items = items[:g["maxinst"]]
            self._fill_inst(g, items)
            g["svao"].render(moderngl.TRIANGLES, instances=len(items))
        self.ctx.enable(moderngl.CULL_FACE)

        # main pass
        self.fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        bg = scene.get("bg", (0.0, 0.0, 0.0))
        self.ctx.clear(bg[0], bg[1], bg[2], 0.0, depth=1.0)
        p = self.prog
        p["u_view"].write(view.T.tobytes())
        p["u_proj"].write(proj.T.tobytes())
        p["u_campos"].value = tuple(eye.astype(np.float32))
        p["u_l0dir"].value = tuple(l0dir.astype(np.float32))
        p["u_l0col"].value = tuple(l0col.astype(np.float32))
        p["u_l1dir"].value = tuple(l1dir.astype(np.float32))
        p["u_l1col"].value = tuple(l1col.astype(np.float32))
        p["u_l2dir"].value = tuple(l2dir.astype(np.float32))
        p["u_l2col"].value = tuple(l2col.astype(np.float32))
        amb = scene.get("ambient", {})
        p["u_amb_top"].value = tuple(amb.get("top", (0.10, 0.12, 0.16)))
        p["u_amb_bot"].value = tuple(amb.get("bot", (0.02, 0.02, 0.03)))
        p["u_rimcol"].value = tuple(amb.get("rim", (0.35, 0.45, 0.6)))
        p["u_lightVP"].write(lightVP.T.tobytes())
        p["u_shadow"].value = 0
        p["u_tex"].value = 1
        p["u_shadow_on"].value = 1.0 if scene.get("shadows", True) else 0.0
        fog = scene.get("fog", ((0.012, 0.014, 0.021), 0.055))
        p["u_fog_col"].value = tuple(fog[0])
        p["u_fog_density"].value = float(fog[1])
        if "u_time" in p:
            p["u_time"].value = float(scene.get("time", 0.0))
        self.shadow_tex.use(0)
        for mesh, items in per_mesh.items():
            g = self.gpu[mesh]
            if len(items) > g["maxinst"]: items = items[:g["maxinst"]]
            self._fill_inst(g, items)
            g["vao"].render(moderngl.TRIANGLES, instances=len(items))

        # textured / unlit planes
        if textured:
            self.ctx.disable(moderngl.CULL_FACE)
            for o in textured:
                M = o["model"] if "model" in o else _compose(o)
                items = [(M, np.array([1, 1, 1, 1], np.float32),
                          np.array([0.0, 0.5, o.get("emissive", 1.0), 1.0], np.float32))]
                g = self.gpu[o.get("base", "plane")]
                self._fill_inst(g, items)
                o["texture"].use(1)
                g["vao"].render(moderngl.TRIANGLES, instances=1)
            self.ctx.enable(moderngl.CULL_FACE)

        # ---- DOF
        camdof = cam.get("dof", {})
        focus = camdof.get("focus", 8.0); aperture = camdof.get("aperture", 0.0)
        hw, hh = W // 2, H // 2
        self.dof_fbo_a.use(); self.ctx.viewport = (0, 0, hw, hh)
        self.hdr.use(0); self.depth_tex.use(1)
        d = self.dofp
        d["u_texel"].value = (2.0 / W, 2.0 / H)
        d["u_dir"].value = (1.0, 0.0)
        d["u_focus"].value = float(focus); d["u_aperture"].value = float(aperture)
        d["u_near"].value = float(cam.get("near", 0.05)); d["u_far"].value = float(cam.get("far", 200))
        self.quad_vaos["dof"].render(moderngl.TRIANGLES, vertices=6)
        self.dof_fbo_b.use()
        self.dof_a.use(0)
        d["u_dir"].value = (0.0, 1.0)
        self.quad_vaos["dof"].render(moderngl.TRIANGLES, vertices=6)

        # ---- bloom
        bw, bh = W // 4, H // 4
        self.bloom_fbo[0][0].use(); self.ctx.viewport = (0, 0, bw, bh)
        self.dof_b.use(0)  # bloom from DOF'd HDR
        self.bright["u_thresh"].value = float(scene.get("bloom_thresh", 0.9))
        self.quad_vaos["bright"].render(moderngl.TRIANGLES, vertices=6)
        for i in range(3):
            t1, t2 = self.bloom_tex[i]
            f1, f2 = self.bloom_fbo[i]
            src = self.bloom_tex[i - 1][0] if i > 0 else self.bloom_tex[0][0]
            w = max(bw >> i, 8); h = max(bh >> i, 8)
            f1.use(); self.ctx.viewport = (0, 0, w, h)
            src.use(0)
            self.blurp["u_texel"].value = (1.0 / w, 1.0 / h)
            self.blurp["u_dir"].value = (1.0, 0.0)
            self.quad_vaos["blur"].render(moderngl.TRIANGLES, vertices=6)
            f2.use()
            t1.use(0)
            self.blurp["u_dir"].value = (0.0, 1.0)
            self.quad_vaos["blur"].render(moderngl.TRIANGLES, vertices=6)
            b = self.blurp
            f1.use()
            t2.use(0)
            b["u_dir"].value = (1.0, 0.0)
            self.quad_vaos["blur"].render(moderngl.TRIANGLES, vertices=6)
            self.bloom_tex[i] = (t2, t1)
            self.bloom_fbo[i] = (f2, f1)

        # ---- composite
        self.out_fbo.use(); self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.dof_b.use(0)
        self.bloom_tex[1][0].use(1)
        f = self.final
        f["u_bloom_amt"].value = float(scene.get("bloom", 0.55))
        bgc = scene.get("bg_colors", ((0.05, 0.06, 0.09), (0.005, 0.006, 0.012)))
        f["u_bg_in"].value = tuple(bgc[0])
        f["u_bg_out"].value = tuple(bgc[1])
        gl = scene.get("bg_glow", ((0.10, 0.20, 0.38), (0.5, 0.60), 0.55))
        f["u_glow_col"].value = tuple(gl[0])
        f["u_glow_pos"].value = tuple(gl[1])
        f["u_glow_int"].value = float(gl[2])
        f["u_exposure"].value = float(scene.get("exposure", 1.0))
        f["u_vignette"].value = float(scene.get("vignette", 0.55))
        f["u_grain"].value = float(scene.get("grain", 0.012))
        f["u_time"].value = float(scene.get("time", 0.0))
        if "u_dof_amt" in f: f["u_dof_amt"].value = 1.0
        f["u_res"].value = (float(W), float(H))
        self.quad_vaos["final"].render(moderngl.TRIANGLES, vertices=6)
        self.ctx.enable(moderngl.DEPTH_TEST)

        data = self.out_fbo.read(components=3, alignment=1)
        img = Image.frombytes("RGB", (W, H), data).transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def _fill_inst(self, g, items):
        arr = self.inst_cache[g["vao"] if False else "box"] if False else None
        buf = np.zeros((len(items), 24), np.float32)
        for i, (M, col, params) in enumerate(items):
            buf[i, :16] = M.T.reshape(-1)
            buf[i, 16:20] = col
            buf[i, 20:24] = params
        g["inst"].write(buf.tobytes())


def RX_(view, roll):
    if abs(roll) < 1e-6: return view
    return RZ(roll) @ view


def _compose(o):
    """model matrix from pos/rot(radians)/size"""
    M = np.eye(4, dtype=np.float32)
    p = o.get("pos", (0, 0, 0))
    r = o.get("rot", (0, 0, 0))
    s = o.get("size", (1, 1, 1))
    if np.isscalar(s): s = (s, s, s)
    M = T(*p) @ RZ(r[2]) @ RY(r[1]) @ RX(r[0]) @ S(*s)
    return M
