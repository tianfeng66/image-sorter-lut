// GPU port of the per-pixel LUT stack (lut_engine.apply_look / apply_tune / apply_adjust_room)
// for instant previews while sliders move. The final guided filter is skipped; the server
// render replaces this canvas once the sliders settle.
const LUT_LIVE_MAX_LAYERS = 4;
const LUT_LIVE_MAX_MOVES = 64;

const LUT_LIVE_VS = `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const LUT_LIVE_FS = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_src;
uniform sampler2D u_room;
uniform int u_probe;
uniform int u_height;
uniform int u_n;
uniform int u_kind[${LUT_LIVE_MAX_LAYERS}];
uniform float u_strength[${LUT_LIVE_MAX_LAYERS}];
uniform vec4 u_tone[${LUT_LIVE_MAX_LAYERS}];
uniform vec3 u_sh[${LUT_LIVE_MAX_LAYERS}];
uniform vec3 u_hi[${LUT_LIVE_MAX_LAYERS}];
uniform vec3 u_mid[${LUT_LIVE_MAX_LAYERS}];
uniform float u_gray[${LUT_LIVE_MAX_LAYERS}];
uniform int u_wash[${LUT_LIVE_MAX_LAYERS}];
uniform ivec3 u_mstart[${LUT_LIVE_MAX_LAYERS}];
uniform ivec3 u_mcount[${LUT_LIVE_MAX_LAYERS}];
uniform vec3 u_moves[${LUT_LIVE_MAX_MOVES}];
uniform vec4 u_tt[${LUT_LIVE_MAX_LAYERS}];
uniform float u_tblack[${LUT_LIVE_MAX_LAYERS}];
uniform vec3 u_tmid[${LUT_LIVE_MAX_LAYERS}];
uniform float u_bands[${LUT_LIVE_MAX_LAYERS * 6}];
uniform int u_skin_only;
uniform vec2 u_look;
out vec4 o;

const float BAND_C[6] = float[6](0.0, 60.0, 120.0, 180.0, 240.0, 300.0);

float ss(float t) { t = clamp(t, 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
float luma(vec3 c) { return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b; }
float hdist(float a, float b) { return abs(mod(a - b + 180.0, 360.0) - 180.0); }
float hweight(float h, float c, float w) { return ss(1.0 - hdist(h, c) / max(w, 1e-6)); }

vec3 rgb2hsv(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  float mx = max(c.r, max(c.g, c.b));
  float mn = min(c.r, min(c.g, c.b));
  float df = mx - mn;
  float h = 0.0;
  if (df > 1e-8) {
    if (mx == c.r) h = mod((c.g - c.b) / df, 6.0);
    else if (mx == c.g) h = (c.b - c.r) / df + 2.0;
    else h = (c.r - c.g) / df + 4.0;
  }
  return vec3(h * 60.0, mx > 1e-8 ? df / mx : 0.0, mx);
}

vec3 hsv2rgb(float h, float s, float v) {
  h = mod(h, 360.0);
  float c = v * clamp(s, 0.0, 1.5);
  float x = c * (1.0 - abs(mod(h / 60.0, 2.0) - 1.0));
  float m = v - c;
  int sec = int(h / 60.0) % 6;
  vec3 rgb = sec == 0 ? vec3(c, x, 0.0) : sec == 1 ? vec3(x, c, 0.0) : sec == 2 ? vec3(0.0, c, x)
    : sec == 3 ? vec3(0.0, x, c) : sec == 4 ? vec3(x, 0.0, c) : vec3(c, 0.0, x);
  return clamp(rgb + m, 0.0, 1.0);
}

float tone(float y, float black, float gamma, float contrast, float sparkle) {
  float x = (y - black) / max(1.0 - black, 1e-6);
  x = pow(clamp(x, 0.0, 1.0), max(gamma, 0.05));
  x = clamp((x - 0.48) * contrast + 0.50, 0.0, 1.0);
  if (sparkle != 0.0) x += clamp((x - 0.78) / 0.22, 0.0, 1.0) * sparkle;
  return clamp(x, 0.0, 1.0);
}

vec2 srcStats(vec3 c) {
  float mx = max(c.r, max(c.g, c.b));
  float ch = mx - min(c.r, min(c.g, c.b));
  return vec2(luma(c), mx > 1e-5 ? ch / max(mx, 1e-5) : 0.0);
}

vec3 applyParams(vec3 src, int i) {
  src = clamp(src, 0.0, 1.0);
  vec4 t = u_tone[i];
  float y = luma(src);
  float y2 = tone(y, t.x, t.y, t.z, t.w);
  float scale = y > 1e-5 ? y2 / max(y, 1e-5) : 1.0;
  vec3 g = clamp(src * scale, 0.0, 1.0);
  float shw = clamp(1.0 - y2 / 0.42, 0.0, 1.0);
  float hiw = clamp((y2 - 0.62) / 0.38, 0.0, 1.0);
  float midw = (1.0 - shw) * (1.0 - hiw);
  shw *= ss((y - 0.05) / 0.10);
  g = clamp(g + shw * u_sh[i] + hiw * u_hi[i] + midw * u_mid[i], 0.0, 1.0);
  vec3 hsv = rgb2hsv(g);
  vec2 st = srcStats(src);
  float gate = 1.0;
  if (u_wash[i] == 1) {
    float notDark = ss((st.x - 0.055) / 0.11);
    float notPale = 1.0 - ss((st.x - 0.84) / 0.10) * ss(1.0 - st.y / 0.16);
    gate = notDark * notPale * ss((st.y - 0.03) / 0.13);
  }
  float dh = 0.0, ds = 0.0, dv = 0.0;
  for (int k = 0; k < u_mcount[i].x; k++) { vec3 m = u_moves[u_mstart[i].x + k]; dh += hweight(hsv.x, m.x, m.y) * gate * m.z; }
  for (int k = 0; k < u_mcount[i].y; k++) { vec3 m = u_moves[u_mstart[i].y + k]; ds += hweight(hsv.x, m.x, m.y) * gate * m.z; }
  for (int k = 0; k < u_mcount[i].z; k++) { vec3 m = u_moves[u_mstart[i].z + k]; dv += hweight(hsv.x, m.x, m.y) * gate * m.z; }
  g = hsv2rgb(hsv.x + dh, hsv.y, clamp(hsv.z * (1.0 + dv), 0.0, 1.0));
  float y3 = luma(g);
  g = y3 + (g - y3) * clamp(1.0 + ds, 0.0, 1.85);
  if (u_gray[i] > 1e-6) {
    vec3 h2 = rgb2hsv(g);
    float w = u_gray[i] * ss(1.0 - h2.y / 0.18);
    g = g * (1.0 - w) + h2.z * w;
  }
  return clamp(g, 0.0, 1.0);
}

vec3 applyLook(vec3 rgb, int i) {
  rgb = clamp(rgb, 0.0, 1.0);
  float s = clamp(u_strength[i], 0.0, 1.5);
  if (s <= 1e-6) return rgb;
  vec3 g = applyParams(rgb, i);
  if (s >= 0.999 && s <= 1.001) return g;
  return clamp(rgb * (1.0 - s) + g * s, 0.0, 1.0);
}

vec3 applyTune(vec3 rgb, int i) {
  vec3 src = clamp(rgb, 0.0, 1.0);
  float s = clamp(u_strength[i], 0.0, 1.5);
  if (s <= 1e-6) return src;
  vec4 t = u_tt[i];
  float y = luma(src);
  float y2 = tone(y, u_tblack[i] * s, 1.0 + (t.z - 1.0) * s, 1.0 + (t.w - 1.0) * s, 0.0);
  float scale = y > 1e-5 ? y2 / max(y, 1e-5) : 1.0;
  vec3 c = clamp(src * scale, 0.0, 1.0);
  float midw = (1.0 - clamp(1.0 - y2 / 0.42, 0.0, 1.0)) * (1.0 - clamp((y2 - 0.62) / 0.38, 0.0, 1.0));
  c = clamp(c + midw * u_tmid[i] * s, 0.0, 1.0);
  vec3 hsv = rgb2hsv(c);
  float dh = t.x * s;
  float gate = ss((hsv.y - 0.03) / 0.13);
  for (int b = 0; b < 6; b++) {
    float amount = u_bands[i * 6 + b] * s;
    if (abs(amount) >= 1e-6) dh += hweight(hsv.x, BAND_C[b], 78.0) * gate * amount;
  }
  c = hsv2rgb(hsv.x + dh, hsv.y, hsv.z);
  float y3 = luma(c);
  c = y3 + (c - y3) * (1.0 + t.y * s);
  return clamp(c, 0.0, 1.0);
}

vec3 applyStack(vec3 c) {
  for (int i = 0; i < u_n; i++) {
    if (u_kind[i] == 1) c = applyLook(c, i);
    else if (u_kind[i] == 2) c = applyTune(c, i);
  }
  return c;
}

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  if (u_probe == 1) {
    vec3 p = vec3(px.x == 0 ? 0.32 : px.x == 1 ? 0.50 : 0.68);
    o = vec4(clamp((applyStack(p) - p) * 4.0 + 0.5, 0.0, 1.0), 1.0);
    return;
  }
  ivec2 at = ivec2(px.x, u_height - 1 - px.y);
  vec3 src = texelFetch(u_src, at, 0).rgb;
  vec3 graded = clamp(applyStack(src), 0.0, 1.0);
  vec2 st = srcStats(src);
  float y = st.x, sat = st.y;
  float lw = 1.0, cw = 1.0;
  if (u_skin_only == 0) {
    float room = texelFetch(u_room, at, 0).r;
    float crush = ss((sat - 0.28) / 0.20) * ss((0.48 - y) / 0.18);
    float pop = ss((room - 0.40) / 0.32);
    lw = clamp(0.18 + 0.82 * room * (1.0 - 0.55 * crush) * (1.0 + 0.20 * pop), 0.06, 1.0);
    cw = clamp(0.16 + 0.84 * room * (1.0 + 0.24 * pop), 0.06, 1.0);
  }
  vec3 hs = rgb2hsv(src);
  vec3 d = graded - src;
  float coolPull = ss(((0.5 * (d.g + d.b) - d.r) - 0.006) / 0.024);
  float skin = ss(1.0 - hdist(hs.x, 24.0) / 40.0) * ss((hs.y - 0.08) / 0.10) * ss((0.58 - hs.y) / 0.22)
    * ss((y - 0.16) / 0.12) * ss((0.84 - y) / 0.14);
  float comp, analog = 0.0;
  if (u_look.y < 0.014) {
    comp = clamp(0.85 * skin * coolPull, 0.0, 1.0);
  } else {
    float c1 = ss((hdist(hs.x, u_look.x) - 78.0) / 52.0) * ss((hs.y - 0.07) / 0.11);
    float coolLook = ss(1.0 - hdist(u_look.x, 155.0) / 75.0);
    comp = clamp(max(0.65 * c1, 0.70 * skin * max(coolLook, coolPull)), 0.0, 1.0);
    analog = ss((50.0 - hdist(hs.x, u_look.x)) / 34.0) * ss((hs.y - 0.10) / 0.12);
    float sky = ss((y - 0.42) / 0.22) * ss(1.0 - sat / 0.30);
    analog = clamp(analog * (1.0 - 0.92 * sky), 0.0, 1.0);
  }
  lw = clamp(lw * (1.0 - 0.08 * comp) * (1.0 - 0.10 * analog), 0.0, 1.0);
  cw = clamp(cw * (1.0 - 0.40 * comp) * (1.0 - 0.36 * analog), 0.0, 1.0);
  float gy = luma(graded);
  vec3 outc = vec3(y + (gy - y) * lw);
  outc = outc + (src - y) + ((graded - gy) - (src - y)) * cw;
  outc = clamp(outc, 0.0, 1.0);
  if (min(lw, cw) < 0.08) outc = src;
  o = vec4(outc, 1.0);
}`;

function lutLiveHsv(r, g, b) {
  const mx = Math.max(r, g, b);
  const df = mx - Math.min(r, g, b);
  let h = 0;
  if (df > 1e-8) {
    if (mx === r) h = (((g - b) / df) % 6 + 6) % 6;
    else if (mx === g) h = (b - r) / df + 2;
    else h = (r - g) / df + 4;
  }
  return [h * 60, mx > 1e-8 ? df / mx : 0];
}

class LutLiveRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
    this.ok = Boolean(this.gl) && this._init();
    this.key = "";
    this.ready = false;
    this.width = 0;
    this.height = 0;
  }

  _init() {
    const gl = this.gl;
    const compile = (type, text) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, text);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.warn("LUT live shader:", gl.getShaderInfoLog(sh));
        return null;
      }
      return sh;
    };
    const vs = compile(gl.VERTEX_SHADER, LUT_LIVE_VS);
    const fs = compile(gl.FRAGMENT_SHADER, LUT_LIVE_FS);
    if (!vs || !fs) return false;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.warn("LUT live link:", gl.getProgramInfoLog(prog));
      return false;
    }
    this.prog = prog;
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.loc = {};
    const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i += 1) {
      const name = gl.getActiveUniform(prog, i).name.replace(/\[0\]$/, "");
      this.loc[name] = gl.getUniformLocation(prog, name);
    }
    this.srcTex = gl.createTexture();
    this.roomTex = gl.createTexture();
    gl.uniform1i(this.loc.u_src, 0);
    gl.uniform1i(this.loc.u_room, 1);
    this.probeTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.probeTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 3, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    this.probeFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.probeFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.probeTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return true;
  }

  _upload(tex, unit, img) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  async prepare(key, srcUrl, roomUrl) {
    if (!this.ok) return false;
    this.key = key;
    this.ready = false;
    const load = (url) =>
      new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
      });
    const [src, room] = await Promise.all([load(srcUrl), load(roomUrl)]);
    if (this.key !== key) return false;
    this._upload(this.srcTex, 0, src);
    this._upload(this.roomTex, 1, room);
    this.width = src.naturalWidth;
    this.height = src.naturalHeight;
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.ready = true;
    return true;
  }

  _setLayers(spec) {
    const gl = this.gl;
    const L = LUT_LIVE_MAX_LAYERS;
    const kind = new Int32Array(L);
    const strength = new Float32Array(L);
    const tone = new Float32Array(L * 4);
    const sh = new Float32Array(L * 3);
    const hi = new Float32Array(L * 3);
    const mid = new Float32Array(L * 3);
    const gray = new Float32Array(L);
    const wash = new Int32Array(L);
    const mstart = new Int32Array(L * 3);
    const mcount = new Int32Array(L * 3);
    const moves = new Float32Array(LUT_LIVE_MAX_MOVES * 3);
    const tt = new Float32Array(L * 4);
    const tblack = new Float32Array(L);
    const tmid = new Float32Array(L * 3);
    const bands = new Float32Array(L * 6);
    let nm = 0;
    spec.layers.forEach((layer, i) => {
      strength[i] = layer.strength;
      if (layer.kind === "tune") {
        const t = layer.tune;
        kind[i] = 2;
        tt.set([t.hue, t.sat, t.gamma, t.contrast], i * 4);
        tblack[i] = t.black || 0;
        tmid.set((t.mid_tint || [0, 0, 0]).slice(0, 3), i * 3);
        bands.set(layer.bands, i * 6);
        return;
      }
      const p = layer.params;
      kind[i] = 1;
      tone.set([p.black, p.gamma, p.contrast, p.sparkle], i * 4);
      sh.set(p.sh_tint.slice(0, 3), i * 3);
      hi.set(p.hi_tint.slice(0, 3), i * 3);
      mid.set(p.mid_tint.slice(0, 3), i * 3);
      gray[i] = p.gray_protect || 0;
      wash[i] = layer.protectWash ? 1 : 0;
      [p.hue_moves, p.sat_moves, p.val_moves].forEach((list, k) => {
        mstart[i * 3 + k] = nm;
        mcount[i * 3 + k] = list.length;
        list.forEach((m) => {
          moves.set([m[0], m[1], m[2]], nm * 3);
          nm += 1;
        });
      });
    });
    const u = this.loc;
    gl.uniform1i(u.u_n, spec.layers.length);
    gl.uniform1iv(u.u_kind, kind);
    gl.uniform1fv(u.u_strength, strength);
    gl.uniform4fv(u.u_tone, tone);
    gl.uniform3fv(u.u_sh, sh);
    gl.uniform3fv(u.u_hi, hi);
    gl.uniform3fv(u.u_mid, mid);
    gl.uniform1fv(u.u_gray, gray);
    gl.uniform1iv(u.u_wash, wash);
    gl.uniform3iv(u.u_mstart, mstart);
    gl.uniform3iv(u.u_mcount, mcount);
    gl.uniform3fv(u.u_moves, moves);
    gl.uniform4fv(u.u_tt, tt);
    gl.uniform1fv(u.u_tblack, tblack);
    gl.uniform3fv(u.u_tmid, tmid);
    gl.uniform1fv(u.u_bands, bands);
  }

  _lookCast() {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.probeFbo);
    gl.viewport(0, 0, 3, 1);
    gl.uniform1i(this.loc.u_probe, 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    const px = new Uint8Array(12);
    gl.readPixels(0, 0, 3, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const cast = [0, 1, 2].map((c) => (((px[c] + px[c + 4] + px[c + 8]) / 3 / 255) - 0.5) / 4);
    const mean = (cast[0] + cast[1] + cast[2]) / 3;
    const probe = cast.map((c) => Math.min(1, Math.max(0, 0.5 + (c - mean) * 2.4)));
    return lutLiveHsv(probe[0], probe[1], probe[2]);
  }

  static supports(spec) {
    if (!spec || !spec.layers.length || spec.layers.length > LUT_LIVE_MAX_LAYERS) return false;
    let moves = 0;
    for (const layer of spec.layers) {
      if (layer.kind === "tune") continue;
      if (!layer.params) return false;
      moves += layer.params.hue_moves.length + layer.params.sat_moves.length + layer.params.val_moves.length;
    }
    return moves <= LUT_LIVE_MAX_MOVES;
  }

  render(spec) {
    if (!this.ok || !this.ready || !LutLiveRenderer.supports(spec)) return false;
    const gl = this.gl;
    gl.useProgram(this.prog);
    this._setLayers(spec);
    const look = this._lookCast();
    gl.uniform1i(this.loc.u_probe, 0);
    gl.uniform1i(this.loc.u_height, this.height);
    gl.uniform1i(this.loc.u_skin_only, spec.skinOnly ? 1 : 0);
    gl.uniform2f(this.loc.u_look, look[0], look[1]);
    gl.viewport(0, 0, this.width, this.height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    return true;
  }
}

window.LutLiveRenderer = LutLiveRenderer;
