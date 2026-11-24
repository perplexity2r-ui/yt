"""Procedural soundtrack: deep pad + pulse + whooshes + pops + UI ticks, 48 kHz stereo."""
import numpy as np, wave, sys, os, math

SR = 48000
DUR = 10.0
N = int(SR * DUR)
rng = np.random.default_rng(11)

def env(n, a=0.01, r=0.2, sr=SR, curve=2.0):
    e = np.ones(n, np.float64)
    na = max(1, int(a * sr)); nr = max(1, int(r * sr))
    na = min(na, n); nr = min(nr, n)
    e[:na] = np.linspace(0, 1, na) ** curve
    e[-nr:] *= np.linspace(1, 0, nr) ** curve
    return e

def place(buf, sig, t0, gain=1.0, pan=0.0):
    i0 = int(t0 * SR)
    if i0 >= len(buf): return
    i1 = min(len(buf), i0 + len(sig))
    seg = sig[:i1 - i0] * gain
    l = math.sqrt((1 - pan) / 2) * 1.414
    r = math.sqrt((1 + pan) / 2) * 1.414
    buf[i0:i1, 0] += seg * l
    buf[i0:i1, 1] += seg * r

def sine(f, n, phase=0.0):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t + phase)

def saw(f, n, detune=0.0, phase=0.0):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k in range(1, 12):
        out += np.sin(2 * np.pi * f * (1 + detune) * k * t + phase) / k
    return out * 0.55

def noise(n):
    return rng.normal(0, 1, n)

def lowpass_fft(x, cutoff, slope=2.0):
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= 1.0 / (1.0 + (f / max(cutoff, 20)) ** (2 * slope))
    return np.fft.irfft(X, n)

def bandpass_fft(x, lo, hi, slope=2.0):
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= 1.0 / (1.0 + (lo / np.maximum(f, 1)) ** (2 * slope))
    X *= 1.0 / (1.0 + (f / hi) ** (2 * slope))
    return np.fft.irfft(X, n)

def reverb(x, decay=1.6, mix=0.35):
    """cheap convolution reverb with a decaying noise impulse"""
    n = int(decay * SR)
    ir = noise(n) * np.exp(-np.arange(n) / (n / 5.0))
    ir = lowpass_fft(ir, 5000, 1.0)
    ir[:200] = 0
    wet = np.convolve(x, ir)[:len(x)]
    wet /= (np.abs(wet).max() + 1e-9)
    return x * (1 - mix) + wet * mix * (np.abs(x).max() + 1e-9)

def kick(dur=0.42, f0=110, f1=42):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t * 26)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return s * np.exp(-t * 7.0)

def hat(dur=0.06):
    n = int(dur * SR)
    s = bandpass_fft(noise(n), 6000, 15000, 1.0)
    return s * np.exp(-np.arange(n) / (n / 3.5))

def click(f=2500, dur=0.05, tone=0.35):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * f * t) * 0.6 + noise(n) * 0.2
    return bandpass_fft(s, 1200, 9000, 1.0) * np.exp(-t / (dur * 0.28))

def pop(f=420, dur=0.16):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f_t = f * (1 + 0.6 * np.exp(-t * 40))
    s = np.sin(2 * np.pi * np.cumsum(f_t) / SR)
    return s * np.exp(-t * 18)

def whoosh(dur=0.7, lo=200, hi=5000, up=True):
    n = int(dur * SR)
    s = noise(n)
    # sweeping bandpass via time-varying amplitude of two fixed bands
    a = bandpass_fft(s, lo, hi * 0.4, 1.0)
    b = bandpass_fft(s, lo * 2, hi, 1.0)
    t = np.linspace(0, 1, n)
    mix = t if up else 1 - t
    out = a * (1 - mix) + b * mix
    return out * np.sin(np.pi * t) ** 1.6

def riser(dur=1.2, f0=180, f1=1400):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (f1 / f0) ** (t / dur)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.5
    s += bandpass_fft(noise(n), 800, 9000, 1.0) * 0.35
    return s * (t / dur) ** 2

# ------------------------------------------------------------------ arrangement
buf = np.zeros((N, 2))

# --- pad chords: Am - F - C - G (root motion) 1 chord per ~2.5 s
chords = [
    (0.0, 2.6, [110.0, 164.81, 220.0, 329.63]),     # Am
    (2.5, 5.0, [87.31, 130.81, 174.61, 261.63]),    # F
    (4.9, 7.4, [130.81, 196.0, 261.63, 392.0]),     # C
    (7.3, 10.0, [98.0, 146.83, 196.0, 293.66]),     # G
]
for (t0, t1, notes) in chords:
    dur = t1 - t0 + 0.6
    n = int(dur * SR)
    pad = np.zeros(n)
    for f in notes:
        pad += saw(f, n, detune=0.0025) * 0.5 + saw(f, n, detune=-0.003) * 0.5
    pad = lowpass_fft(pad, 900, 1.2)
    pad *= env(n, a=0.9, r=1.1, curve=1.5)
    pad /= (np.abs(pad).max() + 1e-9)
    place(buf, pad, t0, gain=0.155, pan=-0.05)
    # sub bass
    sub = np.sin(2 * np.pi * notes[0] / 2 * np.arange(n) / SR) * env(n, 0.5, 1.0)
    place(buf, sub, t0, gain=0.13)

# --- pulse: 8th-note muted pluck from 1.3 s
t = 1.30
while t < 9.6:
    n = int(0.14 * SR)
    f = 220.0 if (int(t * 4) % 4) else 164.81
    s = (np.sin(2 * np.pi * f * np.arange(n) / SR) * 0.6 +
         bandpass_fft(noise(n), 1500, 5000, 1.0) * 0.25)
    s *= np.exp(-np.arange(n) / (n / 4.0))
    place(buf, s, t, gain=0.055, pan=0.25 if int(t * 4) % 2 else -0.25)
    t += 0.25

# --- kick on the beat from 1.5
for k in range(int(1.5 / 0.5), int(9.6 / 0.5) + 1):
    place(buf, kick(), k * 0.5, gain=0.20)
# --- hats
tt = 1.75
while tt < 9.7:
    place(buf, hat(), tt, gain=0.030, pan=0.3)
    tt += 0.25

# --- SFX
# intro swell
place(buf, riser(1.5, 120, 900), 0.0, gain=0.16)
place(buf, whoosh(0.9, 250, 6000, up=True), 0.05, gain=0.10)
# hero bloom hit
n = int(0.9 * SR)
boom = np.sin(2 * np.pi * 58 * np.arange(n) / SR) * np.exp(-np.arange(n) / (n / 3.5))
place(buf, boom, 0.42, gain=0.22)
place(buf, bandpass_fft(noise(n), 300, 3000, 1.0) * np.exp(-np.arange(n) / (n / 6)), 0.42, gain=0.05)
# machine pops (7)
for i in range(7):
    place(buf, pop(360 + 40 * i, 0.18), 1.26 + i * 0.078 + 0.05, gain=0.10, pan=-0.5 + i * 0.16)
# scene transitions: whooshes
for (tc, upd) in [(2.7, False), (4.15, True), (5.6, True), (7.4, False), (9.0, True)]:
    place(buf, whoosh(0.75, 200, 7000, up=upd), tc, gain=0.13)
    place(buf, click(1800, 0.09), tc + 0.02, gain=0.07, pan=-0.1)
# haiku: flying cube + confirm
place(buf, whoosh(0.45, 600, 9000, up=True), 3.10, gain=0.05, pan=-0.4)
place(buf, click(3200, 0.10), 3.55, gain=0.07, pan=0.2)
place(buf, click(4400, 0.08), 3.66, gain=0.05, pan=0.3)
# sonnet cards (3 pops)
for i, tc in enumerate([4.34, 4.56, 4.78]):
    place(buf, pop(300 + 60 * i, 0.2), tc, gain=0.075, pan=-0.2 + 0.2 * i)
# opus coins (5) + counter ticks
for i in range(5):
    tc = 6.28 + i * 0.18
    place(buf, click(1400 + 120 * i, 0.12, 0.5), tc, gain=0.08, pan=-0.3 + 0.1 * i)
    place(buf, pop(520 + 30 * i, 0.12), tc + 0.28, gain=0.045, pan=0.1)
# conclusion: two strikes
n = int(1.1 * SR)
for t0, f in [(8.02, 62), (8.36, 82)]:
    s = np.sin(2 * np.pi * f * np.arange(n) / SR) * np.exp(-np.arange(n) / (n / 3.0))
    place(buf, s, t0, gain=0.16)
    place(buf, bandpass_fft(noise(n), 400, 6000, 1.0) * np.exp(-np.arange(n) / (n / 8)), t0, gain=0.05)
# CTA sparkle
for i in range(6):
    place(buf, click(2200 + 500 * i, 0.10), 9.2 + i * 0.07, gain=0.045, pan=-0.5 + 0.2 * i)

# ------------------------------------------------------------------ mixdown
mix = buf.copy()
mix[:, 0] = reverb(mix[:, 0], 1.4, 0.22)
mix[:, 1] = reverb(mix[:, 1], 1.4, 0.22)
# gentle top end + de-harsh
mix[:, 0] = lowpass_fft(mix[:, 0], 15000, 1.0)
mix[:, 1] = lowpass_fft(mix[:, 1], 15000, 1.0)
# fade in/out
fade = np.ones(N)
ni = int(0.05 * SR); no = int(0.35 * SR)
fade[:ni] = np.linspace(0, 1, ni) ** 1.5
fade[-no:] = np.linspace(1, 0, no) ** 1.2
mix *= fade[:, None]
# soft clip & normalize
mix = np.tanh(mix * 1.6) * 0.92
mix /= (np.abs(mix).max() + 1e-9)
mix *= 0.89

out = "/home/user/motion/out/soundtrack.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("wrote", out, os.path.getsize(out) // 1024, "KB")
