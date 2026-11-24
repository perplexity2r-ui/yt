"""Procedural soundtrack: deep pad + pulse + whooshes + pops + UI ticks, 48 kHz stereo."""
import numpy as np, wave, sys, os, math

SR = 48000
DUR = 13.0
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
    (0.0, 3.2, [110.0, 164.81, 220.0, 329.63]),     # Am
    (3.1, 5.7, [87.31, 130.81, 174.61, 261.63]),    # F
    (5.6, 8.4, [110.0, 164.81, 220.0, 261.63]),     # Am (tension)
    (8.3, 10.9, [130.81, 196.0, 261.63, 392.0]),    # C
    (10.8, 13.2, [146.83, 220.0, 293.66, 440.0]),   # D (bright resolve)
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
t = 0.55
while t < 12.7:
    n = int(0.14 * SR)
    f = 220.0 if (int(t * 4) % 4) else 164.81
    s = (np.sin(2 * np.pi * f * np.arange(n) / SR) * 0.6 +
         bandpass_fft(noise(n), 1500, 5000, 1.0) * 0.25)
    s *= np.exp(-np.arange(n) / (n / 4.0))
    place(buf, s, t, gain=0.055, pan=0.25 if int(t * 4) % 2 else -0.25)
    t += 0.25

# --- kick on the beat from 1.5
for k in range(int(0.5 / 0.5), int(12.7 / 0.5) + 1):
    place(buf, kick(), k * 0.5 - 0.25, gain=0.16)
# --- hats
tt = 0.75
while tt < 12.8:
    place(buf, hat(), tt, gain=0.030, pan=0.3)
    tt += 0.25

# --- SFX for the character short
def bubble_pop(t0, f=520, gain=0.075):
    place(buf, pop(f, 0.13), t0, gain=gain, pan=0.0)

def keyboard(t0, dur=0.7, gain=0.045):
    tt = t0
    while tt < t0 + dur:
        n = int(0.03 * SR)
        s = bandpass_fft(noise(n), 2000, 8000, 1.0) * np.exp(-np.arange(n) / (n / 2.5))
        place(buf, s, tt, gain=gain, pan=(-0.3 if rng.random() < 0.5 else 0.3))
        tt += 0.055 + rng.random() * 0.05

place(buf, riser(1.2, 120, 800), 0.0, gain=0.12)
place(buf, whoosh(0.8, 250, 6000, up=True), 0.02, gain=0.09)
keyboard(0.45, 0.9, 0.05)
# bubbles
for tb, f in [(0.15, 560), (1.35, 620), (3.90, 480), (5.95, 700), (8.55, 860),
              (9.10, 640), (11.05, 700), (11.60, 560)]:
    bubble_pop(tb + 0.06, f, 0.085)
# checklist ticks
for i in range(5):
    tt = 3.55 + i * 0.25
    place(buf, click(2600 + 180 * i, 0.06), tt, gain=0.05, pan=0.2)
# tension: error riser + alarm
place(buf, riser(1.6, 140, 900), 6.1, gain=0.10)
for i in range(5):
    tt = 6.35 + i * 0.30
    n = int(0.14 * SR)
    s = (np.sin(2 * np.pi * 880 * np.arange(n) / SR) * 0.5 +
         np.sin(2 * np.pi * 660 * np.arange(n) / SR) * 0.4) * np.exp(-np.arange(n) / (n / 3))
    place(buf, s, tt, gain=0.045, pan=-0.2 + 0.1 * i)
n = int(0.8 * SR)
boom = np.sin(2 * np.pi * 60 * np.arange(n) / SR) * np.exp(-np.arange(n) / (n / 3.0))
place(buf, boom, 6.3, gain=0.14)
# success chime (C major arp)
for i, f in enumerate([523.25, 659.25, 783.99, 1046.5]):
    n = int(0.55 * SR)
    s = np.sin(2 * np.pi * f * np.arange(n) / SR) * np.exp(-np.arange(n) / (n / 4.0))
    s += np.sin(2 * np.pi * f * 2 * np.arange(n) / SR) * 0.25 * np.exp(-np.arange(n) / (n / 6.0))
    place(buf, s, 8.58 + i * 0.075, gain=0.075, pan=-0.3 + 0.2 * i)
# smug ending: vinyl-ish ding
n = int(1.2 * SR)
s = np.sin(2 * np.pi * 440 * np.arange(n) / SR) * np.exp(-np.arange(n) / (n / 3.0))
place(buf, s, 11.65, gain=0.07, pan=0.15)
place(buf, pop(380, 0.2), 11.62, gain=0.06)
# soft whooshes between scenes
for tc in [3.2, 5.5, 8.3, 10.9]:
    place(buf, whoosh(0.55, 300, 5000, up=True), tc, gain=0.075)

# ------------------------------------------------------------------ mixdown
mix = buf.copy()
mix[:, 0] = reverb(mix[:, 0], 1.4, 0.22)
mix[:, 1] = reverb(mix[:, 1], 1.4, 0.22)
# gentle top end + de-harsh
mix[:, 0] = lowpass_fft(mix[:, 0], 15000, 1.0)
mix[:, 1] = lowpass_fft(mix[:, 1], 15000, 1.0)
# fade in/out
fade = np.ones(N)
ni = int(0.06 * SR); no = int(0.5 * SR)
fade[:ni] = np.linspace(0, 1, ni) ** 1.5
fade[-no:] = np.linspace(1, 0, no) ** 1.2
mix *= fade[:, None]
# soft clip & normalize
mix = np.tanh(mix * 1.6) * 0.92
mix /= (np.abs(mix).max() + 1e-9)
mix *= 0.89

out = "/home/user/film2/out/soundtrack.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("wrote", out, os.path.getsize(out) // 1024, "KB")
