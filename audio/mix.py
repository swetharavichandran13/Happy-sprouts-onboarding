"""Builds the soundtrack: original soft music bed (no sound effects) + voiceover clips, mixed with independent levels.
Inputs: vo_*.wav (gen_vo.py), timeline from index.html (video seconds). Output: final.wav (48k stereo-free mono)."""
import numpy as np, soundfile as sf, json
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly
SR = 24000
VS = json.load(open('timeline.json'))          # {'VS':[...], 'S0':[...], 'total':...}
T = VS['total']; N = int(T * SR)
S0v = VS['S0v']                                  # video-time start of each of the 27 screen visits
vo_dur = json.load(open('vo_durations.json'))

# ---------------- voiceover placement (video seconds)
place = [('i0', .8), ('i1', S0v[1]), ('i2', S0v[5] + .1), ('i3', S0v[8] - .9)]
place += [(f'w{k}', S0v[9 + k - 1] + .35) for k in range(1, 8)]      # walkthrough: screens 1-7 only; none on 8, 9 or in loop 3
vo = np.zeros(N)
for name, t0 in place:
    x, sr = sf.read(f'vo_{name}.wav'); assert sr == SR
    i = int(t0 * SR); assert i + len(x) <= N, name
    # tiny fades on every clip so cuts never click
    f = int(.02 * SR); x = x.copy(); x[:f] *= np.linspace(0, 1, f); x[-f:] *= np.linspace(1, 0, f)
    vo[i:i + len(x)] += x
    print(f'{name}: {t0:.2f}-{t0 + len(x) / SR:.2f}s')
CR = 100                                                       # control-rate grid (Hz) for the level envelopes
nc = N // CR + 2; t = np.arange(N) / SR; tc = np.arange(nc) / CR
blk = np.abs(np.pad(vo, (0, nc * (SR // CR) - N))).reshape(nc, SR // CR).max(1)
voice_c = np.clip(gaussian_filter1d((blk > .004).astype(float), .25 * CR) * 3, 0, 1)   # ~1 while speaking
voice_on = np.interp(t, tc, voice_c)

# ---------------- music bed (original; soft marimba-like plucks over a warm pad; no chimes / effects)
rng = np.random.default_rng(7); mu = np.zeros(N)
def f(m): return 440 * 2 ** ((m - 69) / 12)
def add(t, sig, g=1.):
    i = int(t * SR)
    if i >= N: return
    j = min(N, i + len(sig)); mu[i:j] += sig[:j - i] * g
def pluck(fr, dur=1.4, br=.5):
    t = np.arange(int(SR * dur)) / SR; env = np.exp(-t * 3.2) * (1 - np.exp(-t * 400))
    return sum(np.sin(2 * np.pi * fr * k * t) * (br ** (k - 1)) / k for k in (1, 2, 3, 4)) * env
def pad(frs, dur):
    t = np.arange(int(SR * dur)) / SR; env = np.minimum(1, t / .9) * np.minimum(1, (dur - t) / .9)
    return sum(np.sin(2 * np.pi * fr * t) + .5 * np.sin(2 * np.pi * fr * 1.003 * t) for fr in frs) * env / len(frs)
bpm = 104; beat = 60 / bpm; bar = 4 * beat
prog = [(48, [60, 64, 67]), (45, [57, 60, 64]), (41, [57, 60, 65]), (43, [59, 62, 67])]   # C Am F G
mel = [[0, 4, 7, 4, 9, 7, 4, 2], [4, 7, 9, 7, 12, 9, 7, 4], [0, 2, 4, 7, 4, 2, 0, -3], [2, 4, 7, 9, 7, 4, 2, 0]]
for b in range(int(T / bar) + 2):
    t0 = b * bar; root, ch = prog[b % 4]
    add(t0, pad([f(m) for m in ch], bar + .5), .06)
    for k in range(8): add(t0 + k * beat / 2, pluck(f(ch[k % 3] + 12), .7, .35), .09)
    add(t0, pluck(f(root), 1.6, .25), .20); add(t0 + 2 * beat, pluck(f(root + 7), 1.2, .25), .12)
    if b >= 2:
        for k, d in enumerate(mel[b % 4]):
            if (b % 8 >= 4) or k % 2 == 0: add(t0 + k * beat / 2, pluck(f(72 + d), 1.0, .55), .08)
rv = np.zeros(N)
for d, g in ((.047, .35), (.071, .28), (.113, .2), (.167, .14)):
    k = int(d * SR); rv[k:] += mu[:-k] * g
mu = mu + rv * .8
mu = mu / np.sqrt(np.mean(mu ** 2))                    # unit RMS, levels are set below in dB

# ---------------- independent levels (dBFS RMS of the music bed)
VS_ = VS['VS']                                         # video-time start of loops 1,2,3 (+ end)
lc = np.full(nc, -36.)                                 # between voice lines / screens 08-09: soft
lc = np.where(voice_c > .5, -43., lc)                  # under the voice: well behind it
lc = np.where(tc >= VS_[2] + .5, -35., lc)             # loop 3: soft but audible
lc = gaussian_filter1d(lc, 1.1 * CR)                   # long, smooth ramps = no audible jumps
lvl = np.interp(t, tc, lc)
gain = 10 ** (lvl / 20)
fade_in = np.minimum(1, t / 1.5); fade_out = np.minimum(1, np.maximum(0, (T - t)) / 3.0)
mu = mu * gain * fade_in * fade_out

# ---------------- voice level and final mix
vo = vo / np.abs(vo).max() * .85                       # voice stays loud on its own
mix = vo + mu
mix = np.tanh(mix * 1.0) if np.abs(mix).max() > .98 else mix
sf.write('final_raw.wav', mix, SR)
rms = lambda a: 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9)
sp = voice_on > .5
print('voice RMS while speaking %.1f dB; music RMS under voice %.1f dB; music loop 3 %.1f dB' % (rms(vo[sp]), rms(mu[sp]), rms(mu[t >= VS_[2] + 2])))
