"""Happy Sprouts wordmark: monoline, round-capped, bouncy letters (supersampled), RGBA."""
import cv2, numpy as np

CORAL = (255, 90, 104); BLUE = (80, 150, 246); YELLOW = (254, 197, 40)
PIST = (140, 194, 100); APRI = (255, 160, 108)

def arc(cx, cy, r, a0, a1, n=40):
    t = np.radians(np.linspace(a0, a1, n)); return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)
def line(*p): return np.array(p, float)
def cat(*s): return np.concatenate(s)
def spline(pts, n=14):
    p = np.array(pts, float); P = np.vstack([p[0], p, p[-1]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2]); return np.array(out)

S_PATH = [(.9, -1.28), (.66, -1.56), (.3, -1.6), (.06, -1.34), (.16, -1.02), (.5, -.84), (.84, -.66), (.96, -.32), (.74, -.03), (.36, -.02), (.08, -.16)]
SS = np.array(S_PATH)
LET = {
    'H': ([line((0, -1.6), (0, 0)), line((.95, -1.6), (.95, 0)), line((0, -.8), (.95, -.8))], 1.0),
    'a': ([arc(.5, -.5, .5, 0, 360, 60), line((1, -1), (1, 0))], 1.05),
    'p': ([arc(.52, -.5, .5, 0, 360, 60), line((0, -1), (0, .7))], 1.05),
    'y': ([spline([(0, -1), (0, -.45), (.5, 0), (1, -.4), (1, -1)]), spline([(1, -1), (1, .25), (.7, .68), (.25, .66), (0, .45)])], 1.05),
    'S': ([spline(S_PATH)], 1.0),
    'r': ([line((0, -1), (0, 0)), spline([(0, -.5), (.15, -.9), (.5, -1.0), (.72, -.95)])], .8),
    'o': ([arc(.5, -.5, .5, 0, 360, 60)], 1.0),
    'u': ([spline([(0, -1), (0, -.45), (.5, 0), (1, -.45)]), line((1, -1), (1, 0))], 1.05),
    't': ([spline([(.32, -1.55), (.32, -.3), (.55, 0), (.85, -.05)]), line((-.08, -1.02), (.78, -1.02))], .85),
    's': ([spline([(.85, -.84), (.62, -1.0), (.28, -1.0), (.08, -.78), (.2, -.55), (.5, -.46), (.82, -.3), (.9, -.12), (.62, 0), (.28, 0), (.04, -.14)])], .95),
}
BOUNCE = {  # (dy, rot deg)
    'Happy': [(0, -3), (.05, 4), (-.04, -4), (.05, 3), (-.02, -3)],
    'Sprouts': [(0, 3), (-.04, -4), (.05, 3), (-.03, -3), (.04, 4), (0, -3), (.04, 3)],
}
COLS = {'Happy': [CORAL, BLUE, YELLOW, PIST, CORAL], 'Sprouts': [BLUE, CORAL, YELLOW, PIST, BLUE, CORAL, YELLOW]}

def draw_word(word, unit, sw, gap, img, ox, oy):
    """draw into img (BGRA float canvas at supersample) with origin ox,oy = left of baseline"""
    x = ox; cols = COLS[word]; bo = BOUNCE[word]
    for i, ch in enumerate(word):
        strokes, adv = LET[ch]; dy, rot = bo[i]
        col = cols[i][::-1]
        cx, cy = adv * unit / 2, -.5 * unit
        c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
        layer = np.zeros(img.shape[:2], np.uint8)
        for st in strokes:
            P = st * unit; P = P - [cx, cy]; P = np.stack([P[:, 0] * c - P[:, 1] * s, P[:, 0] * s + P[:, 1] * c], 1) + [cx, cy]
            P = P + [x, oy + dy * unit]
            cv2.polylines(layer, [np.round(P * 1).astype(np.int32).reshape(-1, 1, 2)], False, 255, int(sw * unit), cv2.LINE_AA)
            for q in (P[0], P[-1]): cv2.circle(layer, tuple(np.round(q).astype(int)), int(sw * unit / 2), 255, -1, cv2.LINE_AA)
        # subtle white inner highlight for gentle depth
        a = layer.astype(np.float32) / 255
        for k in range(3): img[..., k] = img[..., k] * (1 - a) + col[k] * a
        img[..., 3] = np.maximum(img[..., 3], a * 255)
        x += (adv + gap) * unit
    return x - gap * unit


def sprout(img, cx, cy, size):
    col = PIST[::-1]; layer = np.zeros(img.shape[:2], np.uint8)
    cv2.line(layer, (int(cx), int(cy)), (int(cx), int(cy - size * .55)), 255, int(size * .17), cv2.LINE_AA)
    for sgn in (-1, 1):
        ell = np.zeros_like(layer)
        cv2.ellipse(ell, (int(cx + sgn * size * .32), int(cy - size * .72)), (int(size * .36), int(size * .17)), sgn * -28, 0, 360, 255, -1, cv2.LINE_AA)
        layer = np.maximum(layer, ell)
    a = layer / 255.
    for k in range(3): img[..., k] = img[..., k] * (1 - a) + col[k] * a
    img[..., 3] = np.maximum(img[..., 3], a * 255)


def wordmark(width_px=560, ss=3):
    """returns BGRA uint8 image of the wordmark, about width_px wide"""
    unit = 150 * ss; sw = .38; gap = .3
    W, H = int(11 * unit), int(7.2 * unit)
    out = []
    img = np.zeros((H, W, 4), np.float32)
    x1 = draw_word('Happy', unit, sw, gap, img, int(.6 * unit), int(2.95 * unit))
    # centre line 2 under line 1
    tmp = np.zeros((H, W, 4), np.float32)
    x2 = draw_word('Sprouts', unit, sw, gap, tmp, int(.6 * unit), int(5.55 * unit))
    # re-centre each line horizontally
    w1, w2 = x1 - .6 * unit, x2 - .6 * unit; wmax = max(w1, w2)
    M = np.float32([[1, 0, (wmax - w1) / 2], [0, 1, 0]]); img = cv2.warpAffine(img, M, (W, H))
    M = np.float32([[1, 0, (wmax - w2) / 2], [0, 1, 0]]); tmp = cv2.warpAffine(tmp, M, (W, H))
    a1, a2 = img[..., 3:] / 255, tmp[..., 3:] / 255
    comp = img * (1 - a2) + tmp * a2; comp[..., 3:] = np.maximum(img[..., 3:], tmp[..., 3:])
    # sprout above the 'a' (second letter of line 1) like the original dotted-i
    off = (wmax - w1) / 2
    ax = .6 * unit + off + (LET["H"][1] + .3) * unit + .52 * unit
    sprout(comp, ax, 2.95 * unit - 1.12 * unit, .9 * unit)
    ys, xs = np.nonzero(comp[..., 3] > 8); pad = int(.1 * unit)
    comp = comp[ys.min() - pad:ys.max() + pad, xs.min() - pad:xs.max() + pad]
    sc = width_px / comp.shape[1]
    comp = cv2.resize(comp, (width_px, int(comp.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    return np.clip(comp, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    w = wordmark(560); bg = np.full(w.shape[:3][:2] + (3,), (240, 250, 254), np.uint8)
    a = w[..., 3:] / 255.; cv2.imwrite('/home/claude/hs_work/wm_test.png', (bg * (1 - a) + w[..., :3] * a).astype(np.uint8)); print(w.shape)
