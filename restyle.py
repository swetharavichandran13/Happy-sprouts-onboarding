"""Happy Sprouts minimal restyle.
Reads the ORIGINAL Little Joys exports (ORIG dir) and writes edited exports to src/ which prep_all.py then slices into the
animation assets. Nothing here touches the original repo.
Edits: green characters -> pistachio, green UI accents -> watermelon coral, subtle faces (brows + catchlights),
Happy Sprouts wordmark (Screen 1), new tagline (Screen 8), fictional sachet packaging (Screen 7)."""
import cv2, numpy as np, os, sys
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
import hs_logo

ORIG = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/'
OUT = 'src/'; os.makedirs(OUT, exist_ok=True)
CORAL = np.array([255, 90, 104], np.float32)       # watermelon coral (RGB)
CREAM = np.array([254, 251, 242], np.float32)
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'pjs700.ttf')

def rd(n): return cv2.imread(ORIG + n)
def wr(n, im): cv2.imwrite(OUT + n, im, [cv2.IMWRITE_JPEG_QUALITY, 97])

# ------------------------------------------------------------------ green -> pistachio (characters)
def green_mask(im, smin=.18):
    h = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); H, S = h[..., 0], h[..., 1] / 255.
    return (H >= 38) & (H <= 95) & (S > smin)

def char_green(im):
    """green connected blobs that touch the image border = characters"""
    m = green_mask(im, .35); m = ndi.binary_closing(m, iterations=2)
    L, n = ndi.label(m); keep = np.zeros_like(m); H, W = m.shape
    for k, sl in enumerate(ndi.find_objects(L), 1):
        if sl is None: continue
        ys, xs = sl; area = (L[sl] == k).sum()
        near = ys.start < 40 or xs.start < 40 or ys.stop > H - 40 or xs.stop > W - 40
        if area > 20000 and near: keep |= (L == k)
    return ndi.binary_dilation(keep, iterations=5)

PIST_H, PIST_S, PIST_V = 47, .56, 1.02
def to_pistachio(im, mask):
    h = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    m = mask & green_mask(im, .08)
    H, S, V = h[..., 0], h[..., 1], h[..., 2]
    # pull hue toward a yellower green, soften saturation, lift value slightly, keep shading structure
    w = np.clip((S / 255. - .08) / .5, 0, 1)
    H2 = np.where(m, H * (1 - w) + PIST_H * w, H)
    S2 = np.where(m, S * (PIST_S * w + (1 - w)), S)
    V2 = np.where(m, np.clip(V * PIST_V + 16 * w, 0, 255), V)
    out = cv2.cvtColor(np.dstack([H2, S2, V2]).clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[..., None]
    return (out * a + im * (1 - a)).astype(np.uint8)

# ------------------------------------------------------------------ green -> coral (UI accents)
def ui_coral(im, region=None, exclude=None, ref_green=None):
    """unmix a green accent over its background and re-mix with coral (keeps anti-aliasing)."""
    f = im[..., ::-1].astype(np.float32)
    gm = green_mask(im, .22)
    if region is not None: gm &= region
    if exclude is not None: gm &= ~exclude
    gm = ndi.binary_dilation(gm, iterations=2)
    G = np.median(f[gm & green_mask(im, .6)], 0) if ref_green is None else ref_green
    white = f.min(2) > 238
    W = np.where(white[..., None], 255., CREAM)
    d = W - G; t = np.clip(((W - f) * d).sum(2) / (d * d).sum(2), 0, 1)
    mix = W * (1 - t[..., None]) + G * t[..., None]
    ok = (np.abs(mix - f).max(2) < 46) & (t > .04) & gm
    new = W * (1 - t[..., None]) + CORAL * t[..., None]
    out = np.where(ok[..., None], new, f)
    return out[..., ::-1].clip(0, 255).astype(np.uint8), G

# ------------------------------------------------------------------ faces
def find_eyes(img, ymin=0, ymax=None, minpix=1500, maxwhite=40000, thr=14):
    s = img.astype(int); mn = s.min(2)
    pup = (s.max(2) < thr); pup[:ymin] = 0
    if ymax: pup[ymax:] = 0
    L, n = ndi.label(pup); out = []
    for k in range(1, n + 1):
        m = L == k
        if m.sum() < minpix: continue
        d = ndi.binary_dilation(m, iterations=int(14 * (minpix / 1500) ** .5 + 4))
        wh = (mn > 205) & (np.abs(s[..., 0] - s[..., 2]) < 30)
        wl, wn = ndi.label(wh | m); ids = np.unique(wl[d & wh]); ids = ids[ids > 0]
        ids = [j for j in ids if (wl == j).sum() < maxwhite]
        em = ndi.binary_fill_holes(np.isin(wl, ids) | m)
        out.append((em, m))
    return out

def decorate(im, eyes, scale=1.0, tilt_hint=None):
    """subtle brows + pupil catchlights. eyes: list of (eye_mask, pupil_mask). Brows drawn above each eye relative to the
    face axis (pair-wise); catchlights are drawn on the pupil."""
    f = im.astype(np.float32); H, W = im.shape[:2]
    info = []
    for em, pm in eyes:
        ys, xs = np.nonzero(em); py, px = np.nonzero(pm)
        info.append(dict(c=np.array([xs.mean(), ys.mean()]), w=xs.max() - xs.min(), h=ys.max() - ys.min(), pc=np.array([px.mean(), py.mean()]), pr=np.sqrt(pm.sum() / np.pi), em=em, bb=(xs.min(), xs.max(), ys.min(), ys.max())))
    for i, e in enumerate(info):
        # face axis from nearest partner (within 2.6 eye widths)
        best = None
        for j, o in enumerate(info):
            if i == j: continue
            d = np.linalg.norm(o['c'] - e['c'])
            if d < 2.6 * max(e['w'], 40) and (best is None or d < best[0]): best = (d, o)
        if best:
            ax = best[1]['c'] - e['c']; ax /= np.linalg.norm(ax)
            if ax[0] < 0: ax = -ax
            up = np.array([ax[1], -ax[0]])      # perpendicular
            if up[1] > 0: up = -up
        else:
            ax = np.array([1., 0.]); up = np.array([0., -1.])
            if tilt_hint is not None: ax, up = tilt_hint
        r = max(e['w'], e['h']) / 2
        # body colour sampled from a ring above/around the eye (skin of the character)
        ring = ndi.binary_dilation(e['em'], iterations=int(r * .45)) & ~ndi.binary_dilation(e['em'], iterations=int(r * .12))
        rr = f[ring]; sat = rr.max(1) - rr.min(1); rr = rr[(sat > 40) & (rr.max(1) < 252)]
        if len(rr) < 20: continue
        body = np.median(rr, 0)
        # Brow centre sits above the eye; arch slightly raised on outer side for a cheerful look
        sgn = 1
        cen = e['c'] + up * r * 1.55
        layer = np.zeros((H, W), np.uint8)
        t = np.linspace(-1, 1, 30)
        pts = []
        for tt in t:
            along = tt * r * .78
            arch = (1 - tt * tt) * r * .22
            tilt = tt * r * .10 * (1 if np.dot(ax, [1, 0]) >= 0 else -1)
            p = cen + ax * along + up * (arch + tilt * (-1))
            pts.append(p)
        pts = np.array(pts)
        th = max(2, int(r * .20))
        cv2.polylines(layer, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, 255, th, cv2.LINE_AA)
        for q in (pts[0], pts[-1]): cv2.circle(layer, tuple(np.round(q).astype(int)), th // 2, 255, -1, cv2.LINE_AA)
        # only on character body pixels (never on cream / island / photo)
        near = np.abs(f - body).max(2) < 70
        a = layer.astype(np.float32) / 255 * near
        dark = body * .55
        a = a[..., None] * .85
        f = f * (1 - a) + dark * a
        # catchlight
        cl = np.zeros((H, W), np.uint8)
        cc = e['pc'] + np.array([.30, -.32]) * e['pr']
        cv2.circle(cl, tuple(np.round(cc).astype(int)), max(2, int(e['pr'] * .2)), 255, -1, cv2.LINE_AA)
        a2 = (cl.astype(np.float32) / 255 * (e['em'] | ndi.binary_dilation(e['em'], iterations=0)))[..., None]
        f = f * (1 - a2 * .95) + 255 * a2 * .95
    return f.clip(0, 255).astype(np.uint8)

def rowfill(img, y0, y1, xa=14, xb=1174):
    o = img.copy().astype(np.float32); t = np.linspace(0, 1, img.shape[1])[:, None]
    for y in range(y0, y1): o[y] = img[y, xa - 8:xa + 8].mean(0) * (1 - t) + img[y, xb - 8:xb + 8].mean(0) * t
    return o.astype(np.uint8)

def blend_rgba(im, rgba, x, y):
    h, w = rgba.shape[:2]; a = rgba[..., 3:4].astype(np.float32) / 255
    reg = im[y:y + h, x:x + w].astype(np.float32); im[y:y + h, x:x + w] = (reg * (1 - a) + rgba[..., :3] * a).astype(np.uint8)

# ================================================================== helpers for the pipeline
ISLAND = (412 / 1188., 776 / 1188., 26 / 1188., 146 / 1188.)   # dynamic island rect as fractions of width
def island_rect(shape, sc):
    return (int(412 * sc), int(776 * sc), int(26 * sc), int(146 * sc))
def drop_island_eyes(eyes, sc, tile_off=0, width=None):
    out = []
    for em, pm in eyes:
        ys, xs = np.nonzero(em); cx, cy = xs.mean(), ys.mean()
        if width:   # tiled carousel image: position inside its own slide
            k = int(cx // width); cx -= k * width
        x0, x1, y0, y1 = island_rect(None, sc)
        if x0 - 20 * sc <= cx <= x1 + 20 * sc and cy < y1 + 40 * sc: continue
        out.append((em, pm))
    return out

def save_pair(name, plain, deco): wr(name.replace('.jpg', '__plain.jpg'), plain); wr(name, deco)

# ================================================================== SCREEN 1
def screen1():
    im = rd('1__Splash.jpg'); eyes = find_eyes(im, 150, None, 1500)
    ch = char_green(im); im = to_pistachio(im, ch)
    ya, yb = 860, 1380; im2 = im.copy(); im2[ya:yb] = rowfill(im, ya, yb)[ya:yb]; im = im2
    wm = hs_logo.wordmark(620); h, w = wm.shape[:2]
    blend_rgba(im, wm, (1188 - w) // 2, 1130 - h // 2)
    save_pair('1__Splash.jpg', im, decorate(im, eyes)); return len(eyes)

def screen2(): wr('2__Loading.jpg', rd('2__Loading.jpg'))

def screen3():
    im = rd('3__Age.jpg'); eyes = find_eyes(im, 30, None, 450)
    ch = char_green(im); im = to_pistachio(im, ch)
    ui = np.zeros(im.shape[:2], bool); ui[300:2150] = 1
    im, G = ui_coral(im, region=ui, exclude=ch)
    wr('3__Age.jpg', decorate(im, eyes)); return len(eyes)

# ------------------------------------------------------------------ Screen 4 : coral outline on selected card
def ring_coral(im):
    f = im[..., ::-1].astype(np.float32)
    gm = green_mask(im, .35); L, n = ndi.label(ndi.binary_closing(gm, iterations=2)); ring = np.zeros_like(gm)
    for k, sl in enumerate(ndi.find_objects(L), 1):
        if sl is None: continue
        ys, xs = sl
        if (xs.stop - xs.start) > 0.8 * im.shape[1] * .6 and (ys.stop - ys.start) > 300: ring |= (L == k)
    ring = ndi.binary_dilation(ring, iterations=3)
    gm = green_mask(im, .2) & ring
    G = np.median(f[gm & green_mask(im, .5)], 0)
    sat = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1] / 255.
    t = np.clip((sat - .15) / (np.median(sat[gm & green_mask(im, .5)]) - .15), 0, 1) * gm
    lum = f.mean(2) / G.mean()
    new = CORAL[None, None] * np.clip(lum, .6, 1.12)[..., None]
    a = cv2.GaussianBlur(t.astype(np.float32), (0, 0), .7)[..., None]
    return (f * (1 - a) + new * a)[..., ::-1].clip(0, 255).astype(np.uint8), int(ring.sum())

def screen4():
    im = rd('4__Focus_scroll.jpg'); out, n = ring_coral(im); wr('4__Focus_scroll.jpg', out); return n

# ------------------------------------------------------------------ Screens 5 & 6 carousels
def carousel(n):
    im = rd(f'{n}__' + ('Insight' if n == 5 else 'Solution') + '_carousel.jpg'); H, W = im.shape[:2]; sc = (W / 3) / 1188
    eyes = find_eyes(im, 0, int(160 * sc), 600, 40000, thr=32); eyes = drop_island_eyes(eyes, sc, width=W // 3)
    ch = char_green(im); im = to_pistachio(im, ch)
    ui = np.zeros(im.shape[:2], bool); ui[int(150 * sc):845] = 1
    im, G = ui_coral(im, region=ui, exclude=ch)
    wr(f'{n}__carousel_plain.jpg', im); wr(f'{n}__carousel.jpg', decorate(im, eyes)); return len(eyes)
def screen5(): return carousel(5)
def screen6(): return carousel(6)

# ------------------------------------------------------------------ Screen 7 : sachet packaging + skip + face
def pouch(w, h, col, shade, seal, tilt=0, ss=3):
    """RGBA stand-up sachet, simple flat shapes with soft shading; wordmark label, no character."""
    W, H = int((w + 60) * ss), int((h + 60) * ss); P = lambda v: int(v * ss)
    ox = 30 * ss; oy = 30 * ss
    mask = np.zeros((H, W), np.uint8)
    pts = []
    for s_ in np.linspace(0, 1, 60): pts.append((ox + P(-8 * np.sin(np.pi * min(s_ * .95 + .02, 1)) + 0), oy + P(s_ * h)))
    left = [(ox + P(6) - P(10 * np.sin(np.pi * min(s_ * .9, 1))), oy + P(s_ * h)) for s_ in np.linspace(0, 1, 60)]
    right = [(ox + P(w - 6) + P(10 * np.sin(np.pi * min(s_ * .9, 1))), oy + P(s_ * h)) for s_ in np.linspace(0, 1, 60)]
    poly = np.array(left + right[::-1], np.int32); cv2.fillPoly(mask, [poly], 255, cv2.LINE_AA)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P(14), P(14))))
    # tear notches
    for sx in (ox + P(2), ox + P(w - 2)): cv2.circle(mask, (sx, oy + P(22)), P(5), 0, -1, cv2.LINE_AA)
    body = np.zeros((H, W, 3), np.float32); body[:] = col[::-1]
    xs = np.linspace(0, 1, W)[None, :, None]
    body *= (1 - .16 * (xs ** 2.2) - .10 * ((1 - xs) ** 4))                # side shading
    sealm = np.zeros((H, W), np.float32); sealm[oy:oy + P(34)] = 1
    body = body * (1 - sealm[..., None]) + np.array(seal[::-1], np.float32) * sealm[..., None]
    for x in range(ox, W - ox, P(7)): body[oy:oy + P(34), x:x + max(1, ss)] *= .90   # crimp ridges
    cv2.line(body, (ox, oy + P(34)), (W - ox, oy + P(34)), tuple(c * .8 for c in shade[::-1]), max(1, ss), cv2.LINE_AA)
    cv2.line(body, (ox + P(10), oy + P(50)), (W - ox - P(10), oy + P(50)), tuple(c * .85 for c in col[::-1]), P(2), cv2.LINE_AA)
    cv2.line(body, (ox + P(10), oy + P(53)), (W - ox - P(10), oy + P(53)), (255, 255, 255), max(1, ss), cv2.LINE_AA)
    # cream label with wordmark
    lw, lh = P(w * .76), P(h * .56); lx, ly = ox + (P(w) - lw) // 2, oy + P(h * .30)
    lab = np.zeros((H, W), np.uint8); cv2.rectangle(lab, (lx + P(24), ly), (lx + lw - P(24), ly + lh), 255, -1, cv2.LINE_AA)
    cv2.rectangle(lab, (lx, ly + P(24)), (lx + lw, ly + lh - P(24)), 255, -1, cv2.LINE_AA)
    for cx_, cy_ in ((lx + P(24), ly + P(24)), (lx + lw - P(24), ly + P(24)), (lx + P(24), ly + lh - P(24)), (lx + lw - P(24), ly + lh - P(24))): cv2.circle(lab, (cx_, cy_), P(24), 255, -1, cv2.LINE_AA)
    a = (lab.astype(np.float32) / 255)[..., None]; body = body * (1 - a) + CREAM[::-1] * a
    wm = hs_logo.wordmark(int(lw / ss * .78) * ss); wh, ww = wm.shape[:2]
    wx, wy = lx + (lw - ww) // 2, ly + (lh - wh) // 2
    blend_rgba_f(body, wm, wx, wy)
    # soft sheen
    sh = np.zeros((H, W), np.float32); cv2.rectangle(sh, (ox + P(20), oy + P(60)), (ox + P(30), oy + P(h - 10)), 1, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), P(4)) * .35; body = body * (1 - sh[..., None]) + 255 * sh[..., None]
    am = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), .8 * ss) / 255
    rgba = np.dstack([body.clip(0, 255), am * 255]).astype(np.uint8)
    rgba = cv2.resize(rgba, (W // ss, H // ss), interpolation=cv2.INTER_AREA)
    if tilt:
        M = cv2.getRotationMatrix2D((rgba.shape[1] / 2, rgba.shape[0] / 2), tilt, 1); rgba = cv2.warpAffine(rgba, M, (rgba.shape[1], rgba.shape[0]), flags=cv2.INTER_LINEAR)
    return rgba

def blend_rgba_f(f, rgba, x, y):
    h, w = rgba.shape[:2]; H, W = f.shape[:2]
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    rgba = rgba[y0 - y:y1 - y, x0 - x:x1 - x]; a = rgba[..., 3:4].astype(np.float32) / 255
    f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - a) + rgba[..., :3].astype(np.float32) * a

def screen7():
    im = rd('7__Preview_scroll.jpg'); sc = im.shape[1] / 1188.; Hh, Ww = im.shape[:2]
    eyes = find_eyes(im, 0, int(200 * sc), 450, 40000, thr=32); eyes = drop_island_eyes(eyes, sc)
    ui = np.zeros(im.shape[:2], bool); ui[:320] = 1
    im, G = ui_coral(im, region=ui)
    # --- packaging: replace the character-cap jars with two fictional sachets (positions measured on the 743px export)
    f = im.astype(np.float32); top = 1481; bot = 1828
    shadow = np.zeros(im.shape[:2], np.float32)
    A = pouch(358, 360, (147, 197, 114), (110, 160, 80), (122, 175, 92), tilt=-1.2)
    B = pouch(340, 360, (255, 176, 128), (225, 140, 96), (240, 150, 108), tilt=1.4)
    ax, ay = 8, 1492; bx, by = 356, 1496
    for rg, (x, y) in ((A, (ax, ay)), (B, (bx, by))):
        h, w = rg.shape[:2]
        sh3 = np.zeros((h, w, 4), np.uint8); sh3[..., :3] = 255; sh3[..., 3] = rg[..., 3]
        sh = np.zeros(im.shape[:2], np.float32); tmp = np.zeros((im.shape[0], im.shape[1], 3), np.float32)
        blend_rgba_f(tmp, sh3, x + 6, y + 8); shadow = np.maximum(shadow, cv2.GaussianBlur(tmp[..., 0] / 255., (0, 0), 9))
    reg = np.zeros(im.shape[:2], np.float32); reg[top:bot] = 1
    f = f * (1 - (shadow * .30 * reg)[..., None])
    for rg, (x, y) in ((A, (ax, ay)), (B, (bx, by))):
        h, w = rg.shape[:2]; yy1 = min(y + h, bot + 2)
        blend_rgba_f(f, rg[:yy1 - y], x, y)
    im = f.clip(0, 255).astype(np.uint8)
    wr('7__scroll_plain.jpg', im); wr('7__scroll.jpg', decorate(im, eyes)); return len(eyes)

# ------------------------------------------------------------------ Screen 8 : tagline
def screen8():
    im = rd('8__Tagline.jpg'); eyes = find_eyes(im, 2150, None, 1500)
    ya, yb = 1195, 1655; im2 = im.copy(); im2[ya:yb] = rowfill(im, ya, yb)[ya:yb]; im = im2
    ss = 4; SZ = 108; font = ImageFont.truetype(FONT, SZ * ss)
    layer = Image.new('RGBA', (1188 * ss, (yb - ya) * ss), (0, 0, 0, 0)); dr = ImageDraw.Draw(layer)
    top3 = 1502; asc = -font.getbbox('l', anchor='ls')[1] / ss; base3 = top3 + asc
    DARK = (56, 55, 51); lines = [('Good habits,', DARK), ('happy little', tuple(int(c) for c in CORAL)), ('humans.', tuple(int(c) for c in CORAL))]
    for i, (t, c) in enumerate(lines):
        base = base3 - (2 - i) * 138
        dr.text((100 * ss, (base - ya) * ss), t, font=font, fill=c + (255,), anchor='ls')
    small = layer.resize((1188, yb - ya), Image.LANCZOS); arr = np.array(small)
    a = arr[..., 3:4].astype(np.float32) / 255
    seg = im[ya:yb].astype(np.float32)
    im[ya:yb] = (seg * (1 - a) + arr[..., :3][..., ::-1] * a).astype(np.uint8)
    ui = np.zeros(im.shape[:2], bool); ui[1780:2000] = 1
    im, G = ui_coral(im, region=ui)
    save_pair('8__Tagline.jpg', im, decorate(im, eyes)); return len(eyes)

# ------------------------------------------------------------------ Screen 9
def screen9():
    im = rd('9__Sign_Up.jpg'); eyes = find_eyes(im, 30, None, 450); eyes = drop_island_eyes(eyes, im.shape[1] / 1188.)
    ch = char_green(im); im = to_pistachio(im, ch)
    ui = np.zeros(im.shape[:2], bool); ui[900:2300] = 1
    im, G = ui_coral(im, region=ui, exclude=ch)
    save_pair('9__Sign_Up.jpg', im, decorate(im, eyes)); return len(eyes)

if __name__ == '__main__':
    todo = sys.argv[2:] or ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    for k in todo: print(k, globals()['screen' + k]())
