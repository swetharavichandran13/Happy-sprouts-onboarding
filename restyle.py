"""Happy Sprouts restyle (v2, approved selections: F2 open-eye faces, leafy edges on every blob, G2 gummies, W3 logo).
Reads the ORIGINAL Little Joys exports (folder given as argv[1]) and writes edited exports to src/ (+ eye sprites to a/,
src/eyes.json). prep_all.py then slices src/ into the animation assets. The original repo is never touched.
usage: python3 restyle.py /path/to/original/exports"""
import cv2, numpy as np, os, sys, json
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFont
import charsvg

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
ORIG = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/'
OUT = 'src/'; os.makedirs(OUT, exist_ok=True); os.makedirs('a', exist_ok=True); os.makedirs('brand', exist_ok=True)
PAL = dict(teal='#4AB8BD', sage='#8AB17D', yellow='#E9C64A', apricot='#F4A261', coral='#E76F51')
rgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
CORAL = rgb(PAL['coral'])
CREAM = np.array([254, 251, 242], np.float32)
FONT = 'fonts/pjs700.ttf'
rd = lambda n: cv2.imread(ORIG + n)
def wr(n, im): cv2.imwrite(OUT + n, im, [cv2.IMWRITE_JPEG_QUALITY, 97])

# ------------------------------------------------------------------ colour helpers
def hsvm(im):
    h = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); return h[..., 0].astype(int), h[..., 1].astype(int), h[..., 2].astype(int)
def green_mask(im, smin=.22):
    H, S, V = hsvm(im); return (H >= 38) & (H <= 95) & (S / 255. > smin)
def red_mask(im, smin=.22):
    H, S, V = hsvm(im); return ((H <= 8) | (H >= 170)) & (S / 255. > smin) & (V > 120)

def to_coral(im, region, gmask_fn, G=None, ref_min=.6, auto_bg=False):
    """unmix an accent colour over its (cream/white) background and re-mix with coral (#E76F51), keeping anti-aliasing."""
    f = im[..., ::-1].astype(np.float32)
    gm = gmask_fn(im) & region; gm = ndi.binary_dilation(gm, iterations=2)
    G = np.median(f[gm & gmask_fn(im, ref_min)], 0) if G is None else G
    white = f.min(2) > 238; W = np.where(white[..., None], 255., CREAM)
    if auto_bg:   # tinted background (e.g. the pink wash behind the first Insight slide)
        far = region & ~ndi.binary_dilation(gm, iterations=8) & (f.min(2) > 200); wt = far.astype(np.float32); den = cv2.GaussianBlur(wt, (0, 0), 20) + 1e-4
        W = np.dstack([cv2.GaussianBlur(f[..., k] * wt, (0, 0), 20) / den for k in range(3)])
    if auto_bg:   # tinted background: red channel is ~255 for both, so measure t on G+B only and recolour the whole halo
        gb = lambda a: a[..., 1] + a[..., 2]
        t = np.clip((gb(W) - gb(f)) / (gb(W) - gb(G[None, None] if G.ndim == 1 else G) + 1e-3), 0, 1)
        zone = ndi.binary_dilation(gm, iterations=3) & region; a = np.where(zone, np.clip((t - .03) / .97, 0, 1), 0)[..., None]
        out = np.where(zone[..., None] & (t[..., None] > .03), W * (1 - t[..., None]) + CORAL * t[..., None], f)
        return out[..., ::-1].clip(0, 255).astype(np.uint8)
    d = W - G; t = np.clip(((W - f) * d).sum(2) / (d * d).sum(2), 0, 1)
    mix = W * (1 - t[..., None]) + G * t[..., None]
    ok = (np.abs(mix - f).max(2) < 46) & (t > .04) & gm
    new = W * (1 - t[..., None]) + CORAL * t[..., None]
    out = np.where(ok[..., None], new, f)
    # darker rim / shadow pixels of the same hue (e.g. under a button): scale coral by luminance
    res = gmask_fn(im, .1) & ndi.binary_dilation(ok, iterations=10) & ~ok & region
    lum = np.clip(f.mean(2) / G.mean(), .55, 1.1)
    out = np.where(res[..., None], CORAL * lum[..., None], out)
    return out[..., ::-1].clip(0, 255).astype(np.uint8)

def rowfill(img, y0, y1, xa=14, xb=1174):
    o = img.copy().astype(np.float32); t = np.linspace(0, 1, img.shape[1])[:, None]
    for y in range(y0, y1): o[y] = img[y, xa - 8:xa + 8].mean(0) * (1 - t) + img[y, xb - 8:xb + 8].mean(0) * t
    return o.astype(np.uint8)

def alpha_over(bg, rgba):
    """bg BGR uint8, rgba BGRA uint8 (same size) -> BGR"""
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (bg.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a).clip(0, 255).astype(np.uint8)

def resize_rgba(rgba, size):
    a = rgba[..., 3:4].astype(np.float32) / 255; pm = np.dstack([rgba[..., :3].astype(np.float32) * a, a * 255])
    pm = cv2.resize(pm, size, interpolation=cv2.INTER_AREA); aa = pm[..., 3:4] / 255
    col = np.where(aa > 1e-3, pm[..., :3] / np.maximum(aa, 1e-3), 0)
    return np.dstack([col, pm[..., 3]]).clip(0, 255).astype(np.uint8)

# status bar (clock, icons, island) as an alpha overlay measured on the clean Screen 2 export
def status_overlay():
    s2 = rd('2__Loading.jpg').astype(np.float32)[:150]; bg2 = np.median(s2[:, :60], axis=1)[:, None, :]
    Lm = lambda x: x @ np.array([.114, .587, .299], np.float32)
    al = np.clip((Lm(bg2) - Lm(s2)) / (Lm(bg2) - 34), 0, 1)
    return np.dstack([np.full(s2.shape[:2] + (3,), [37, 27, 24], np.float32), al * 255]).astype(np.uint8)
STATUS = status_overlay()
def add_status(im, sc=1.0, island=True):
    st = STATUS.copy() if abs(sc - 1) < 1e-3 else resize_rgba(STATUS, (int(round(STATUS.shape[1] * sc)), int(round(STATUS.shape[0] * sc))))
    if not island: st = st.copy(); st[:, int(412 * sc):int(776 * sc), 3] = 0   # carousel: the island is a separate static layer in index.html
    out = im.copy(); h, w = st.shape[:2]; out[:h, :w] = alpha_over(out[:h, :w], st[:, :im.shape[1]][:h]); return out

# ------------------------------------------------------------------ character detection / geometry
SPEC = dict(green=lambda H, S: (H >= 38) & (H <= 95) & (S > 90), yellow=lambda H, S: (H >= 15) & (H <= 35) & (S > 120),
            blue=lambda H, S: (H >= 95) & (H <= 130) & (S > 70), red=lambda H, S: ((H <= 8) | (H >= 170)) & (S > 120))
def find_chars(im, rows, sc, names):
    H, W = im.shape[:2]; h, s, v = hsvm(im); out = []
    for nm in names:
        m = SPEC[nm](h, s) & (v > 110); m[rows[1]:] = 0; m[:rows[0]] = 0
        m = ndi.binary_closing(m, iterations=3); L, n = ndi.label(m)
        for k, sl in enumerate(ndi.find_objects(L), 1):
            if sl is None: continue
            ys, xs = sl; area = (L[sl] == k).sum()
            near = ys.start < 40 * sc or xs.start < 40 * sc or ys.stop > H - 40 * sc or xs.stop > W - 40 * sc
            if area > 15000 * sc * sc and near: out.append(dict(name=nm, mask=ndi.binary_fill_holes(L == k)))
    return out

def find_eyes(im, rows, minpix, thr=14, maxwhite=40000):
    s = im.astype(int); mn = s.min(2); pup = (s.max(2) < thr); pup[:rows[0]] = 0; pup[rows[1]:] = 0
    L, n = ndi.label(pup); res = []
    for k in range(1, n + 1):
        m = L == k
        if m.sum() < minpix: continue
        d = ndi.binary_dilation(m, iterations=int(14 * (minpix / 1500) ** .5 + 4)); wh = (mn > 205) & (np.abs(s[..., 0] - s[..., 2]) < 30)
        wl, wn = ndi.label(wh | m); ids = np.unique(wl[d & wh]); ids = ids[ids > 0]; ids = [j for j in ids if (wl == j).sum() < maxwhite]
        em = ndi.binary_fill_holes(np.isin(wl, ids) | m); ys, xs = np.nonzero(em)
        res.append(dict(c=np.array([xs.mean(), ys.mean()]), w=float(xs.max() - xs.min()), mask=em))
    return res

def bridge_island(mask, sc):
    """slice exports have a baked dynamic island cutting through the top character: bridge it."""
    x0, x1, y0, y1 = int(412 * sc), int(776 * sc), 0, int(150 * sc)
    out = mask.copy(); H, W = mask.shape
    def lower(x):
        col = np.nonzero(mask[:, x])[0]; return col.max() if len(col) else 0
    lx = max(x0 - 6, 0); rx = min(x1 + 6, W - 1); yl, yr = lower(lx), lower(rx)
    for x in range(x0, x1):
        t = (x - x0) / max(x1 - x0, 1); yb = int(yl * (1 - t) + yr * t); out[:yb + 1, x] = True
    return out

def contour_pts(mask, grow=2):
    m = ndi.binary_dilation(ndi.binary_fill_holes(mask), iterations=grow).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)[::5]
    out = []
    for i in range(len(c)):
        a, b = c[i], c[(i + 1) % len(c)]; k = max(1, int(np.linalg.norm(b - a) / 4))
        for t in range(k): out.append(a + (b - a) * t / k)
    c = np.array(out); c[:, 0] = gaussian_filter1d(c[:, 0], 3, mode='wrap'); c[:, 1] = gaussian_filter1d(c[:, 1], 3, mode='wrap'); return c

def leafify(c, W, H, period=88., amp=17., rib=118.):
    """soft serrated contour + curved midrib + veins along the free (not screen-edge) part of the outline. c in 1188 coords."""
    cnt = c.astype(np.float32).reshape(-1, 1, 2)
    t = np.roll(c, -1, 0) - np.roll(c, 1, 0); t /= np.linalg.norm(t, axis=1)[:, None]; n = np.stack([t[:, 1], -t[:, 0]], 1)
    test = c[::40] + n[::40] * 6
    if np.mean([cv2.pointPolygonTest(cnt, (float(x), float(y)), False) > 0 for x, y in test]) > .5: n = -n
    near = (c[:, 0] > W - 14) | (c[:, 1] < 14) | (c[:, 0] < 14) | (c[:, 1] > H - 14)
    wgt = gaussian_filter1d((~near).astype(float), 25, mode='wrap'); wgt = np.clip((wgt - .5) * 2.5, 0, 1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    first = int(np.argmax(wgt > .99)) if (wgt > .99).any() else 0
    ph = ((s - s[first]) % period) / period; saw = np.where(ph < .72, ph / .72, (1 - ph) / .28)
    off = (saw - .35) * amp * wgt; cs = c + n * off[:, None]
    tips = [i for i in range(2, len(off) - 2) if off[i] >= off[i - 1] and off[i] > off[i + 1] and off[i] > amp * .5 and wgt[i] > .9]
    mid = c - n * rib
    mid = np.stack([gaussian_filter1d(mid[:, 0], 30, mode='wrap'), gaussian_filter1d(mid[:, 1], 30, mode='wrap')], 1)
    inside = np.array([cv2.pointPolygonTest(cnt, (float(x), float(y)), True) for x, y in mid])
    keep = (wgt > .95) & (inside > rib * .6) & (mid[:, 0] < W - 30) & (mid[:, 1] > 20) & (mid[:, 0] > 30)
    idx = np.nonzero(keep)[0]; midrib = []; veins = []
    if len(idx) > 10:
        runs = []; cur = [idx[0]]
        for i in idx[1:]:
            if i == cur[-1] + 1: cur.append(i)
            else: runs.append(cur); cur = [i]
        runs.append(cur); run = max(runs, key=len)
        if len(run) > 25:
            midrib = mid[run][::5].round(1).tolist()
            for ti in tips:
                j = min(run, key=lambda k: abs(k - ti))
                if abs(j - ti) < 60 and min(run) - 10 <= ti <= max(run) + 10:
                    a = mid[j]; b = cs[ti]; veins.append([a.round(1).tolist(), ((a + b) / 2).round(1).tolist(), (a + (b - a) * .82).round(1).tolist()])
    return dict(path=cs[::2].round(1).tolist(), midrib=midrib, veins=veins)

def clean_bg(im, erase, sc):
    """paint the erased region with the per-row background colour (backgrounds are flat cream / vertical gradient)."""
    H, W = im.shape[:2]; valid = ~ndi.binary_dilation(erase, iterations=max(2, int(8 * sc))) & (im.min(2) < 252)
    rowc = np.zeros((H, 3), np.float32); ok = np.zeros(H, bool)
    for y in range(H):
        if erase[y].any():
            v = valid[y]
            if v.sum() > 8: rowc[y] = np.median(im[y][v], 0); ok[y] = True
    ys = np.nonzero(ok)[0]
    for y in range(H):
        if erase[y].any() and not ok[y]: rowc[y] = rowc[ys[np.argmin(np.abs(ys - y))]]
    for k in range(3): rowc[:, k] = gaussian_filter1d(rowc[:, k], 3)
    a = cv2.GaussianBlur(erase.astype(np.float32), (0, 0), 2.0)[..., None]
    fill = np.broadcast_to(rowc[:, None, :], im.shape)
    return (im.astype(np.float32) * (1 - a) + fill * a).clip(0, 255).astype(np.uint8)

MANUAL = {('s1', 'green'): ([895, 193], [1150, 352])}   # second eye is cut by the screen edge
def frame_for(key, nm, mask, es, sc):
    mine = es
    if (key, nm) in MANUAL: e1, e2 = [np.array(p, float) * sc for p in MANUAL[(key, nm)]]
    elif len(mine) >= 2:
        mine = sorted(mine, key=lambda e: -e['w'])[:2]; mine = sorted(mine, key=lambda e: e['c'][0]); e1, e2 = mine[0]['c'], mine[1]['c']
    else: raise RuntimeError(f'need manual eyes for {key} {nm}: {mine}')
    v = (e2 - e1) / sc; return dict(mid=(((e1 + e2) / 2) / sc).round(1).tolist(), ang=float(np.degrees(np.arctan2(v[1], v[0]))), d=float(np.linalg.norm(v))), mine

CM = dict(s1=dict(green='teal', yellow='coral', blue='apricot'), s3=dict(yellow='apricot', green='teal'), s8=dict(yellow='apricot'),
          s9=dict(blue='teal', green='apricot'), p7=dict(yellow='yellow'), c5=dict(red='coral'), c6=dict(green='sage'))
JOBS = dict(s1=((0, 2576), 14, 1500, ('green', 'yellow', 'blue')), s3=((0, 2576), 14, 450, ('green', 'yellow')), s8=((2100, 2576), 14, 1500, ('yellow',)),
            s9=((0, 2576), 14, 450, ('green', 'blue')), p7=((0, 260), 32, 450, ('yellow',)), c5=((0, 300), 32, 600, ('red',)), c6=((0, 300), 32, 600, ('green',)))
ORDER = dict(s1=['yellow', 'blue', 'green'])

def edge_fill(m, sc):
    """characters cut by a screen edge: close any bite (old mouth) that opens to that edge"""
    out = m.copy()
    for k in range(4):
        r = np.rot90(out, k).copy(); band = max(8, int(40 * sc)); Hh = r.shape[0]
        if r[-band:].any():
            cols = np.nonzero(r[-band:].any(0))[0]
            for x in range(cols.min(), cols.max() + 1):
                rows = np.nonzero(r[max(0, Hh - int(500 * sc)):, x])[0]
                if len(rows): r[max(0, Hh - int(500 * sc)) + rows.max():, x] = True
        out = np.rot90(r, -k)
    return out

def redraw(key, im, prefn=None, sprites=False, screen=None, status=True, slice_idx=0):
    """erase the old characters of one image (or slice), run prefn on the clean background, draw the new leafy F2 characters."""
    rows, thr, mp, names = JOBS[key]; H, W = im.shape[:2]; sc = W / 1188.
    cs = find_chars(im, rows, sc, names); es = find_eyes(im, rows, mp, thr)
    dists = [ndi.distance_transform_edt(~c['mask']) for c in cs]
    for c in cs: c['eyes'] = []
    for e in es:
        dv = [d[int(e['c'][1]), int(e['c'][0])] for d in dists]; j = int(np.argmin(dv))
        if dv[j] < 120 * sc: cs[j]['eyes'].append(e)
    if key == 's1':   # yellow continues under the blue character
        y = next(c for c in cs if c['name'] == 'yellow'); b = next(c for c in cs if c['name'] == 'blue')
        y['mask'] = y['mask'] | (b['mask'] & ndi.binary_dilation(y['mask'], iterations=70))
    if key in ('c5', 'c6'):
        # several fragments (island) -> one character
        m = np.zeros((H, W), bool)
        for c in cs: m |= c['mask']
        m = bridge_island(ndi.binary_closing(m, iterations=6), sc); cs = [dict(name=cs[0]['name'], mask=m, eyes=[e for c in cs for e in c['eyes']])]
    if key == 'p7':
        cs = cs[:1]; cs[0]['mask'] = bridge_island(ndi.binary_closing(cs[0]['mask'], iterations=6), sc)
    for c in cs:   # eyes that bulge over the silhouette: absorb them so the outline has no bite out of it
        m = c['mask'].copy()
        for e in c['eyes']: m |= ndi.binary_dilation(e['mask'], iterations=max(1, int(4 * sc)))
        # old mouth (dark / red tongue), also when it is cut by the screen edge
        s_ = im.astype(int); cand = ((s_.max(2) < 110) | ((s_[..., 2] > 190) & (s_[..., 1] < 110) & (s_[..., 0] < 130))) & ndi.binary_dilation(c['mask'], iterations=max(3, int(25 * sc)))
        cand[:int(150 * sc)] = False; cand = ndi.binary_opening(cand, iterations=2); m |= ndi.binary_dilation(cand, iterations=max(1, int(3 * sc)))
        pad = 60; m = ndi.binary_closing(np.pad(m, pad, mode='edge'), iterations=max(4, int(22 * sc)))[pad:-pad, pad:-pad]; c['mask'] = edge_fill(ndi.binary_fill_holes(m), sc)
    erase = np.zeros((H, W), bool)
    for c in cs: erase |= ndi.binary_dilation(c['mask'], iterations=max(3, int(28 * sc)))
    for e in es: erase |= ndi.binary_dilation(e['mask'], iterations=max(3, int(14 * sc)))
    if key != 's8' and rows[0] == 0:   # baked dynamic island (re-added later by the status overlay / index)
        erase[int(24 * sc):int(152 * sc), int(406 * sc):int(782 * sc)] = True
    erase &= (np.arange(H)[:, None] >= rows[0]) if key == 's8' else True
    bg = clean_bg(im, erase, sc)
    if prefn: bg = prefn(bg)
    spec_chars = []; meta = []
    order = ORDER.get(key) or [c['name'] for c in cs]
    for nm in order:
        c = next(c for c in cs if c['name'] == nm)
        fr, mine = frame_for(key, nm, c['mask'], c['eyes'], sc)
        pts = contour_pts(c['mask']) / sc; cw, ch = 1188, H / sc
        lf = leafify(pts, cw, ch)
        def push(P, d=90):   # let outline points that sit on a screen edge run off-canvas so the gloss rim does not show along the edge
            P = np.array(P, float); P[P[:, 0] < 14, 0] -= d; P[P[:, 0] > cw - 14, 0] += d; P[P[:, 1] < 14, 1] -= d; P[P[:, 1] > ch - 14, 1] += d; return P.round(1).tolist()
        lf['path'] = push(lf['path'])
        spec_chars.append(dict(contour=push(pts), leaf=lf, colour=PAL[CM[key][nm]], frame=fr, eyes='sprite' if sprites else 'baked', name=nm))
    spec = dict(W=1188, H=int(round(H / sc)), layer='base', chars=spec_chars)
    base = charsvg.render(spec, f'/tmp/_base_{key}.png'); base = resize_rgba(base, (W, H)); out = alpha_over(bg, base)
    sprites_meta = []
    if sprites:
        spec['layer'] = 'eyes'; el = charsvg.render(spec, f'/tmp/_eyes_{key}.png'); a = el[..., 3] > 10; L, n = ndi.label(a)
        order_y = sorted(spec_chars, key=lambda c: c['frame']['mid'][1])
        i = 0
        for c in order_y:
            f = c['frame']; k = f['d'] / 300 * 1.1; ang = np.radians(f['ang'])
            for sx in (-150, 150):
                p = np.array(f['mid']) + np.array([np.cos(ang), np.sin(ang)]) * sx * k; x, y = int(round(p[0])), int(round(p[1]))
                lab = L[min(max(y, 0), L.shape[0] - 1), min(max(x, 0), L.shape[1] - 1)]
                assert lab > 0, (key, c['name'], p)
                ys, xs = np.nonzero(L == lab); x0, x1, y0, y1 = max(xs.min() - 4, 0), xs.max() + 5, max(ys.min() - 4, 0), ys.max() + 5
                crop = el[y0:y1, x0:x1].copy(); crop[..., 3] = np.where(L[y0:y1, x0:x1] == lab, crop[..., 3], 0)
                fn = f'a/eye{screen}_{i}.png'; cv2.imwrite(fn, crop)
                sprites_meta.append(dict(s=screen, f=fn, x=int(x0), y=int(y0), w=int(x1 - x0), h=int(y1 - y0))); i += 1
    if status: out = add_status(out, sc, island=key not in ('c5', 'c6'))
    return out, sprites_meta

# ------------------------------------------------------------------ W3 logo (Grandstander, teal "Happy" + coral "Sprouts" + teal seedling)
def make_logo():
    from playwright.sync_api import sync_playwright
    html = '''<!doctype html><meta charset=utf-8><style>@font-face{font-family:G;src:url(file://%s/fonts/grandstander-latin-800.woff2);font-weight:800}html,body{margin:0;background:transparent}</style><body></body><script>
const T='%s',C='%s',f='G',sz=150,w=800;const cv=document.createElement('canvas').getContext('2d');cv.font=`${w} ${sz}px ${f}`;
document.fonts.load(`${w} 20px G`).then(()=>{cv.font=`${w} ${sz}px ${f}`;const m=(t,i)=>{let x=0;for(let k=0;k<i;k++)x+=cv.measureText(t[k]).width;return{x,w:cv.measureText(t[i]).width}};
 const hh=cv.measureText('H').actualBoundingBoxAscent,xh=cv.measureText('o').actualBoundingBoxAscent,t2='Sprouts',wd=cv.measureText(t2).width;const ls=m(t2,6);const y1=hh+150,y2=y1+sz*.95;
 const sx=50+ls.x+ls.w*.5,sy=y2-xh+6;
 const sprout=`<g transform="translate(${sx} ${sy}) rotate(4) scale(1.45)" fill="${T}"><path d="M-2 0 C-2 -20 -1 -34 0 -46" stroke="${T}" stroke-width="7" fill="none" stroke-linecap="round"/><path d="M0 -40 C-6 -62 -30 -72 -52 -62 C-50 -40 -28 -30 0 -40Z"/><path d="M1 -46 C4 -72 26 -86 50 -78 C50 -54 28 -40 1 -46Z"/></g>`;
 const W=wd+100,H=y2+90;document.body.innerHTML=`<svg id=s xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W*2}" height="${H*2}"><text x="50" y="${y1}" font-family="${f}" font-weight="${w}" font-size="${sz}" fill="${T}">Happy</text><text x="50" y="${y2}" font-family="${f}" font-weight="${w}" font-size="${sz}" fill="${C}">${t2}</text>${sprout}</svg>`;window.__ok=true})
</script>''' % (HERE, PAL['teal'], PAL['coral'])
    with sync_playwright() as p:
        b = p.chromium.launch(); open('/tmp/_logo.html', 'w').write(html); pg = b.new_page(viewport={'width': 1800, 'height': 1000}); pg.goto('file:///tmp/_logo.html'); pg.wait_for_function('window.__ok===true'); pg.wait_for_timeout(300)
        pg.locator('#s').screenshot(path='/tmp/_logo.png', omit_background=True); b.close()
    im = cv2.imread('/tmp/_logo.png', cv2.IMREAD_UNCHANGED); ys, xs = np.nonzero(im[..., 3] > 6); pad = 12
    im = im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]; cv2.imwrite('brand/logo_w3.png', im); return im
LOGO = make_logo()
def logo_at(width): return resize_rgba(LOGO, (width, int(LOGO.shape[0] * width / LOGO.shape[1])))
def paste(im, rgba, x, y):
    h, w = rgba.shape[:2]; im[y:y + h, x:x + w] = alpha_over(im[y:y + h, x:x + w], rgba)

# ================================================================== SCREENS
EYES = []
def screen1():
    im = rd('1__Splash.jpg')
    def pre(bg):
        ya, yb = 860, 1380; o = bg.copy(); o[ya:yb] = rowfill(bg, ya, yb)[ya:yb]
        lg = logo_at(660); paste(o, lg, (1188 - lg.shape[1]) // 2, 1130 - lg.shape[0] // 2); return o
    out, m = redraw('s1', im, pre, sprites=True, screen=1); wr('1__Splash.jpg', out); EYES.extend(m)

def screen2():   # bodies are recoloured per sprite and the ground glows on the clean plate in prep_all.py (gum.py); this export stays as the original
    wr('2__Loading.jpg', rd('2__Loading.jpg'))

def screen3():
    im = rd('3__Age.jpg')
    def pre(bg):
        ui = np.zeros(bg.shape[:2], bool); ui[300:2175] = 1; return to_coral(bg, ui, green_mask)
    out, m = redraw('s3', im, pre); wr('3__Age.jpg', out)

def screen4():
    im = rd('4__Focus_scroll.jpg'); f = im[..., ::-1].astype(np.float32)
    gm = green_mask(im, .35); L, n = ndi.label(ndi.binary_closing(gm, iterations=2)); ring = np.zeros_like(gm)
    for k, sl in enumerate(ndi.find_objects(L), 1):
        if sl is None: continue
        ys, xs = sl
        if (xs.stop - xs.start) > .48 * im.shape[1] and (ys.stop - ys.start) > 300: ring |= (L == k)
    ring = ndi.binary_dilation(ring, iterations=3); gm = green_mask(im, .2) & ring
    G = np.median(f[gm & green_mask(im, .5)], 0); sat = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1] / 255.
    t = np.clip((sat - .15) / (np.median(sat[gm & green_mask(im, .5)]) - .15), 0, 1) * gm
    lum = f.mean(2) / G.mean(); new = CORAL[None, None] * np.clip(lum, .6, 1.12)[..., None]
    a = cv2.GaussianBlur(t.astype(np.float32), (0, 0), .7)[..., None]
    wr('4__Focus_scroll.jpg', (f * (1 - a) + new * a)[..., ::-1].clip(0, 255).astype(np.uint8))

def carousel(n):
    im = rd(f'{n}__' + ('Insight' if n == 5 else 'Solution') + '_carousel.jpg'); H, W = im.shape[:2]; outs = []
    for k in range(3):
        x0, x1 = int(round(k * W / 3)), int(round((k + 1) * W / 3)); cr = im[:, x0:x1].copy(); sc = cr.shape[1] / 1188.
        def pre(bg):
            ui = np.zeros(bg.shape[:2], bool); ui[int(150 * sc):845] = 1
            bg = to_coral(bg, ui, green_mask)
            if n == 5:
                rg = np.zeros(bg.shape[:2], bool); rg[330:640] = 1; bg = to_coral(bg, rg, red_mask, G=np.array([251, 86, 100], np.float32), auto_bg=True)
            return bg
        o, m = redraw(f'c{n}', cr, pre); outs.append(o)
    w = min(o.shape[1] for o in outs); tile = np.hstack([o[:, :w] for o in outs]); wr(f'{n}__carousel.jpg', tile)

# ---- Screen 7: sachets (W3 logo) + coral Skip + approved characters
def pouch(w, h, col, shade, seal, tilt=0, ss=3):
    W, H = int((w + 60) * ss), int((h + 60) * ss); P = lambda v: int(v * ss); ox = 30 * ss; oy = 30 * ss
    mask = np.zeros((H, W), np.uint8)
    left = [(ox + P(6) - P(10 * np.sin(np.pi * min(s_ * .9, 1))), oy + P(s_ * h)) for s_ in np.linspace(0, 1, 60)]
    right = [(ox + P(w - 6) + P(10 * np.sin(np.pi * min(s_ * .9, 1))), oy + P(s_ * h)) for s_ in np.linspace(0, 1, 60)]
    cv2.fillPoly(mask, [np.array(left + right[::-1], np.int32)], 255, cv2.LINE_AA)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P(14), P(14))))
    for sx in (ox + P(2), ox + P(w - 2)): cv2.circle(mask, (sx, oy + P(22)), P(5), 0, -1, cv2.LINE_AA)
    body = np.zeros((H, W, 3), np.float32); body[:] = col[::-1]; xs = np.linspace(0, 1, W)[None, :, None]
    body *= (1 - .16 * (xs ** 2.2) - .10 * ((1 - xs) ** 4))
    sealm = np.zeros((H, W), np.float32); sealm[oy:oy + P(34)] = 1
    body = body * (1 - sealm[..., None]) + np.array(seal[::-1], np.float32) * sealm[..., None]
    for x in range(ox, W - ox, P(7)): body[oy:oy + P(34), x:x + max(1, ss)] *= .90
    cv2.line(body, (ox, oy + P(34)), (W - ox, oy + P(34)), tuple(float(c) * .8 for c in shade[::-1]), max(1, ss), cv2.LINE_AA)
    cv2.line(body, (ox + P(10), oy + P(50)), (W - ox - P(10), oy + P(50)), tuple(float(c) * .85 for c in col[::-1]), P(2), cv2.LINE_AA)
    cv2.line(body, (ox + P(10), oy + P(53)), (W - ox - P(10), oy + P(53)), (255., 255., 255.), max(1, ss), cv2.LINE_AA)
    lw, lh = P(w * .80), P(h * .52); lx, ly = ox + (P(w) - lw) // 2, oy + P(h * .30)
    lab = np.zeros((H, W), np.uint8); r = P(24)
    cv2.rectangle(lab, (lx + r, ly), (lx + lw - r, ly + lh), 255, -1, cv2.LINE_AA); cv2.rectangle(lab, (lx, ly + r), (lx + lw, ly + lh - r), 255, -1, cv2.LINE_AA)
    for cx_, cy_ in ((lx + r, ly + r), (lx + lw - r, ly + r), (lx + r, ly + lh - r), (lx + lw - r, ly + lh - r)): cv2.circle(lab, (cx_, cy_), r, 255, -1, cv2.LINE_AA)
    a = (lab.astype(np.float32) / 255)[..., None]; body = body * (1 - a) + CREAM[::-1] * a
    lg = logo_at(int(lw * .80)); wh, ww = lg.shape[:2]; wx, wy = lx + (lw - ww) // 2, ly + (lh - wh) // 2
    ab = lg[..., 3:4].astype(np.float32) / 255; body[wy:wy + wh, wx:wx + ww] = body[wy:wy + wh, wx:wx + ww] * (1 - ab) + lg[..., :3] * ab
    sh = np.zeros((H, W), np.float32); cv2.rectangle(sh, (ox + P(20), oy + P(60)), (ox + P(30), oy + P(h - 10)), 1, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), P(4)) * .35; body = body * (1 - sh[..., None]) + 255 * sh[..., None]
    am = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), .8 * ss) / 255
    rg = cv2.resize(np.dstack([body.clip(0, 255), am * 255]).astype(np.uint8), (W // ss, H // ss), interpolation=cv2.INTER_AREA)
    if tilt:
        M = cv2.getRotationMatrix2D((rg.shape[1] / 2, rg.shape[0] / 2), tilt, 1); rg = cv2.warpAffine(rg, M, (rg.shape[1], rg.shape[0]), flags=cv2.INTER_LINEAR)
    return rg

def blend_f(f, rgba, x, y):
    h, w = rgba.shape[:2]; H, W = f.shape[:2]; x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    rgba = rgba[y0 - y:y1 - y, x0 - x:x1 - x]; a = rgba[..., 3:4].astype(np.float32) / 255
    f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - a) + rgba[..., :3].astype(np.float32) * a

def screen7():
    im = rd('7__Preview_scroll.jpg')
    def pre(bg):
        ui = np.zeros(bg.shape[:2], bool); ui[:320] = 1; bg = to_coral(bg, ui, green_mask)
        f = bg.astype(np.float32); top, bot = 1481, 1828; shadow = np.zeros(bg.shape[:2], np.float32)
        A = pouch(358, 360, rgb(PAL['sage']), (110, 150, 98), (108, 150, 94), tilt=-1.2); B = pouch(340, 360, rgb(PAL['apricot']), (225, 140, 96), (232, 140, 92), tilt=1.4)
        pl = ((A, (8, 1492)), (B, (356, 1496)))
        for rg, (x, y) in pl:
            h, w = rg.shape[:2]; sh3 = np.zeros((h, w, 4), np.uint8); sh3[..., :3] = 255; sh3[..., 3] = rg[..., 3]
            tmp = np.zeros(bg.shape, np.float32); blend_f(tmp, sh3, x + 6, y + 8); shadow = np.maximum(shadow, cv2.GaussianBlur(tmp[..., 0] / 255., (0, 0), 9))
        reg = np.zeros(bg.shape[:2], np.float32); reg[top:bot] = 1; f = f * (1 - (shadow * .30 * reg)[..., None])
        for rg, (x, y) in pl: h, w = rg.shape[:2]; yy1 = min(y + h, bot + 2); blend_f(f, rg[:yy1 - y][..., [2, 1, 0, 3]][..., [2, 1, 0, 3]], x, y)
        return f.clip(0, 255).astype(np.uint8)
    out, m = redraw('p7', im, pre, status=False); wr('7__scroll.jpg', out)

# ---- Screen 8: tagline
def screen8():
    im = rd('8__Tagline.jpg')
    def pre(bg):
        ya, yb = 1195, 1655; o = bg.copy(); o[ya:yb] = rowfill(bg, ya, yb)[ya:yb]
        ss = 4; SZ = 108; font = ImageFont.truetype(FONT, SZ * ss); layer = Image.new('RGBA', (1188 * ss, (yb - ya) * ss), (0, 0, 0, 0)); dr = ImageDraw.Draw(layer)
        asc = -font.getbbox('l', anchor='ls')[1] / ss; base3 = 1502 + asc; DARK = (56, 55, 51); cc = tuple(int(c) for c in CORAL)
        for i, (t, c) in enumerate([('Good habits,', DARK), ('happy little', cc), ('humans.', cc)]):
            dr.text((100 * ss, (base3 - (2 - i) * 138 - ya) * ss), t, font=font, fill=c + (255,), anchor='ls')
        arr = np.array(layer.resize((1188, yb - ya), Image.LANCZOS)); a = arr[..., 3:4].astype(np.float32) / 255
        o[ya:yb] = (o[ya:yb].astype(np.float32) * (1 - a) + arr[..., :3][..., ::-1] * a).astype(np.uint8)
        ui = np.zeros(o.shape[:2], bool); ui[1780:2000] = 1; return to_coral(o, ui, green_mask)
    out, m = redraw('s8', im, pre, sprites=True, screen=8, status=False); wr('8__Tagline.jpg', out); EYES.extend(m)

def screen9():
    im = rd('9__Sign_Up.jpg')
    def pre(bg):
        ui = np.zeros(bg.shape[:2], bool); ui[900:2300] = 1; return to_coral(bg, ui, green_mask)
    out, m = redraw('s9', im, pre, sprites=True, screen=9); wr('9__Sign_Up.jpg', out); EYES.extend(m)

if __name__ == '__main__':
    todo = sys.argv[2:] or ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    for k in todo:
        if k in ('5', '6'): carousel(int(k)); print(k, 'ok')
        else: globals()['screen' + k](); print(k, 'ok')
    if os.path.exists('src/eyes.json') and len(sys.argv) > 2:
        old = [e for e in json.load(open('src/eyes.json')) if str(e['s']) not in [t for t in todo]]; EYES[:0] = old
    json.dump(sorted(EYES, key=lambda e: (e['s'], e['f'])), open('src/eyes.json', 'w'))
    charsvg.close()
