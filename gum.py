"""G2 gummy recolour (LAB: keep the lightness structure, replace chroma)."""
import cv2,numpy as np
PAL=dict(teal='4AB8BD',sage='8AB17D',yellow='E9C64A',apricot='F4A261',coral='E76F51')
def rgb(h):return np.array([int(h[i:i+2],16) for i in (0,2,4)],np.float32)
def lab(c):return cv2.cvtColor(np.uint8([[c[::-1]]]),cv2.COLOR_BGR2LAB)[0,0].astype(np.float32)
def recolor(src,col):
    im=cv2.imread(src,-1);a=im[...,3].astype(np.float32)/255;bgr=im[...,:3]
    # clean ground-shadow smear: keep only reasonably opaque, saturated body
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);body=(a>.5)&(hsv[...,1]>70)
    body=cv2.morphologyEx(body.astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8));body=cv2.dilate(body,np.ones((3,3),np.uint8))
    a=cv2.GaussianBlur(body.astype(np.float32),(0,0),1.0)
    L=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).astype(np.float32);m=body>0
    Lm=np.median(L[...,0][m]);t=lab(rgb(PAL[col])[::-1][::-1]) if False else lab(rgb(PAL[col]))
    # lab() expects RGB order reversed to BGR inside
    out=L.copy();dl=(L[...,0]-Lm)
    out[...,0]=np.clip(t[0]+dl*1.05,0,255)
    white=np.clip((L[...,0]-Lm-40)/60,0,1)  # keep speculars whiter
    out[...,1]=t[1]*(1-white)+128*white;out[...,2]=t[2]*(1-white)+128*white
    o=cv2.cvtColor(out.clip(0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
    return np.dstack([o,(a*255).astype(np.uint8)])

from scipy import ndimage as ndi
def glow_recolor(img, PALh=PAL):
    """recolour the soft ground glows of a gummy-free plate: green->teal, red->coral, yellow->sage, blue->apricot"""
    fl = img[..., ::-1].astype(np.float32); bg = np.median(fl[:, :60], axis=1)[:, None, :]
    refs = [np.array(c, np.float32) for c in ((60, 160, 40), (232, 40, 50), (250, 170, 20), (30, 110, 230))]
    tg = [rgb(PALh[n]) for n in ('teal', 'coral', 'sage', 'apricot')]
    d = bg - fl; T = []; C = []
    for r in refs:
        dr = bg[0, 0] - r; T.append(np.clip((d * dr).sum(2) / (dr * dr).sum(), 0, 1)); C.append((d * dr).sum(2) / (np.linalg.norm(d, axis=2) * np.linalg.norm(dr) + 1e-3))
    k = np.argmax(np.stack(C), 0); t = np.take_along_axis(np.stack(T), k[None], 0)[0]
    m = (np.abs(d).max(2) > 3) & (t > .01); m[:1200] = False; m[1650:] = False
    out = fl.copy()
    for j in range(4):
        out = np.where((m & (k == j))[..., None], bg + t[..., None] * (tg[j] - bg), out)
    return out[..., ::-1].clip(0, 255).astype(np.uint8)
