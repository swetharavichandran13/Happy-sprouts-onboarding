import cv2,numpy as np,json
from scipy import ndimage as ndi
U='src/'
names={1:'1__Splash.jpg',2:'2__Loading.jpg',3:'3__Age.jpg',8:'8__Tagline.jpg',9:'9__Sign_Up.jpg'}
S={i:cv2.imread(U+n) for i,n in names.items()}
# plain = before brows/catchlights, used to build the eyelid patches so blinks stay clean
SP={i:cv2.imread(U+n.replace('.jpg','__plain.jpg')) if i in (1,8,9) else S[i] for i,n in names.items()}
for i,s in S.items(): cv2.imwrite(f'a/s{i}.jpg',s,[cv2.IMWRITE_JPEG_QUALITY,97])
# ---------- eyes (screens 1,8,9)
def eyes(i,ymin,ymax,minpix):
    s=SP[i].astype(int);mn=s.min(2);pup=(s.max(2)<14);pup[:ymin]=0;pup[ymax:]=0
    L,n=ndi.label(pup);out=[]
    for k in range(1,n+1):
        m=L==k
        if m.sum()<minpix:continue
        d=ndi.binary_dilation(m,iterations=14)
        wh=(mn>205)&(np.abs(s[...,0]-s[...,2])<30)
        wl,wn=ndi.label(wh|m);ids=np.unique(wl[d&wh]);ids=ids[ids>0]
        ids=[j for j in ids if (wl==j).sum()<40000]
        out.append(ndi.binary_fill_holes(np.isin(wl,ids)|m))
    return out
P=[]
for i,(a,b,mp) in {1:(150,2576,1500),8:(2150,2576,1500),9:(30,2576,450)}.items():
    for j,em in enumerate(eyes(i,a,b,mp)):
        ys,xs=np.nonzero(em)
        dm=ndi.binary_dilation(em,iterations=9);ring=ndi.binary_dilation(em,iterations=34)&~ndi.binary_dilation(em,iterations=12)
        s=SP[i].astype(float);yy,xx=np.nonzero(ring);ok=(s[yy,xx].min(1)>40)&(s[yy,xx].max(1)<256);yy,xx=yy[ok],xx[ok]
        pad=40;x0,x1,y0,y1=max(xs.min()-pad,0),min(xs.max()+pad,1187),max(ys.min()-pad,0),min(ys.max()+pad,2575)
        X=lambda x,y:np.stack([np.ones_like(x),x,y,x*x,x*y,y*y],1)
        gx,gy=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1));gx=gx.astype(float);gy=gy.astype(float)
        sx,sy=(xx-x0)/100.,(yy-y0)/100.;fill=np.zeros(gx.shape+(3,))
        for c in range(3):
            w=np.linalg.lstsq(X(sx,sy),s[yy,xx,c],rcond=None)[0]
            fill[...,c]=(X(((gx-x0)/100.).ravel(),((gy-y0)/100.).ravel())@w).reshape(gx.shape)
        fill=np.clip(fill,0,255)
        al=cv2.GaussianBlur(dm[y0:y1+1,x0:x1+1].astype(np.float32),(0,0),2.5)
        cx,cy=int(xs.mean()-x0),int(ys.mean()-y0);r=int((xs.max()-xs.min())*.32)
        crease=np.zeros(gx.shape,np.uint8);cv2.ellipse(crease,(cx,cy-r//3),(r,int(r*.55)),0,15,165,255,7,cv2.LINE_AA)
        crease=cv2.GaussianBlur(crease,(0,0),1.2)[...,None]/255*.55
        fill=fill*(1-crease)+fill*.45*crease
        fn=f'a/eye{i}_{j}.png';cv2.imwrite(fn,np.dstack([fill,al*255]).astype(np.uint8))
        P.append(dict(s=i,f=fn,x=int(x0),y=int(y0),w=int(x1-x0+1),h=int(y1-y0+1),ey0=int(ys.min()-y0),ey1=int(ys.max()-y0)))
print('eyes per screen',{i:sum(p['s']==i for p in P) for i in (1,8,9)})
# ---------- gummies (screen 2)
s=S[2].astype(np.float32);bg=np.median(s[:,:60],axis=1)[:,None,:];body=np.abs(s-bg).max(2)>48;body[:300]=0
L,n=ndi.label(ndi.binary_dilation(body,iterations=3));G=[]
for k in range(1,n+1):
    m=L==k
    if m.sum()<8000:continue
    ys,xs=np.nonzero(m);G.append((xs.min(),xs.max(),ys.min(),ys.max(),m))
G.sort(key=lambda g:g[0]);meta=[];allm=np.zeros(body.shape,np.uint8)
for (x0,x1,y0,y1,m),c in zip(G,['green','red','yellow','blue']):
    mm=ndi.binary_fill_holes(m);allm|=mm.astype(np.uint8)
    a=cv2.GaussianBlur(ndi.binary_erosion(mm,iterations=1).astype(np.float32),(0,0),1.3);p=8;sl=(slice(y0-p,y1+p),slice(x0-p,x1+p))
    cv2.imwrite(f'a/g_{c}.png',np.dstack([S[2][sl],(a[sl]*255).astype(np.uint8)]));meta.append(dict(c=c,x=int(x0-p),y=int(y0-p),w=int(x1-x0+2*p),h=int(y1-y0+2*p)))
cv2.imwrite('a/s2_clean.jpg',cv2.inpaint(S[2].copy(),cv2.dilate(allm,np.ones((9,9),np.uint8)),9,cv2.INPAINT_TELEA),[cv2.IMWRITE_JPEG_QUALITY,97])
# ---------- status overlay
s2=S[2].astype(np.float32)[:150];bg2=np.median(s2[:,:60],axis=1)[:,None,:]
Lm=lambda x:x@np.array([.114,.587,.299],np.float32)
al=np.clip((Lm(bg2)-Lm(s2))/(Lm(bg2)-34),0,1)
cv2.imwrite('a/status.png',np.dstack([np.full(s2.shape[:2]+(3,),[37,27,24],np.float32),al*255]).astype(np.uint8))
# ---------- age picker (screen 3)
def rowfill(img,y0,y1,xa=14,xb=1174):
    o=img.copy().astype(np.float32);t=np.linspace(0,1,img.shape[1])[:,None]
    for y in range(y0,y1):o[y]=img[y,xa-8:xa+8].mean(0)*(1-t)+img[y,xb-8:xb+8].mean(0)*t
    return o.astype(np.uint8)
s3=S[3].astype(np.float32);Y0,Y1=690,1990;c=S[3].copy();c[Y0:Y1]=rowfill(S[3],Y0,Y1)[Y0:Y1];base=c.astype(np.float32)
pc=s3[1215,380];m=(np.abs(s3-pc).max(2)<7);m[:1100]=0;m[1600:]=0;m[:,:300]=0;m[:,900:]=0
ys,xs=np.nonzero(ndi.binary_opening(m,iterations=6));x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
ss=4;big=np.zeros(((y1-y0+40)*ss,(x1-x0+40)*ss),np.uint8)
cv2.rectangle(big,(80*ss,20*ss),((x1-x0+20-60)*ss,(y1-y0+20)*ss),255,-1);cv2.rectangle(big,(20*ss,80*ss),((x1-x0+20)*ss,(y1-y0+20-60)*ss),255,-1)
for cx,cy in((80,80),(x1-x0-40,80),(80,y1-y0-40),(x1-x0-40,y1-y0-40)):cv2.circle(big,(cx*ss,cy*ss),60*ss,255,-1)
a=cv2.resize(big,(x1-x0+40,y1-y0+40),interpolation=cv2.INTER_AREA)[...,None]/255.
reg=base[y0-20:y1+20,x0-20:x1+20];base[y0-20:y1+20,x0-20:x1+20]=reg*(1-a)+pc*a
cv2.imwrite('a/s3_base.jpg',base.astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,97]);YC=float((y0+y1)/2)
bgclean=cv2.imread('a/s3_base.jpg').astype(np.float32)
def sprites(y_a,y_b,bgf,tag,col,xl=250,xr=950):
    if bgf.ndim==1:bgf=np.broadcast_to(bgf,s3.shape).copy()
    d=np.abs(bgf[y_a:y_b]-s3[y_a:y_b]).max(2)>22;d[:,:xl]=0;d[:,xr:]=0
    rows=np.nonzero(d.any(1))[0];segs=[];st=pr=rows[0]
    for r in rows[1:]:
        if r-pr>18:segs.append((st,pr));st=r
        pr=r
    segs.append((st,pr));out=[]
    for i,(a0,a1) in enumerate(segs):
        ya,yb=y_a+a0-6,y_a+a1+7;cs=np.nonzero(d[a0:a1+1].any(0))[0];xa,xb=cs.min()-6,cs.max()+7
        px=s3[ya:yb,xa:xb];bgp=bgf[ya:yb,xa:xb];Lb,Lp=Lm(bgp),Lm(px);k=np.argmax(Lb-Lp);lc=Lp.ravel()[k]
        al=np.clip((Lb-Lp)/np.maximum(Lb-lc,1),0,1)
        fn=f'a/w_{tag}{i}.png';cv2.imwrite(fn,np.dstack([np.broadcast_to(np.array(col,np.float32),px.shape),al*255]).astype(np.uint8))
        out.append(dict(f=fn[2:],w=int(xb-xa),h=int(yb-ya),cx=float((xa+xb)/2),cy=float((ya+yb)/2)))
    return out
Gc=[178,172,172];D=[50,50,56]
top=sprites(700,1170,bgclean,'t',Gc);bot=sprites(1580,1990,bgclean,'b',Gc);sel=[q for q in sprites(int(y0)+10,int(y1)-10,pc,'s',D,365,825) if q['h']>30]
print('wheel',len(top),len(bot),len(sel))
W=dict(top=top,bot=bot,sel=sel,YC=YC)
open('data.js','w').write('const EYES=%s,GUM=%s,WHEEL=%s;'%(json.dumps(P),json.dumps(meta),json.dumps(W)))
# ---------- carousels
for n,f in ((5,'5__carousel.jpg'),(6,'6__carousel.jpg')):
    im=cv2.imread(U+f);H,Wd=im.shape[:2]
    for k in range(3):
        s_=im[:,int(round(k*Wd/3)):int(round((k+1)*Wd/3))].copy();sc=s_.shape[1]/1188
        ya,yb,xa,xb=int(26*sc),int(146*sc),int(412*sc),int(776*sc)
        for y in range(ya,yb):
            l=s_[y,xa-6:xa-2].mean(0);r=s_[y,xb+2:xb+6].mean(0);t=np.linspace(0,1,xb-xa)[:,None];s_[y,xa:xb]=l*(1-t)+r*t
        cv2.imwrite(f'b/c{n}_{k}.jpg',s_,[cv2.IMWRITE_JPEG_QUALITY,96])
# ---------- scroll pages
for n,f,isl in ((4,'4__Focus_scroll.jpg',0),(7,'7__scroll.jpg',1)):
    im=cv2.imread(U+f);sc=im.shape[1]/1188;g=Lm(im.astype(np.float32));mask=np.zeros(im.shape[:2],np.uint8)
    for (xa,xb,ya,yb,th) in ((150,292,52,125,105),(838,1078,58,110,105)):
        a_,b_,c_,d_=int(xa*sc),int(xb*sc),int(ya*sc),int(yb*sc);mask[c_:d_,a_:b_]=(g[c_:d_,a_:b_]<th)
    if isl:
        a_,b_,c_,d_=int(412*sc),int(776*sc),int(30*sc),int(146*sc);sub=im[c_:d_,a_:b_].astype(int);mask[c_:d_,a_:b_]|=(np.abs(sub-[37,27,24]).max(2)<24).astype(np.uint8)
    mask=cv2.dilate(mask,np.ones((5,5),np.uint8));cv2.imwrite(f'b/p{n}.jpg',cv2.inpaint(im,mask,5,cv2.INPAINT_TELEA),[cv2.IMWRITE_JPEG_QUALITY,95])
    print('page',n,im.shape,'-> height at 1188 wide',round(im.shape[0]/sc))
