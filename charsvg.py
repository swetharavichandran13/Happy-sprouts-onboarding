"""Vector renderer for the Happy Sprouts blob characters (glossy body, leafy edge, F2 open-eye face).
render(spec) -> RGBA numpy image. spec = dict(W, H, layer='base'|'eyes', chars=[...]) in 1188-wide coordinates."""
import json, os, numpy as np, cv2
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
INK = '#3d2219'

PAGE = r'''<!doctype html><meta charset=utf-8><style>html,body{margin:0;background:transparent}svg{display:block}</style><body></body><script>
const S=__SPEC__;const INK='__INK__';
const smooth=pts=>{const n=pts.length;const mid=(a,b)=>[(a[0]+b[0])/2,(a[1]+b[1])/2];let m0=mid(pts[0],pts[1]);let d=`M${m0[0].toFixed(1)} ${m0[1].toFixed(1)}`;
 for(let i=1;i<=n;i++){const p=pts[i%n],q=pts[(i+1)%n],m=mid(p,q);d+=`Q${p[0].toFixed(1)} ${p[1].toFixed(1)} ${m[0].toFixed(1)} ${m[1].toFixed(1)}`}return d+'Z'};
const open=pts=>{let d=`M${pts[0][0]} ${pts[0][1]}`;for(let i=1;i<pts.length-1;i++){const m=[(pts[i][0]+pts[i+1][0])/2,(pts[i][1]+pts[i+1][1])/2];d+=`Q${pts[i][0]} ${pts[i][1]} ${m[0].toFixed(1)} ${m[1].toFixed(1)}`}const l=pts[pts.length-1];return d+`L${l[0]} ${l[1]}`};
const defs=`<defs>
<filter id="bl" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="9"/></filter>
<filter id="gloss" filterUnits="userSpaceOnUse" x="-300" y="-300" width="${S.W+600}" height="${S.H+600}">
 <feOffset in="SourceAlpha" dx="22" dy="28" result="o1"/><feComposite in="SourceAlpha" in2="o1" operator="out" result="rimTL"/><feGaussianBlur in="rimTL" stdDeviation="11" result="rb1"/>
 <feComposite in="rb1" in2="SourceAlpha" operator="in" result="rb1c"/><feFlood flood-color="#fff" flood-opacity=".62"/><feComposite in2="rb1c" operator="in" result="hi"/>
 <feOffset in="SourceAlpha" dx="-26" dy="-34" result="o2"/><feComposite in="SourceAlpha" in2="o2" operator="out" result="rimBR"/><feGaussianBlur in="rimBR" stdDeviation="16" result="rb2"/>
 <feComposite in="rb2" in2="SourceAlpha" operator="in" result="rb2c"/><feFlood flood-color="#7a2a14" flood-opacity=".30"/><feComposite in2="rb2c" operator="in" result="sh"/>
 <feMerge><feMergeNode in="SourceGraphic"/><feMergeNode in="sh"/><feMergeNode in="hi"/></feMerge></filter>
<filter id="drop" filterUnits="userSpaceOnUse" x="-300" y="-300" width="${S.W+600}" height="${S.H+600}"><feGaussianBlur in="SourceAlpha" stdDeviation="26"/><feOffset dy="22"/><feComponentTransfer><feFuncA type="linear" slope=".30"/></feComponentTransfer></filter>
<radialGradient id="vol" cx=".3" cy=".25" r=".9"><stop offset="0" stop-color="#fff" stop-opacity=".30"/><stop offset=".5" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".10"/></radialGradient>
<clipPath id="cv"><rect width="${S.W}" height="${S.H}"/></clipPath></defs>`;
function body(c){let d=smooth(c.leaf?c.leaf.path:c.contour);
 let s=`<path d="${d}" fill="#7a5a3a" filter="url(#drop)" opacity=".7"/><path d="${d}" fill="${c.colour}" filter="url(#gloss)"/><path d="${d}" fill="url(#vol)"/>`;
 if(c.leaf&&c.leaf.midrib&&c.leaf.midrib.length>3){const mr=c.leaf.midrib;
  s+=`<g fill="none" stroke-linecap="round" clip-path="url(#cv)"><path d="${open(mr)}" stroke="#000" stroke-opacity=".08" stroke-width="12" transform="translate(3 5)"/><path d="${open(mr)}" stroke="#fff" stroke-opacity=".42" stroke-width="6"/>`;
  for(const v of c.leaf.veins){const dx=v[2][0]-v[0][0],dy=v[2][1]-v[0][1],L=Math.hypot(dx,dy)||1;const cx=(v[0][0]+v[2][0])/2+dy/L*26+dx/L*30,cy=(v[0][1]+v[2][1])/2-dx/L*26+dy/L*30;const q=`M${v[0]} Q${cx.toFixed(1)} ${cy.toFixed(1)} ${v[2]}`;
   s+=`<path d="${q}" stroke="#000" stroke-opacity=".07" stroke-width="7" transform="translate(2 3)"/><path d="${q}" stroke="#fff" stroke-opacity=".40" stroke-width="4.2"/>`}s+='</g>'}
 return s}
function eyeShape(x){return `<g transform="translate(${x} 0)"><ellipse rx="44" ry="54" fill="${INK}"/><ellipse cx="-4" cy="-6" rx="36" ry="44" fill="#fff" opacity=".07"/><circle cx="-14" cy="-20" r="15" fill="#fff" opacity=".96"/><circle cx="15" cy="17" r="7" fill="#fff" opacity=".9"/></g>`}
function face(c,mode){const f=c.frame;const light=['#E9C64A','#F4A261'].includes(c.colour);const B=light?'#E76F51':'#F4A261',bo=light?.5:.75;
 const k=f.d/300*1.1;let s=`<g transform="translate(${f.mid[0]} ${f.mid[1]}) rotate(${f.ang}) scale(${k})">`;
 if(mode!=='eyes'){
  s+=`<ellipse cx="-215" cy="72" rx="56" ry="32" fill="${B}" opacity="${bo}" filter="url(#bl)"/><ellipse cx="215" cy="72" rx="56" ry="32" fill="${B}" opacity="${bo}" filter="url(#bl)"/>`;
  s+=`<path d="M-40 106 Q-20 138 0 106 Q20 138 40 106" stroke="${INK}" stroke-width="10" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`}
 if(c.eyes!=='none'&&(mode==='eyes'||(mode==='base'&&c.eyes==='baked')))s+=eyeShape(-150)+eyeShape(150);
 return s+'</g>'}
let out=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${S.W} ${S.H}" width="${S.W}" height="${S.H}">${defs}<g clip-path="url(#cv)">`;
for(const c of S.chars){if(S.layer==='base')out+=body(c)+face(c,'base');else if(c.eyes==='sprite')out+=face(c,'eyes')}
document.body.innerHTML=out+'</g></svg>';window.__ok=true;
</script>'''

_pw = None; _br = None
def _browser():
    global _pw, _br
    if _br is None:
        _pw = sync_playwright().start(); _br = _pw.chromium.launch()
    return _br

def render(spec, out_png):
    html = PAGE.replace('__SPEC__', json.dumps(spec)).replace('__INK__', INK)
    p = _browser().new_page(viewport={'width': int(spec['W']), 'height': int(spec['H'])})
    p.set_content(html); p.wait_for_function('window.__ok===true')
    p.screenshot(path=out_png, omit_background=True, full_page=False); p.close()
    return cv2.imread(out_png, cv2.IMREAD_UNCHANGED)

def close():
    global _pw, _br
    if _br: _br.close(); _pw.stop(); _br = None
