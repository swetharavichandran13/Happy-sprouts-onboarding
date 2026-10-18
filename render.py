"""Render the Happy Sprouts cover-flow to MP4.
usage: python3 render.py [workers] [speed] [out.mp4]   (speed 1 = natural pacing; >1 uniformly faster)"""
import sys, os, subprocess, math
from multiprocessing import Pool
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SPEED = sys.argv[2] if len(sys.argv) > 2 else '1'
MP4 = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, 'HappySprouts_onboarding_landscape.mp4')
OUT = os.path.join(HERE, 'frames_' + SPEED.replace('.', '_')); os.makedirs(OUT, exist_ok=True)
URL = f'file://{HERE}/index.html?render=1&speed={SPEED}'
def work(rng):
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1920, 'height': 1080}); pg.goto(URL); pg.evaluate('window.ready')
        for f in range(*rng):
            pg.evaluate(f'render({f / FPS})'); pg.screenshot(path=f'{OUT}/f{f:05d}.jpg', type='jpeg', quality=94)
        b.close()
if __name__ == '__main__':
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(); pg.goto(URL); T = pg.evaluate('window.TOTAL'); b.close()
    n = int(round(T * FPS)); ch = math.ceil(n / K)
    print('frames', n, 'dur', T, flush=True)
    with Pool(K) as P: P.map(work, [(i * ch, min(n, (i + 1) * ch)) for i in range(K)])
    subprocess.run(['ffmpeg', '-y', '-framerate', str(FPS), '-i', OUT + '/f%05d.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'medium', '-movflags', '+faststart', MP4], check=True, capture_output=True)
    print('done', MP4, flush=True)
