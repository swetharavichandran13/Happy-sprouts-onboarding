# Happy Sprouts – animated onboarding (cover-flow)

Same nine screens, cover-flow, scrolling demos, carousel interactions and animations as the Little Joys project,
now with the approved Happy Sprouts visual system (v2):

- **Characters (F2, open eyes):** every blob character is redrawn as a soft vector with a leafy serrated edge, midrib and
  veins, a simple closed-line cat-smile mouth, blush and open dark eyes with catch-lights (silhouettes, positions and blink
  timing preserved). Colours: teal, coral, apricot (Screens 1/3/9), sage (Screen 6), yellow (Screen 7), coral (Screen 5).
- **Gummies (G2 Garden):** teal · coral · sage · apricot, left to right; shapes, jump animation and timing unchanged.
- **Logo (W3):** Grandstander ExtraBold, "Happy" teal over "Sprouts" coral, teal two-leaf seedling growing from the last
  "s" (`brand/logo_w3.png`), used on the splash and on the sachet packaging (Screen 7).
- **Palette:** teal `#4AB8BD`, sage `#8AB17D`, yellow `#E9C64A`, apricot `#F4A261`, coral `#E76F51` on the warm cream background.

Files
- `index.html` – deterministic timeline prototype (`?render=1&speed=1.3323`).
- `restyle.py` – applies the design to the original exports (`python3 restyle.py <folder-with-original-exports>` → `src/`, eye sprites in `a/`);
  `charsvg.py` – vector character renderer; `gum.py` – G2 gummy/glow recolour.
- `prep_all.py` – slices `src/` into the animation assets (`a/`, `b/`, `data.js`).
- `render.py` – 1920×1080 / 30 fps MP4: `python3 render.py <workers> <speed> <out.mp4>`.
- `fonts/` – Plus Jakarta Sans 700 (Screen 8 headline), Grandstander 800 (logo), both OFL.

Video: nine screens × 2 loops. Natural pacing 102.2 s; the delivered cut uses a uniform 1.332× speed-up = 76.7 s.
