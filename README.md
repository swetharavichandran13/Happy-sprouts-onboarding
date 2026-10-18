# Happy Sprouts – animated onboarding (cover-flow)

A minimal rebrand of the Little Joys onboarding animation project: same nine screens, same cover-flow, same
animations, with the Happy Sprouts wordmark, a refreshed palette and the new tagline.

- `index.html` – deterministic timeline prototype (open in a browser; `?speed=1.3323` changes global pacing).
- `restyle.py` – applies the rebrand to the original exports (pistachio characters, coral accents, subtle brows/sparkle,
  wordmark, tagline, sachet packaging). Usage: `python3 restyle.py <folder-with-original-exports>` → writes `src/`.
- `prep_all.py` – slices `src/` into animation assets (`a/`, `b/`, `data.js`): blink eyelids, gummies, age wheel, carousels, scroll pages.
- `render.py` – renders 1920×1080 / 30 fps MP4: `python3 render.py <workers> <speed> <out.mp4>`.
- `hs_logo.py` – generates the Happy Sprouts wordmark. `fonts/` – Plus Jakarta Sans (OFL) used for the Screen 8 headline.

Palette: watermelon coral `#FF5A68` (CTAs, accents), pistachio (former green characters), apricot (packaging accent),
existing periwinkle blue / yellow characters and warm cream background.

Video: nine screens × 2 loops. Natural pacing = 102.2 s; the delivered cut runs at a uniform 1.332× = 76.7 s.
