# Status

_Last updated 2026-09-24. Implementation in commit `2b99d1b`; deployment added afterwards._

**Live:** https://yig.github.io/rbf-your-sdf-demo/ (repo: https://github.com/yig/rbf-your-sdf-demo, public).

The 2D demo works end to end. I tested it in headless Chrome across all seven example shapes, grid and scattered sampling, grids from 4² to 32², and phone width. The 16 unit tests for the algorithm code pass, and the production build runs correctly.

To try it locally: `cd demo && npm install && npm run dev`.

## Deployment

- Hosted on GitHub Pages. The local `demo/` repo's `origin` is `github.com:yig/rbf-your-sdf-demo`, branch `main`.
- `.github/workflows/deploy.yml` runs on every push to `main`: tests, build, publish `dist/`. A failing test blocks the deploy.
- The first deploy succeeded, and the live site was checked in headless Chrome: the full pipeline ran (256 samples, 12 fits, 0.34 s) with no errors.
- The public repo includes `CLAUDE.md`, `PLAN.md` and this file, by choice.

## What the page does

It's one long scrolling article with a sticky control bar. The controls pick the shape (or upload your own SVG), grid versus scattered sampling with size and seed, the number of iterations, and whether the power-diagram step runs. There's also an "Advanced" section. Each of the six steps has its own figure you can pan and zoom:

1. **Samples:** red and blue circles; hovering a sample shows its true tangent point.
2. **Power diagram (optional step):** power cells, the exposed arcs, and tangent points fixed where an exposed region collapses. Samples that disappear from the power diagram are drawn with dashed circles.
3. **First RBF:** the cubic RBF fitted to the samples alone, with its zero level set against the ground truth.
4. **Tangent points:** the descent path on each circle, with a play button, a step slider and a choice of iteration. Clicking a sample plots the objective around its circle, with its exposed arcs shaded.
5. **Re-fit:** the new level set compared with the previous one.
6. **Iterate:** a slider and play button over all solves, plus a chart of the error at each stage.

## Results

With the default 16² grid, a full run takes about 0.4 s. The Chamfer distance to the ground truth drops from 0.012 to 0.0009 on the star and from 0.020 to 0.007 on the Eiffel shape. As in the paper's figure, there's a visible improvement at iteration 7, when clamping starts. The partition-of-unity RBF is on by default. It is much faster than a single global RBF at large grids. The figures fill in as each fit finishes.

## Where I filled in details the spec left open

- **Paper's heuristics as defaults.** Defaults follow the paper, including rules that aren't in CLAUDE.md:
  - samples whose circle touches no other circle wait until iteration 2;
  - points landing outside their exposed arcs are dropped;
  - a point that was accepted never gets replaced by one that falls outside;
  - near-misses snap onto tiny exposed regions from iteration 7;
  - near-duplicate points are merged.

  Each rule can be switched off in Advanced. The paper's thresholds assume a unit box, so they are scaled for the demo's [−1, 1]² domain.
- **Descent method.** It's projected gradient descent with a backtracking line search, not the paper's BFGS. The first step is capped at 0.2 and later ones at 1, as in the paper.
- **Covered samples.** A sample that vanishes from the power diagram is clipped against the cell that contains it plus that cell's neighbors, as CLAUDE.md describes. It does occur: 1 of 600 scattered samples on the Eiffel shape.
- **Example shapes.** Eiffel and horse are outlines traced from the PNGs in `yongsalgorithm/examples/`; the paper's own SVGs are rendered figures, not usable outlines. Star, rotated square, box, blob and ring are hand-written.
- **Drawing the field.** Ordinary 2D canvas drawing rather than the WebGL shader in the plan. The shader would compute in 32-bit floats, which lose accuracy when summing the large cubic terms. Instead a second worker evaluates the field in full precision at the current zoom. This is noted in `PLAN.md`.

## For the future standalone app and 3D

- **Reusable code.** The algorithm code in `src/core/` doesn't touch the page, so it can run on your own samples directly; `README.md` has an example.
- **3D.** The RBF, tangent search and overall loop already work in any dimension. The power-diagram step sits behind a small interface, so 3D needs a sphere version of it, a faster linear solver, and a 3D viewer.
