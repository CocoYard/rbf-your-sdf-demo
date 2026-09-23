# Interactive 2D demo of "RBF Your SDF" — implementation plan

## Context
`demo/CLAUDE.md` asks for a step-by-step web demo (in the style of https://clementjambon.github.io/wods/) of the paper's algorithm (`../main.tex` §3–4) in 2D:
SVG ground-truth path → SDF samples (circles) → cubic RBF → tangent points by projected descent on each circle → refit with zero-valued tangent points → iterate; plus an optional power-diagram / exposed-arc step that fixes tangent points of samples with tiny exposed arcs.
The code must be modular (future standalone app for user data) and 3D-ready.
**Decisions (confirmed with user):** TypeScript core, vanilla TS + Vite, static build (GitHub Pages-able). The existing C++ research code will not be reused; a separate clean native implementation (packaged as a Python module) is planned. A future 3D web version can plug a WASM build of that native core in behind the same interfaces.

## Layout (`demo/`)
```
package.json, vite.config.ts (base: './'), tsconfig.json, index.html
public/examples/*.svg          # eiffel (from ../misc/eiffel2.svg), star, blob, square
src/core/        # pure TS, NO DOM — reusable by a future standalone app / worker
  types.ts       # Points = {dim, data: Float64Array}; Samples {points, values}
  vec.ts         # dim-generic small-vector helpers
  rng.ts         # seeded PRNG (mulberry32) for reproducible scattering
  linalg.ts      # LinearSolver interface + dense LU w/ partial pivoting (default impl)
  kernel.ts      # Kernel interface {phi, dphi}; cubic r^3 (others pluggable)
  rbf.ts         # fitRBF(points, values, kernel, solver) -> RBF {eval, grad, evalBatch}; dim-generic, degree-1 poly, saddle-point system (eq. rbf-system)
  shape2d.ts     # Polygon (list of closed polylines) -> signed distance (min seg dist, sign by even-odd/winding)
  sampling.ts    # gridSamples(bbox, N) / scatteredSamples(bbox, n, seed) over a shape SDF
  power2d.ts     # power diagram: each cell = bbox ∩ half-planes vs all other sites (O(n²), fine at demo sizes); records neighbor ids per edge; empty cell = "hidden" sample
  arcs2d.ts      # exposed arcs of circle i: clip [0,2π) by each power-cell edge's half-plane (and by neighbor disks); hidden samples: clip against neighbors of the containing cell -> infinitesimal arc; total arc length, short-arc detection (ε_degen, keep midpoint with smallest |D̃0| < ε_val)
  tangent2d.ts   # init: best of 64 angles minimizing sgn(d)·D̃(x − d·u); projected GD on circle for f(g)=D̃(x−d g)/d (grad = −∇D̃(y)), step cap, max iters; returns trajectory for viz; optional feasibility filter (point must lie on exposed arc) + monotone-feasibility rule
  pipeline.ts    # Pipeline state machine: {samples, power?, arcs?, fixedTangents, rbf_k, tangents_k, traces} with pure step functions: computeArcs, fitInitial, projectTangents, refit, iterate(k). Options object holds all thresholds from the paper.
src/io/svg.ts    # DOM: load SVG, flatten paths via SVGPathElement.getTotalLength/getPointAtLength, normalize to [-1,1]² -> Polygon
src/worker.ts    # runs pipeline off main thread; posts plain-data results (transferable Float64Arrays)
src/viz/
  field-gl.ts    # WebGL2 fragment shader: evaluates RBF per pixel (centers+coeffs in a float texture), diverging red/blue colormap + contour lines + zero level set via fwidth
  overlay.ts     # Canvas2D/SVG overlay: GT path, sample circles (red +, blue −), power cells, exposed arcs, tangent points, GD trajectories, fixed (short-arc) points highlighted
  view.ts        # pan/zoom transform shared by field + overlay
src/steps/       # one module per article section (text + figure + controls), scroll-driven like WODS
  s0-shape.ts        # choose example/upload SVG; grid vs scattered, N / n slider, seed
  s1-power.ts        # (optional toggle) power diagram, exposed arcs, short-arc fixed tangent points, hidden samples
  s2-rbf.ts          # initial RBF of samples only; shows level set floating off GT
  s3-tangents.ts     # animate projected GD on each circle; scrub descent iterations
  s4-refit.ts        # refit with tangent points (value 0)
  s5-iterate.ts      # iteration slider / play; per-iteration error vs GT (mean |D̃| on GT path)
src/main.ts, src/style.css
tests/*.test.ts  # vitest
```

## Key design points
- **Dimension genericity:** `core/rbf.ts`, `kernel.ts`, `linalg.ts`, `vec.ts`, `pipeline.ts` take `dim`; only `*2d.ts` files are 2D-specific and sit behind interfaces (`ExposedRegionComputer`, `TangentSearch`) so 3D variants (sphere caps, S² search, PoU RBF) slot in.
- **Traces:** each core step returns result + optional trace data (descent paths, clipped cells) so the viz never re-derives algorithm internals.
- **Solver swap:** `LinearSolver` interface → dense LU now; WASM/PoU later.
- Canonical domain: shape normalized to [-1,1]²; paper thresholds (ε_degen=1e-5, ε_val=0.1, ε_tan=1e-4, dedup 5e-4) exposed in an "advanced" panel.
- Recompute is incremental: changing iteration count doesn't redo sampling; changing sampling invalidates downstream.

## Implementation notes (deviations from the plan above)
- Field display uses Canvas2D, not WebGL: a second worker evaluates the RBF in double precision on a lattice sized to the current view; colormap, iso-contours, and the zero level set (marching squares) are drawn from that. This avoids float32 cancellation in the cubic sum.
- Figures, drawing, and plots live in `src/viz/`; worker plumbing, state, and metrics live in `src/app/`; `src/main.ts` wires one figure per section (no separate `src/steps/`).
- The exposed-region logic is behind `ExposedRegionOracle` (`core/regions.ts`); the 2D implementation is `core/regions2d.ts`.
- Example shapes: Eiffel and horse traced from `yongsalgorithm/examples/*.png`; star, rotated square, box, blob, ring hand-written.

## Verification
- `npm test` (vitest): RBF interpolates its data to ~1e-10 and reproduces linear functions exactly; polygon SDF on a square matches analytic; equal-radius power diagram == Voronoi (compare vs brute-force nearest-site); two-circle exposed arcs match analytic angles; a sample fully inside two others yields an infinitesimal arc; tangent search on an exact circle-shaped SDF converges to analytic closest point; pipeline on a square decreases GT error across iterations.
- `npm run dev` and view in browser: walk all steps on eiffel/star with grid & scattered, toggle the power-diagram step; check level set approaches GT over iterations (compare qualitatively to Fig. `fig:eiffel`).
- `npm run build` produces a static `dist/` that works when opened via `npx vite preview`.
