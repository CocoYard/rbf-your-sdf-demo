# RBF Your SDF — interactive 2D demo

A step-by-step, in-browser walkthrough of the paper's algorithm in 2D:

1. Sample the signed distance field of a closed curve (from an SVG) on a grid or at scattered points.
2. *(Optional)* Power diagram → exposed arcs → tangent points fixed by collapsed exposed regions.
3. Fit a cubic RBF (φ(r) = r³ plus a linear polynomial) to the samples.
4. Find each sample's tangent point by projected gradient descent on its circle.
5. Re-fit with the tangent points as zero-valued constraints.
6. Iterate 4–5.

**Live demo:** https://yig.github.io/rbf-your-sdf-demo/

## Running

```sh
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests for the algorithm core
npm run build      # static site in dist/ (relative paths; host anywhere, e.g. GitHub Pages)
```

## Deployment

The site is hosted on GitHub Pages from https://github.com/yig/rbf-your-sdf-demo.
Every push to `main` runs `.github/workflows/deploy.yml`, which installs dependencies,
runs the tests, builds, and publishes `dist/`. A failing test stops the deploy, so a
broken build never goes live. To redeploy without a new commit:
`gh workflow run deploy.yml`. Check progress with `gh run list` or the repo's Actions tab.

## Code layout

```
src/core/     Algorithm, pure TypeScript with no DOM. Reusable by a standalone app or a worker.
  rbf.ts        RBF fit/evaluate (dimension-generic), kernels in kernel.ts, solver in linalg.ts
  tangent.ts    Tangent-point search: lattice initialization + projected descent (dimension-generic)
  pipeline.ts   The full algorithm; records every fit as a Stage for visualization
  regions.ts    ExposedRegionOracle interface used by the pipeline
  power2d.ts, arcs2d.ts, regions2d.ts   2D power diagram and exposed arcs (implement the oracle)
  shape2d.ts, sampling.ts, contour2d.ts 2D ground truth, sampling, marching squares
src/io/svg.ts   SVG → polygons (browser only)
src/app/        Web Worker plumbing: pipeline worker, field-evaluation worker, state store, metrics
src/viz/        Canvas figures (pan/zoom), field rendering, drawing helpers, small plots
src/main.ts     Page wiring; prose lives in index.html
public/examples Example shapes (Eiffel and horse silhouettes traced from the paper's 2D test images)
```

### Using the core on your own data

```ts
import { runPipeline, defaultOptions } from './src/core/pipeline';
import { computeRegions2D, regionOracle2D } from './src/core/regions2d';

const samples = { dim: 2, points, values };            // Float64Arrays
const regions = computeRegions2D(samples, domainBox);   // optional
const stages = runPipeline(samples, defaultOptions(domainSize), regionOracle2D(samples, regions, 1e-5 * domainSize));
const model = stages[stages.length - 1].model;          // evaluate with evalRBF(model, x)
```

### Toward 3D

`rbf.ts`, `tangent.ts`, and `pipeline.ts` are dimension-generic. A 3D version needs an
`ExposedRegionOracle` over spheres (regular triangulation + spherical caps), a faster
`LinearSolver` (e.g. partition of unity, or WASM), and a 3D viewer.
