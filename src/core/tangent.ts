/**
 * Tangent-point search (paper §4.2, "Direction refinement"). Dimension-generic.
 *
 * For a sample (x, d), find the unit direction g minimizing
 *     f(g) = D̃(x − d g) / d,
 * i.e. the point y = x − d g on the sample's circle/sphere that reaches farthest past
 * the current zero level set. Dividing by d orients the objective for either sign and
 * gives the Euclidean gradient ∇f = −∇D̃(y). We take projected gradient steps: move
 * along the tangential component of −∇f, then renormalize g onto the unit sphere.
 */

import { evalModel, evalModelGrad, type Interpolant } from './interpolant';

/** Near-uniform unit directions: evenly spaced angles in 2D, a Fibonacci lattice in 3D. */
export function initialDirections(dim: number, count: number): Float64Array {
  const out = new Float64Array(count * dim);
  if (dim === 2) {
    for (let k = 0; k < count; k++) {
      const t = (2 * Math.PI * k) / count;
      out[2 * k] = Math.cos(t);
      out[2 * k + 1] = Math.sin(t);
    }
  } else if (dim === 3) {
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let k = 0; k < count; k++) {
      const z = 1 - (2 * (k + 0.5)) / count;
      const rr = Math.sqrt(1 - z * z);
      out[3 * k] = rr * Math.cos(golden * k);
      out[3 * k + 1] = rr * Math.sin(golden * k);
      out[3 * k + 2] = z;
    }
  } else {
    throw new Error(`initialDirections: unsupported dimension ${dim}`);
  }
  return out;
}

function objective(model: Interpolant, x: Float64Array, d: number, g: Float64Array, y: Float64Array): number {
  for (let c = 0; c < x.length; c++) y[c] = x[c] - d * g[c];
  return evalModel(model, y) / d;
}

/** Best of the lattice directions: argmin sgn(d) D̃(x − d u). */
export function bestInitialDirection(model: Interpolant, x: Float64Array, d: number, dirs: Float64Array): Float64Array {
  const dim = x.length;
  const g = new Float64Array(dim);
  const y = new Float64Array(dim);
  let best = Infinity, bestK = 0;
  for (let k = 0; k < dirs.length / dim; k++) {
    for (let c = 0; c < dim; c++) g[c] = dirs[k * dim + c];
    const f = objective(model, x, d, g, y);
    if (f < best) {
      best = f;
      bestK = k;
    }
  }
  return dirs.slice(bestK * dim, bestK * dim + dim);
}

export interface DescentOptions {
  /** Maximum number of gradient steps. */
  maxIters: number;
  /** Cap on the tangential step length |Δg| before renormalization (0.2 ≈ 11°, 1 ≈ 45°). */
  maxStep: number;
  /** Stop when the tangential gradient norm falls below this. */
  gradTol: number;
}

export interface DescentResult {
  g: Float64Array;
  f: number;
  iterations: number;
  /** Visited points y = x − d g, including the start: (iterations + 1) × dim. */
  path: Float64Array;
}

export function refineDirection(
  model: Interpolant,
  x: Float64Array,
  d: number,
  g0: Float64Array,
  opts: DescentOptions,
): DescentResult {
  const dim = x.length;
  let g = Float64Array.from(g0);
  const y = new Float64Array(dim);
  const grad = new Float64Array(dim);
  const t = new Float64Array(dim);
  const gNew = new Float64Array(dim);
  const yNew = new Float64Array(dim);
  const path: number[] = [];

  for (let c = 0; c < dim; c++) y[c] = x[c] - d * g[c];
  path.push(...y);
  let f = evalModelGrad(model, y, grad) / d;
  let it = 0;

  for (; it < opts.maxIters; it++) {
    // ∇f = −∇D̃(y); tangential part t = ∇f − (∇f·g) g.
    let dot = 0;
    for (let c = 0; c < dim; c++) dot += -grad[c] * g[c];
    let tn = 0;
    for (let c = 0; c < dim; c++) {
      t[c] = -grad[c] - dot * g[c];
      tn += t[c] * t[c];
    }
    tn = Math.sqrt(tn);
    if (tn < opts.gradTol) break;

    // Backtracking (Armijo) line search starting from the capped step.
    let step = Math.min(1, opts.maxStep / tn);
    let accepted = false;
    let fNew = f;
    for (let ls = 0; ls < 12; ls++) {
      let norm = 0;
      for (let c = 0; c < dim; c++) {
        gNew[c] = g[c] - step * t[c];
        norm += gNew[c] * gNew[c];
      }
      norm = Math.sqrt(norm);
      for (let c = 0; c < dim; c++) gNew[c] /= norm;
      fNew = objective(model, x, d, gNew, yNew);
      if (fNew <= f - 1e-4 * step * tn * tn) {
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) break;

    g = Float64Array.from(gNew);
    for (let c = 0; c < dim; c++) y[c] = yNew[c];
    path.push(...y);
    f = evalModelGrad(model, y, grad) / d;
  }
  return { g, f, iterations: it, path: Float64Array.from(path) };
}
