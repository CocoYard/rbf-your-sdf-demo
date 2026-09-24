/**
 * Partition-of-unity RBF (paper supplement, "Partition of unity"). Dimension-generic.
 *
 * The constraints are split by a k-d tree (median split along the longest axis) into
 * leaves of at most `maxLeafPoints`. Each leaf becomes a ball patch centered at the
 * leaf's mean with radius (1 + λ)·(leaf radius); a local RBF is fitted to every
 * constraint inside the ball. Patches holding more than `maxPatchPoints` are split
 * again, and balls contained in a bigger ball are dropped. The local fits are blended
 *
 *   D̃(x) = Σ_j ŵ_j(x) f_j(x) / Σ_k ŵ_k(x),   ŵ_j = Wendland C2 of |x − ξ_j| / R_j.
 *
 * Points outside every support use the nearest patch's local fit directly.
 * Mirrors PUInterpolator(partition='sphere') in the reference sdfgradients/interpolation.py.
 */

import type { KernelName } from './kernel';
import { denseLU, SingularMatrixError, type LinearSolver } from './linalg';
import { evalRBF, evalRBFGrad, fitRBF, type RBFModel } from './rbf';

export interface PUOptions {
  /** λ: each patch's radius is inflated by 1 + λ so neighbors overlap. */
  overlap: number;
  /** k-d tree leaves hold at most this many constraints. */
  maxLeafPoints: number;
  /** A patch whose inflated ball holds more than this is split again. */
  maxPatchPoints: number;
  /** Patches with fewer constraints are dropped (at least dim + 2 is always required). */
  minPatchPoints: number;
}

/** Defaults of the reference PUInterpolator. */
export function defaultPUOptions(): PUOptions {
  return { overlap: 0.25, maxLeafPoints: 200, maxPatchPoints: 675, minPatchPoints: 10 };
}

export interface PUPatch {
  center: Float64Array;
  radius: number;
  /** Local fit to the constraints in the ball. */
  model: RBFModel;
}

export interface PUStats {
  constraints: number;
  patches: number;
  minSize: number;
  meanSize: number;
  maxSize: number;
  /** Patches dropped because they held too few points or their local system was singular. */
  skipped: number;
}

/** A fitted PU interpolant. Plain data, like RBFModel. */
export interface PUModel {
  kind: 'pu';
  dim: number;
  kernel: KernelName;
  patches: PUPatch[];
  stats: PUStats;
}

function dist2(a: ArrayLike<number>, ao: number, b: ArrayLike<number>, bo: number, dim: number): number {
  let s = 0;
  for (let c = 0; c < dim; c++) {
    const t = a[ao + c] - b[bo + c];
    s += t * t;
  }
  return s;
}

/**
 * Split along the longest axis at the median. Falls back to "< median" and then
 * "< mean" when many points share the median coordinate; null if nothing separates.
 */
function medianSplit(points: Float64Array, dim: number, idx: number[]): [number[], number[]] | null {
  let axis = 0, best = -Infinity;
  const mean = new Float64Array(dim);
  for (let c = 0; c < dim; c++) {
    let lo = Infinity, hi = -Infinity;
    for (const i of idx) {
      const v = points[i * dim + c];
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
      mean[c] += v;
    }
    mean[c] /= idx.length;
    if (hi - lo > best) {
      best = hi - lo;
      axis = c;
    }
  }
  const coords = idx.map((i) => points[i * dim + axis]).sort((a, b) => a - b);
  const median = coords[coords.length >> 1];
  const tests: ((v: number) => boolean)[] = [(v) => v <= median, (v) => v < median, (v) => v < mean[axis]];
  for (const test of tests) {
    const left: number[] = [], right: number[] = [];
    for (const i of idx) (test(points[i * dim + axis]) ? left : right).push(i);
    if (left.length && right.length) return [left, right];
  }
  return null;
}

interface PatchInfo {
  center: Float64Array;
  radius: number;
  members: number[];
}

function partition(points: Float64Array, dim: number, n: number, opts: PUOptions): PatchInfo[] {
  const leaves: number[][] = [];
  const subdivide = (idx: number[]) => {
    const split = idx.length > opts.maxLeafPoints ? medianSplit(points, dim, idx) : null;
    if (!split) leaves.push(idx);
    else {
      subdivide(split[0]);
      subdivide(split[1]);
    }
  };
  subdivide(Array.from({ length: n }, (_, i) => i));

  const patches: PatchInfo[] = [];
  const queue = leaves;
  while (queue.length) {
    const leaf = queue.shift()!;
    const center = new Float64Array(dim);
    for (const i of leaf) for (let c = 0; c < dim; c++) center[c] += points[i * dim + c];
    for (let c = 0; c < dim; c++) center[c] /= leaf.length;
    let rCore = 0;
    for (const i of leaf) rCore = Math.max(rCore, Math.sqrt(dist2(points, i * dim, center, 0, dim)));
    const radius = rCore * (1 + opts.overlap);
    const r2 = radius * radius;
    const members: number[] = [];
    for (let i = 0; i < n; i++) if (dist2(points, i * dim, center, 0, dim) <= r2) members.push(i);
    if (members.length > opts.maxPatchPoints && leaf.length > 2) {
      const split = medianSplit(points, dim, leaf);
      if (split) {
        queue.push(split[0], split[1]);
        continue;
      }
    }
    patches.push({ center, radius, members });
  }

  // Drop balls contained in a bigger ball.
  const order = patches.map((_, i) => i).sort((a, b) => patches[b].radius - patches[a].radius);
  const keep = new Array<boolean>(patches.length).fill(true);
  for (let a = 0; a < order.length; a++) {
    const i = order[a];
    if (!keep[i]) continue;
    for (let b = a + 1; b < order.length; b++) {
      const j = order[b];
      if (!keep[j]) continue;
      const d = Math.sqrt(dist2(patches[i].center, 0, patches[j].center, 0, dim));
      if (d <= patches[i].radius - patches[j].radius) keep[j] = false;
    }
  }
  return patches.filter((_, i) => keep[i]);
}

/** Fit a PU interpolant: partition the constraints, then one local RBF per patch. */
export function fitPU(
  dim: number,
  points: Float64Array,
  values: Float64Array,
  kernel: KernelName = 'cubic',
  opts: PUOptions = defaultPUOptions(),
  solver: LinearSolver = denseLU,
): PUModel {
  const n = values.length;
  if (points.length !== n * dim) throw new Error('fitPU: points/values size mismatch');
  const minPoints = Math.max(opts.minPatchPoints, dim + 2);
  const infos = partition(points, dim, n, opts);
  const patches: PUPatch[] = [];
  const sizes: number[] = [];
  let skipped = 0;

  for (const info of infos) {
    if (info.members.length < minPoints || !(info.radius > 0)) {
      skipped++;
      continue;
    }
    const local = info.members;
    const lp = new Float64Array(local.length * dim);
    const lv = new Float64Array(local.length);
    local.forEach((i, k) => {
      for (let c = 0; c < dim; c++) lp[k * dim + c] = points[i * dim + c];
      lv[k] = values[i];
    });
    try {
      patches.push({ center: info.center, radius: info.radius, model: fitRBF(dim, lp, lv, kernel, solver) });
      sizes.push(local.length);
    } catch (e) {
      if (!(e instanceof SingularMatrixError)) throw e;
      skipped++;
    }
  }
  if (patches.length === 0) throw new Error('Partition of unity: no patch could be fitted.');

  return {
    kind: 'pu',
    dim,
    kernel,
    patches,
    stats: {
      constraints: n,
      patches: patches.length,
      minSize: Math.min(...sizes),
      meanSize: sizes.reduce((a, b) => a + b, 0) / sizes.length,
      maxSize: Math.max(...sizes),
      skipped,
    },
  };
}

/** Wendland C2 profile (1 − s)⁴(4s + 1) and its derivative −20 s (1 − s)³. */
function wendland(s: number): number {
  const t = 1 - s;
  return t * t * t * t * (4 * s + 1);
}
function wendlandDeriv(s: number): number {
  const t = 1 - s;
  return -20 * s * t * t * t;
}

function nearestPatch(model: PUModel, x: ArrayLike<number>, offset: number): PUPatch {
  let best = model.patches[0], bestD = Infinity;
  for (const p of model.patches) {
    const d = dist2(x, offset, p.center, 0, model.dim);
    if (d < bestD) {
      bestD = d;
      best = p;
    }
  }
  return best;
}

/** Evaluate the blended field at the point stored at x[offset .. offset+dim). */
export function evalPU(model: PUModel, x: ArrayLike<number>, offset = 0): number {
  let W = 0, V = 0;
  for (const p of model.patches) {
    const r2 = dist2(x, offset, p.center, 0, model.dim);
    if (r2 >= p.radius * p.radius) continue;
    const w = wendland(Math.sqrt(r2) / p.radius);
    W += w;
    V += w * evalRBF(p.model, x, offset);
  }
  return W > 0 ? V / W : evalRBF(nearestPatch(model, x, offset).model, x, offset);
}

/**
 * Evaluate f and write ∇f into `grad`, including the weights' derivatives:
 * ∇f = (Σ ŵ_j ∇f_j + Σ f_j ∇ŵ_j − f Σ ∇ŵ_j) / Σ ŵ_j.
 */
export function evalPUGrad(model: PUModel, x: ArrayLike<number>, grad: Float64Array | number[]): number {
  const dim = model.dim;
  let W = 0, V = 0;
  const G = new Float64Array(dim), A = new Float64Array(dim), B = new Float64Array(dim), g = new Float64Array(dim);
  for (const p of model.patches) {
    const r2 = dist2(x, 0, p.center, 0, dim);
    if (r2 >= p.radius * p.radius) continue;
    const r = Math.sqrt(r2), s = r / p.radius;
    const w = wendland(s);
    const f = evalRBFGrad(p.model, x, g);
    const dwScale = r > 1e-12 ? wendlandDeriv(s) / (p.radius * r) : 0;
    W += w;
    V += w * f;
    for (let c = 0; c < dim; c++) {
      const dw = dwScale * (x[c] - p.center[c]);
      G[c] += w * g[c];
      A[c] += f * dw;
      B[c] += dw;
    }
  }
  if (W <= 0) return evalRBFGrad(nearestPatch(model, x, 0).model, x, grad);
  const f = V / W;
  for (let c = 0; c < dim; c++) grad[c] = (G[c] + A[c] - f * B[c]) / W;
  return f;
}

/**
 * Evaluate a 2D PU model on an nx × ny lattice (row-major, y outer). Each patch only
 * visits the lattice points inside its bounding square.
 */
export function evalPUGrid2D(
  model: PUModel, x0: number, y0: number, x1: number, y1: number, nx: number, ny: number,
): Float64Array {
  if (model.dim !== 2) throw new Error('evalPUGrid2D requires a 2D model');
  const W = new Float64Array(nx * ny);
  const V = new Float64Array(nx * ny);
  const hx = nx === 1 ? 0 : (x1 - x0) / (nx - 1), hy = ny === 1 ? 0 : (y1 - y0) / (ny - 1);
  const coord = (lo: number, h: number, k: number) => lo + h * k;
  const range = (lo: number, h: number, count: number, a: number, b: number): [number, number] =>
    h === 0 ? [0, count - 1] : [Math.max(0, Math.ceil((a - lo) / h)), Math.min(count - 1, Math.floor((b - lo) / h))];
  const q = new Float64Array(2);
  for (const p of model.patches) {
    const [cx, cy] = p.center, R = p.radius;
    const [i0, i1] = range(x0, hx, nx, cx - R, cx + R);
    const [j0, j1] = range(y0, hy, ny, cy - R, cy + R);
    for (let j = j0; j <= j1; j++) {
      q[1] = coord(y0, hy, j);
      for (let i = i0; i <= i1; i++) {
        q[0] = coord(x0, hx, i);
        const r2 = (q[0] - cx) ** 2 + (q[1] - cy) ** 2;
        if (r2 >= R * R) continue;
        const w = wendland(Math.sqrt(r2) / R);
        const k = j * nx + i;
        W[k] += w;
        V[k] += w * evalRBF(p.model, q);
      }
    }
  }
  for (let j = 0; j < ny; j++) {
    q[1] = coord(y0, hy, j);
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (W[k] > 0) V[k] /= W[k];
      else {
        q[0] = coord(x0, hx, i);
        V[k] = evalRBF(nearestPatch(model, q, 0).model, q);
      }
    }
  }
  return V;
}
