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
 * Points outside every support use the local fit(s) of the patch(es) whose spheres are
 * nearest (see `fallback`).
 * Mirrors PUInterpolator(partition='sphere') in the reference sdfgradients/interpolation.py,
 * except for that fallback: the reference evaluates only the patch with the nearest center.
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
  /** Add off-subspace samples to patches whose constraints are (nearly) collinear / coplanar. */
  rankRepair: boolean;
}

/** Defaults of the reference PUInterpolator. */
export function defaultPUOptions(): PUOptions {
  return { overlap: 0.25, maxLeafPoints: 200, maxPatchPoints: 675, minPatchPoints: 10, rankRepair: true };
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
  /** Degenerate patches that received extra samples (rank repair). */
  repaired: number;
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

/** Degenerate if σ_min < RANK_TOL · σ_max (paper: 0.01). */
const RANK_TOL = 1e-2;

/** Eigen-decomposition of a small symmetric matrix (cyclic Jacobi). Ascending eigenvalues; column k of `vectors` is the k-th eigenvector. */
function symmetricEigen(A: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } {
  const a = Float64Array.from(A);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0, diag = 0;
    for (let p = 0; p < n; p++) {
      diag += a[p * n + p] ** 2;
      for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    }
    if (off <= 1e-30 * diag || off === 0) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (apq === 0) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = cs * akp - sn * akq;
          a[k * n + q] = sn * akp + cs * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = cs * apk - sn * aqk;
          a[q * n + k] = sn * apk + cs * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = cs * vkp - sn * vkq;
          v[k * n + q] = sn * vkp + cs * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i * n + i] - a[j * n + j]);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  order.forEach((src, k) => {
    values[k] = a[src * n + src];
    for (let r = 0; r < n; r++) vectors[r * n + k] = v[r * n + src];
  });
  return { values, vectors };
}

/**
 * Rank repair (paper supplement, "Blended interpolant"). If the patch's constraints
 * do not span the space affinely (σ_min < 0.01 σ_max of the centered points), the
 * local system is rank-deficient and the fit is unconstrained off their subspace.
 * Along the weakest direction u, add the sample closest to the patch center on each
 * side among those at least one patch radius from the subspace. Only the local solve
 * changes: the patch's center, radius and weight are untouched. Repeats until the
 * points span the space. Returns true if anything was added.
 */
function repairRank(points: Float64Array, dim: number, n: number, patch: PatchInfo, local: number[]): boolean {
  let repaired = false;
  for (let pass = 0; pass < dim; pass++) {
    const m = local.length;
    if (m === 0) return repaired;
    const mean = new Float64Array(dim);
    for (const i of local) for (let c = 0; c < dim; c++) mean[c] += points[i * dim + c];
    for (let c = 0; c < dim; c++) mean[c] /= m;
    const C = new Float64Array(dim * dim);
    for (const i of local) {
      for (let r = 0; r < dim; r++) {
        const dr = points[i * dim + r] - mean[r];
        for (let c = 0; c < dim; c++) C[r * dim + c] += dr * (points[i * dim + c] - mean[c]);
      }
    }
    // Singular values of the centered point matrix are √(eigenvalues of CᵀC).
    const { values, vectors } = symmetricEigen(C, dim);
    const sMax = Math.sqrt(Math.max(values[dim - 1], 0)), sMin = Math.sqrt(Math.max(values[0], 0));
    if (sMax > 0 && sMin >= RANK_TOL * sMax) return repaired;
    const u = new Float64Array(dim);
    for (let r = 0; r < dim; r++) u[r] = vectors[r * dim];

    const present = new Set(local);
    const best = [-1, -1];
    const bestD = [Infinity, Infinity];
    for (let i = 0; i < n; i++) {
      if (present.has(i)) continue;
      let off = 0;
      for (let c = 0; c < dim; c++) off += (points[i * dim + c] - mean[c]) * u[c];
      if (Math.abs(off) < patch.radius) continue;
      const side = off > 0 ? 1 : 0;
      const d = dist2(points, i * dim, patch.center, 0, dim);
      if (d < bestD[side]) {
        bestD[side] = d;
        best[side] = i;
      }
    }
    if (best[0] < 0 && best[1] < 0) return repaired;
    for (const b of best) if (b >= 0) local.push(b);
    repaired = true;
  }
  return repaired;
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
  let skipped = 0, repaired = 0;

  for (const info of infos) {
    if (info.members.length < minPoints || !(info.radius > 0)) {
      skipped++;
      continue;
    }
    const local = info.members.slice();
    if (opts.rankRepair && repairRank(points, dim, n, info, local)) repaired++;
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
      repaired,
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

/**
 * Outside every support, blend the local fits of this many patches with the nearest spheres.
 * k = 1 is simply the patch with the nearest sphere: continuous where the field leaves the
 * union of supports, but it jumps where the nearest sphere changes. k ≥ 2 removes those jumps.
 */
const FALLBACK_K = 2;

/**
 * The field outside every support: modified Shepard (Franke–Little) blending of the
 * FALLBACK_K patches whose spheres are nearest. With d_j = |x − ξ_j| − R_j and D the
 * (k+1)-th smallest d, w_j = (1/d_j − 1/D)². As d_j → 0 the blend tends to f_j, which
 * matches the PU just inside patch j's support; when patch j drops out of the k nearest,
 * d_j → D and w_j → 0, so for k ≥ 2 the field is continuous everywhere. Writes ∇f into
 * `grad` if given.
 */
function fallback(model: PUModel, x: ArrayLike<number>, offset: number, grad: Float64Array | number[] | null): number {
  const dim = model.dim;
  const P = model.patches;
  const dist = new Float64Array(P.length);
  for (let j = 0; j < P.length; j++) dist[j] = Math.sqrt(dist2(x, offset, P[j].center, 0, dim)) - P[j].radius;
  // Indices of the k + 1 smallest distances, ascending.
  const near: number[] = [];
  for (let j = 0; j < P.length; j++) {
    if (near.length === FALLBACK_K + 1 && dist[j] >= dist[near[FALLBACK_K]]) continue;
    let pos = near.length;
    while (pos > 0 && dist[near[pos - 1]] > dist[j]) pos--;
    near.splice(pos, 0, j);
    if (near.length > FALLBACK_K + 1) near.pop();
  }
  const outer = near.length > FALLBACK_K ? near[FALLBACK_K] : -1;
  const invD = outer >= 0 ? 1 / dist[outer] : 0;
  const used = near.slice(0, FALLBACK_K);

  const g = new Float64Array(dim);
  const unit = (j: number, out: Float64Array) => {
    // ∇d_j = (x − ξ_j) / |x − ξ_j|
    let r = 0;
    for (let c = 0; c < dim; c++) {
      out[c] = x[offset + c] - P[j].center[c];
      r += out[c] * out[c];
    }
    r = Math.sqrt(r) || 1;
    for (let c = 0; c < dim; c++) out[c] /= r;
  };
  // On a sphere (d = 0) the blend is that patch alone.
  if (dist[used[0]] <= 1e-14) {
    const p = P[used[0]].model;
    return grad ? evalRBFGrad(p, x, grad) : evalRBF(p, x, offset);
  }
  const gradD = new Float64Array(dim);
  if (grad && outer >= 0) unit(outer, gradD);

  let W = 0, V = 0;
  const G = new Float64Array(dim), A = new Float64Array(dim), B = new Float64Array(dim), u = new Float64Array(dim);
  for (const j of used) {
    const a = 1 / dist[j] - invD;
    const w = a * a;
    if (!(w > 0)) continue;
    const f = grad ? evalRBFGrad(P[j].model, x, g) : evalRBF(P[j].model, x, offset);
    W += w;
    V += w * f;
    if (grad) {
      unit(j, u);
      // ∇w = 2a (−∇d_j / d_j² + ∇D / D²)
      for (let c = 0; c < dim; c++) {
        const dw = 2 * a * (-u[c] / (dist[j] * dist[j]) + gradD[c] * invD * invD);
        G[c] += w * g[c];
        A[c] += f * dw;
        B[c] += dw;
      }
    }
  }
  if (W <= 0) {
    // All k tie with the (k+1)-th: any of them is the limit.
    const p = P[used[0]].model;
    return grad ? evalRBFGrad(p, x, grad) : evalRBF(p, x, offset);
  }
  const f = V / W;
  if (grad) for (let c = 0; c < dim; c++) grad[c] = (G[c] + A[c] - f * B[c]) / W;
  return f;
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
  return W > 0 ? V / W : fallback(model, x, offset, null);
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
  if (W <= 0) return fallback(model, x, 0, grad);
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
        V[k] = fallback(model, q, 0, null);
      }
    }
  }
  return V;
}
