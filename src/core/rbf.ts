/**
 * Radial basis function interpolation (paper Eq. rbf / rbf-system):
 *
 *   f(x) = Σ_i α_i φ(|x − x_i|) + c·x + c_0
 *
 * with moment conditions Σ α_i = 0, Σ α_i x_i = 0. Dimension-generic.
 */

import { kernels, type KernelName } from './kernel';
import { denseLU, type LinearSolver } from './linalg';

/** A fitted RBF. Plain data: safe to post between threads and to serialize. */
export interface RBFModel {
  kind: 'rbf';
  dim: number;
  kernel: KernelName;
  /** n × dim, flat. */
  centers: Float64Array;
  /** n kernel weights. */
  alpha: Float64Array;
  /** dim linear coefficients followed by the constant term. */
  poly: Float64Array;
}

export function fitRBF(
  dim: number,
  centers: Float64Array,
  values: Float64Array,
  kernel: KernelName = 'cubic',
  solver: LinearSolver = denseLU,
): RBFModel {
  const n = values.length;
  if (centers.length !== n * dim) throw new Error('fitRBF: centers/values size mismatch');
  const k = kernels[kernel];
  const m = n + dim + 1;
  const A = new Float64Array(m * m);
  const b = new Float64Array(m);

  for (let i = 0; i < n; i++) {
    const ri = i * m;
    for (let j = 0; j <= i; j++) {
      let r2 = 0;
      for (let c = 0; c < dim; c++) {
        const t = centers[i * dim + c] - centers[j * dim + c];
        r2 += t * t;
      }
      const v = k.phi(Math.sqrt(r2));
      A[ri + j] = v;
      A[j * m + i] = v;
    }
    // Polynomial block P (row i = [x_i, 1]) and its transpose.
    for (let c = 0; c < dim; c++) {
      const x = centers[i * dim + c];
      A[ri + n + c] = x;
      A[(n + c) * m + i] = x;
    }
    A[ri + n + dim] = 1;
    A[(n + dim) * m + i] = 1;
    b[i] = values[i];
  }

  const sol = solver.solve(A, m, b);
  return {
    kind: 'rbf',
    dim,
    kernel,
    centers: Float64Array.from(centers),
    alpha: sol.slice(0, n),
    poly: sol.slice(n, m),
  };
}

/** Evaluate f at the point stored at x[offset .. offset+dim). */
export function evalRBF(model: RBFModel, x: ArrayLike<number>, offset = 0): number {
  const { dim, centers, alpha, poly } = model;
  const phi = kernels[model.kernel].phi;
  const n = alpha.length;
  let f = poly[dim];
  for (let c = 0; c < dim; c++) f += poly[c] * x[offset + c];
  if (dim === 2) {
    const px = x[offset];
    const py = x[offset + 1];
    for (let i = 0; i < n; i++) {
      const dx = px - centers[2 * i];
      const dy = py - centers[2 * i + 1];
      f += alpha[i] * phi(Math.sqrt(dx * dx + dy * dy));
    }
    return f;
  }
  for (let i = 0; i < n; i++) {
    let r2 = 0;
    for (let c = 0; c < dim; c++) {
      const t = x[offset + c] - centers[i * dim + c];
      r2 += t * t;
    }
    f += alpha[i] * phi(Math.sqrt(r2));
  }
  return f;
}

/** Evaluate f and write ∇f into `grad` (length ≥ dim). Returns f. */
export function evalRBFGrad(model: RBFModel, x: ArrayLike<number>, grad: Float64Array | number[]): number {
  const { dim, centers, alpha, poly } = model;
  const k = kernels[model.kernel];
  const n = alpha.length;
  let f = poly[dim];
  for (let c = 0; c < dim; c++) {
    f += poly[c] * x[c];
    grad[c] = poly[c];
  }
  const diff = new Float64Array(dim);
  for (let i = 0; i < n; i++) {
    let r2 = 0;
    for (let c = 0; c < dim; c++) {
      const t = x[c] - centers[i * dim + c];
      diff[c] = t;
      r2 += t * t;
    }
    const r = Math.sqrt(r2);
    f += alpha[i] * k.phi(r);
    const s = alpha[i] * k.dphiOverR(r);
    for (let c = 0; c < dim; c++) grad[c] += s * diff[c];
  }
  return f;
}

/** Evaluate a 2D model on an nx × ny lattice spanning [x0,x1]×[y0,y1] (row-major, y outer). */
export function evalRBFGrid2D(
  model: RBFModel,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  nx: number,
  ny: number,
): Float64Array {
  if (model.dim !== 2) throw new Error('evalRBFGrid2D requires a 2D model');
  const out = new Float64Array(nx * ny);
  const p = new Float64Array(2);
  for (let j = 0; j < ny; j++) {
    p[1] = ny === 1 ? y0 : y0 + ((y1 - y0) * j) / (ny - 1);
    for (let i = 0; i < nx; i++) {
      p[0] = nx === 1 ? x0 : x0 + ((x1 - x0) * i) / (nx - 1);
      out[j * nx + i] = evalRBF(model, p);
    }
  }
  return out;
}
