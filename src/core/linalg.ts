/**
 * Linear solvers. The RBF saddle-point system is symmetric but indefinite, so the
 * default solver is dense LU with partial pivoting. The interface exists so a
 * faster backend (e.g. WASM, or a partition-of-unity solver for 3D) can be swapped in.
 */

export interface LinearSolver {
  /** Solve A x = b for a dense row-major n×n matrix. Must not modify its inputs. */
  solve(A: Float64Array, n: number, b: Float64Array): Float64Array;
}

export class SingularMatrixError extends Error {
  constructor(column: number) {
    super(`Matrix is singular to working precision (column ${column}).`);
    this.name = 'SingularMatrixError';
  }
}

export const denseLU: LinearSolver = {
  solve(Ain: Float64Array, n: number, bin: Float64Array): Float64Array {
    const A = Float64Array.from(Ain);
    const b = Float64Array.from(bin);

    let maxAbs = 0;
    for (let i = 0; i < A.length; i++) maxAbs = Math.max(maxAbs, Math.abs(A[i]));
    const tiny = maxAbs * n * 1e-15;

    for (let k = 0; k < n; k++) {
      // Partial pivoting.
      let p = k;
      let best = Math.abs(A[k * n + k]);
      for (let i = k + 1; i < n; i++) {
        const v = Math.abs(A[i * n + k]);
        if (v > best) {
          best = v;
          p = i;
        }
      }
      if (best <= tiny) throw new SingularMatrixError(k);
      if (p !== k) {
        const rk = k * n;
        const rp = p * n;
        for (let j = 0; j < n; j++) {
          const t = A[rk + j];
          A[rk + j] = A[rp + j];
          A[rp + j] = t;
        }
        const t = b[k];
        b[k] = b[p];
        b[p] = t;
      }

      const rk = k * n;
      const pivot = A[rk + k];
      for (let i = k + 1; i < n; i++) {
        const ri = i * n;
        const f = A[ri + k] / pivot;
        if (f === 0) continue;
        A[ri + k] = 0;
        for (let j = k + 1; j < n; j++) A[ri + j] -= f * A[rk + j];
        b[i] -= f * b[k];
      }
    }

    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      const ri = i * n;
      let s = b[i];
      for (let j = i + 1; j < n; j++) s -= A[ri + j] * x[j];
      x[i] = s / A[ri + i];
    }
    return x;
  },
};
