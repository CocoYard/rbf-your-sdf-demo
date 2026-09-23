/**
 * The full algorithm (paper §4), dimension-generic:
 *
 *   D̃₀ ← RBF(samples)
 *   [optional] fix tangent points of samples whose exposed region collapsed; refit
 *   repeat K times:
 *     y_i ← tangent point of each sample, by projected descent on its sphere against D̃
 *     D̃  ← RBF(samples ∪ {(y_i, 0)})
 *
 * Every fit is recorded as a Stage so a UI can show the algorithm step by step.
 */

import type { KernelName } from './kernel';
import { denseLU, type LinearSolver } from './linalg';
import { evalRBF, fitRBF, type RBFModel } from './rbf';
import type { ExposedRegionOracle } from './regions';
import { bestInitialDirection, initialDirections, refineDirection } from './tangent';
import type { Samples } from './types';

/** Per-sample tangent point status within a stage. */
export const Status = {
  /** No tangent point this stage. */
  None: 0,
  /** |d| ≈ 0: the sample is itself on the surface. */
  OnSurface: 1,
  /** Fixed by a collapsed exposed region (never refined). */
  Fixed: 2,
  /** Found by projected descent and accepted. */
  Projected: 3,
  /** Isolated sphere, initialization deferred to the next iteration. */
  Deferred: 4,
  /** Descent result fell outside the exposed region and was culled. */
  Infeasible: 5,
  /** Update would have become infeasible; the previous feasible point was kept. */
  KeptPrevious: 6,
  /** Nearly feasible on a tiny exposed region; clamped onto it. */
  Clamped: 7,
  /** Within the de-duplication radius of another constraint; dropped. */
  Duplicate: 8,
} as const;
export type StatusCode = (typeof Status)[keyof typeof Status];

/** Statuses whose tangent point is a zero-valued constraint of the stage's fit. */
export function isConstraint(s: number): boolean {
  return s === Status.Fixed || s === Status.Projected || s === Status.KeptPrevious || s === Status.Clamped;
}

export interface PipelineOptions {
  kernel: KernelName;
  /** Number of outer iterations (projection + refit). */
  iterations: number;
  /** Lattice size for the first direction guess (paper: 64). */
  initDirections: number;
  descent: { maxIters: number; firstMaxStep: number; maxStep: number; gradTol: number };
  /** Defer initializing samples whose sphere intersects no other sphere. */
  deferIsolated: boolean;
  /** Cull projections that land outside the exposed region (needs an oracle). */
  filterInfeasible: boolean;
  /** Never let a previously feasible projection become infeasible. */
  monotoneFeasibility: boolean;
  clamp: { enabled: boolean; afterIteration: number; eps: number };
  /** Accept a collapsed-region candidate only if |D̃₀| < epsVal. */
  epsVal: number;
  /** Constraints closer than this to an earlier constraint are dropped. */
  dedupRadius: number;
  /** A projection counts as feasible if within this distance of the exposed region. */
  feasibilityTol: number;
}

/**
 * Defaults following the paper, whose thresholds assume a unit bounding box;
 * `scale` is the size of the domain in the same units (2 for [−1, 1]²).
 */
export function defaultOptions(scale = 1): PipelineOptions {
  return {
    kernel: 'cubic',
    iterations: 10,
    initDirections: 64,
    descent: { maxIters: 10, firstMaxStep: 0.2, maxStep: 1, gradTol: 1e-4 },
    deferIsolated: true,
    filterInfeasible: true,
    monotoneFeasibility: true,
    clamp: { enabled: true, afterIteration: 7, eps: 0.1 * scale },
    epsVal: 0.1 * scale,
    dedupRadius: 5e-4 * scale,
    feasibilityTol: 1e-4 * scale,
  };
}

export interface Stage {
  kind: 'samples' | 'collapsed' | 'projection';
  /** 0 for the initial fits, k for outer iteration k. */
  iteration: number;
  label: string;
  /** Tangent point per sample (NaN when none), n × dim. */
  tangents: Float64Array;
  status: Uint8Array;
  /** The fit of the samples plus the constraint tangent points. */
  model: RBFModel;
  /**
   * Projection stages only: per sample, the descent path (points y visited, starting
   * from the initial guess) taken against the previous stage's model; null if the
   * sample was not refined in this iteration.
   */
  paths?: (Float64Array | null)[];
}

export interface PipelineCallbacks {
  onStage?: (stage: Stage, index: number) => void;
  onProgress?: (message: string) => void;
}

function isolatedSamples(samples: Samples): boolean[] {
  const { dim, points, values } = samples;
  const n = values.length;
  const iso = new Array<boolean>(n).fill(true);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let d2 = 0;
      for (let c = 0; c < dim; c++) {
        const t = points[i * dim + c] - points[j * dim + c];
        d2 += t * t;
      }
      const D = Math.sqrt(d2);
      const ri = Math.abs(values[i]), rj = Math.abs(values[j]);
      // Spheres intersect iff |ri − rj| < D < ri + rj.
      if (D < ri + rj && D > Math.abs(ri - rj)) {
        iso[i] = false;
        iso[j] = false;
      }
    }
  }
  return iso;
}

/**
 * Fit samples plus zero-valued tangent points, dropping near-duplicate constraints.
 * Marks dropped tangent points as Duplicate in `status`.
 */
function fitWithTangents(
  samples: Samples,
  tangents: Float64Array,
  status: Uint8Array,
  opts: PipelineOptions,
  solver: LinearSolver,
): RBFModel {
  const { dim } = samples;
  const n = samples.values.length;
  const cell = Math.max(opts.dedupRadius, 1e-12);
  const grid = new Map<string, number[]>();
  const pts: number[] = [];
  const vals: number[] = [];
  const key = (p: ArrayLike<number>, off: number, shift: number[]) => {
    let s = '';
    for (let c = 0; c < dim; c++) s += Math.floor(p[off + c] / cell) + shift[c] + ',';
    return s;
  };
  const neighborShifts: number[][] = [[]];
  for (let c = 0; c < dim; c++) {
    const next: number[][] = [];
    for (const s of neighborShifts) for (const o of [-1, 0, 1]) next.push([...s, o]);
    neighborShifts.splice(0, neighborShifts.length, ...next);
  }
  const zero = new Array(dim).fill(0);
  const tooClose = (p: ArrayLike<number>, off: number) => {
    for (const sh of neighborShifts) {
      const bucket = grid.get(key(p, off, sh));
      if (!bucket) continue;
      for (const k of bucket) {
        let d2 = 0;
        for (let c = 0; c < dim; c++) {
          const t = p[off + c] - pts[k * dim + c];
          d2 += t * t;
        }
        if (d2 < opts.dedupRadius * opts.dedupRadius) return true;
      }
    }
    return false;
  };
  const add = (p: ArrayLike<number>, off: number, v: number) => {
    const k = vals.length;
    for (let c = 0; c < dim; c++) pts.push(p[off + c]);
    vals.push(v);
    const kk = key(p, off, zero);
    const bucket = grid.get(kk);
    if (bucket) bucket.push(k);
    else grid.set(kk, [k]);
  };

  // Samples are always kept (even if two coincide the solver will complain loudly).
  for (let i = 0; i < n; i++) add(samples.points, i * dim, samples.values[i]);
  // Fixed tangent points take precedence over projected ones.
  for (const pass of [0, 1]) {
    for (let i = 0; i < n; i++) {
      const s = status[i];
      if (!isConstraint(s) || (pass === 0) !== (s === Status.Fixed)) continue;
      if (tooClose(tangents, i * dim)) status[i] = Status.Duplicate;
      else add(tangents, i * dim, 0);
    }
  }
  return fitRBF(dim, Float64Array.from(pts), Float64Array.from(vals), opts.kernel, solver);
}

export function runPipeline(
  samples: Samples,
  opts: PipelineOptions,
  oracle: ExposedRegionOracle | null = null,
  callbacks: PipelineCallbacks = {},
  solver: LinearSolver = denseLU,
): Stage[] {
  const { dim, points, values } = samples;
  const n = values.length;
  const stages: Stage[] = [];
  const push = (s: Stage) => {
    stages.push(s);
    callbacks.onStage?.(s, stages.length - 1);
  };

  const status = new Uint8Array(n);
  const tangents = new Float64Array(n * dim).fill(NaN);
  const onSurfaceEps = 1e-12;
  for (let i = 0; i < n; i++) if (Math.abs(values[i]) < onSurfaceEps) status[i] = Status.OnSurface;

  // Stage 0: samples only.
  callbacks.onProgress?.('Fitting RBF to samples');
  const model0 = fitRBF(dim, points, values, opts.kernel, solver);
  push({
    kind: 'samples', iteration: 0, label: 'Samples only',
    tangents: Float64Array.from(tangents), status: Uint8Array.from(status), model: model0,
  });

  // Collapsed exposed regions: tangent points determined by geometry alone.
  if (oracle) {
    let any = false;
    for (let i = 0; i < n; i++) {
      if (status[i] !== Status.None) continue;
      const cands = oracle.collapsedCandidates(i);
      if (!cands) continue;
      let best = Infinity, bestK = -1;
      for (let k = 0; k < cands.length / dim; k++) {
        const v = Math.abs(evalRBF(model0, cands, k * dim));
        if (v < best) {
          best = v;
          bestK = k;
        }
      }
      if (bestK >= 0 && best < opts.epsVal) {
        for (let c = 0; c < dim; c++) tangents[i * dim + c] = cands[bestK * dim + c];
        status[i] = Status.Fixed;
        any = true;
      }
    }
    if (any) {
      callbacks.onProgress?.('Fitting RBF with collapsed-region tangent points');
      const st = Uint8Array.from(status);
      const model = fitWithTangents(samples, tangents, st, opts, solver);
      push({ kind: 'collapsed', iteration: 0, label: 'Collapsed-region tangent points', tangents: Float64Array.from(tangents), status: st, model });
    }
  }

  const isolated = opts.deferIsolated ? isolatedSamples(samples) : new Array<boolean>(n).fill(false);
  const dirs = initialDirections(dim, opts.initDirections);
  const g: (Float64Array | null)[] = new Array(n).fill(null);
  const refinedBefore = new Array<boolean>(n).fill(false);
  const wasFeasible = new Array<boolean>(n).fill(false);
  const prevY: (Float64Array | null)[] = new Array(n).fill(null);
  const x = new Float64Array(dim);

  for (let k = 1; k <= opts.iterations; k++) {
    callbacks.onProgress?.(`Iteration ${k}: projecting tangent points`);
    const model = stages[stages.length - 1].model;
    const st = new Uint8Array(n);
    const tan = new Float64Array(n * dim).fill(NaN);
    const paths: (Float64Array | null)[] = new Array(n).fill(null);

    for (let i = 0; i < n; i++) {
      if (status[i] === Status.OnSurface || status[i] === Status.Fixed) {
        st[i] = status[i];
        for (let c = 0; c < dim; c++) tan[i * dim + c] = tangents[i * dim + c];
        continue;
      }
      const d = values[i];
      for (let c = 0; c < dim; c++) x[c] = points[i * dim + c];
      if (!g[i]) {
        if (k === 1 && isolated[i]) {
          st[i] = Status.Deferred;
          continue;
        }
        g[i] = bestInitialDirection(model, x, d, dirs);
      }
      const res = refineDirection(model, x, d, g[i]!, {
        maxIters: opts.descent.maxIters,
        maxStep: refinedBefore[i] ? opts.descent.maxStep : opts.descent.firstMaxStep,
        gradTol: opts.descent.gradTol,
      });
      refinedBefore[i] = true;
      paths[i] = res.path;
      const y = new Float64Array(dim);
      for (let c = 0; c < dim; c++) y[c] = x[c] - d * res.g[c];

      let accepted: Float64Array | null = y;
      let code: StatusCode = Status.Projected;
      if (oracle && opts.filterInfeasible) {
        const { distance, closest } = oracle.distance(i, y);
        if (distance <= opts.feasibilityTol) {
          g[i] = res.g;
        } else if (
          opts.clamp.enabled && k >= opts.clamp.afterIteration &&
          distance < opts.clamp.eps && oracle.size(i) < opts.clamp.eps
        ) {
          accepted = closest;
          code = Status.Clamped;
          const gc = new Float64Array(dim);
          for (let c = 0; c < dim; c++) gc[c] = (x[c] - closest[c]) / d;
          g[i] = gc;
        } else if (opts.monotoneFeasibility && wasFeasible[i] && prevY[i]) {
          accepted = prevY[i];
          code = Status.KeptPrevious;
        } else {
          accepted = null;
          code = Status.Infeasible;
          g[i] = res.g; // warm start for the next iteration
        }
      } else {
        g[i] = res.g;
      }

      st[i] = code;
      if (accepted) {
        wasFeasible[i] = true;
        prevY[i] = accepted;
        for (let c = 0; c < dim; c++) tan[i * dim + c] = accepted[c];
      } else if (code === Status.Infeasible) {
        // Show where the descent landed even though it is not used.
        for (let c = 0; c < dim; c++) tan[i * dim + c] = y[c];
      }
    }

    callbacks.onProgress?.(`Iteration ${k}: fitting RBF`);
    const fitStatus = Uint8Array.from(st);
    const newModel = fitWithTangents(samples, tan, fitStatus, opts, solver);
    push({ kind: 'projection', iteration: k, label: `Iteration ${k}`, tangents: tan, status: fitStatus, model: newModel, paths });
  }
  return stages;
}
