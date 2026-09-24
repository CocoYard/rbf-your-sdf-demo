/**
 * One interface over the two interpolants: a single global RBF, or a partition of
 * unity of local RBFs. Everything downstream of a fit (tangent search, metrics,
 * display) goes through these functions.
 */

import type { KernelName } from './kernel';
import { denseLU, type LinearSolver } from './linalg';
import { defaultPUOptions, evalPU, evalPUGrad, evalPUGrid2D, fitPU, type PUModel, type PUOptions } from './pu';
import { evalRBF, evalRBFGrad, evalRBFGrid2D, fitRBF, type RBFModel } from './rbf';

export type Interpolant = RBFModel | PUModel;

export interface InterpolantOptions {
  method: 'global' | 'pu';
  pu: PUOptions;
}

export function defaultInterpolantOptions(): InterpolantOptions {
  return { method: 'global', pu: defaultPUOptions() };
}

export function fitInterpolant(
  dim: number,
  points: Float64Array,
  values: Float64Array,
  kernel: KernelName,
  opts: InterpolantOptions,
  solver: LinearSolver = denseLU,
): Interpolant {
  return opts.method === 'pu'
    ? fitPU(dim, points, values, kernel, opts.pu, solver)
    : fitRBF(dim, points, values, kernel, solver);
}

export function evalModel(model: Interpolant, x: ArrayLike<number>, offset = 0): number {
  return model.kind === 'pu' ? evalPU(model, x, offset) : evalRBF(model, x, offset);
}

export function evalModelGrad(model: Interpolant, x: ArrayLike<number>, grad: Float64Array | number[]): number {
  return model.kind === 'pu' ? evalPUGrad(model, x, grad) : evalRBFGrad(model, x, grad);
}

export function evalModelGrid2D(
  model: Interpolant, x0: number, y0: number, x1: number, y1: number, nx: number, ny: number,
): Float64Array {
  return model.kind === 'pu'
    ? evalPUGrid2D(model, x0, y0, x1, y1, nx, ny)
    : evalRBFGrid2D(model, x0, y0, x1, y1, nx, ny);
}
