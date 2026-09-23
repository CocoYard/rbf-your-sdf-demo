/** Messages exchanged with the pipeline and field workers. All plain data. */

import type { ExposedRegion2D } from '../core/arcs2d';
import type { PipelineOptions, Stage } from '../core/pipeline';
import type { PowerCell } from '../core/power2d';
import type { RBFModel } from '../core/rbf';
import type { SamplingSpec } from '../core/sampling';
import type { Box2, Samples } from '../core/types';

export interface RunRequest {
  type: 'run';
  runId: number;
  loops: Float64Array[];
  domain: Box2;
  sampling: SamplingSpec;
  /** Use the power-diagram / exposed-arc step in the pipeline. */
  useRegions: boolean;
  /** ε_degen: exposed regions shorter than this are "collapsed". */
  epsDegen: number;
  epsTan: number;
  options: PipelineOptions;
}

export interface StageMetric {
  /** Mean |D̃| over points on the ground-truth boundary. */
  meanAbs: number;
  /** Two-sided mean distance between the zero level set and the ground truth. */
  chamfer: number;
}

export type PipelineMessage =
  | {
      type: 'init';
      runId: number;
      samples: Samples;
      cells: (PowerCell | null)[];
      regions: ExposedRegion2D[];
      /** Indices of samples whose exposed region collapsed (before the |D̃₀| test). */
      collapsed: number[];
    }
  | { type: 'progress'; runId: number; message: string }
  | { type: 'stage'; runId: number; index: number; stage: Stage; metric: StageMetric }
  | { type: 'done'; runId: number; ms: number }
  | { type: 'error'; runId: number; message: string };

export interface FieldRequest {
  reqId: number;
  model: RBFModel;
  box: Box2;
  nx: number;
  ny: number;
}

export interface FieldResponse {
  reqId: number;
  values: Float32Array;
}
