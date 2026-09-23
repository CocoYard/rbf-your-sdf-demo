/**
 * Plain-data types shared by the algorithm core. Everything here is
 * structured-clone friendly so it can cross a Web Worker boundary, and
 * dimension-generic (`dim` = 2 for the demo, 3 for a future volumetric version).
 */

/** SDF samples: `points` is a flat array of `values.length * dim` coordinates. */
export interface Samples {
  dim: number;
  points: Float64Array;
  values: Float64Array;
}

export function sampleCount(s: Samples): number {
  return s.values.length;
}

/** Axis-aligned box in 2D. */
export interface Box2 {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}
