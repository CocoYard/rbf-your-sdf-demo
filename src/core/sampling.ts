/** Sample positions (grid or scattered) and evaluate a ground-truth SDF at them. */

import { signedDistance, type Shape2D } from './shape2d';
import type { Box2, Samples } from './types';

export type SamplingSpec =
  | { kind: 'grid'; resolution: number }
  | { kind: 'scattered'; count: number; seed: number };

/** Small, fast, seedable PRNG (mulberry32). Returns numbers in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function samplePositions2D(spec: SamplingSpec, box: Box2): Float64Array {
  if (spec.kind === 'grid') {
    const n = Math.max(2, Math.round(spec.resolution));
    const out = new Float64Array(n * n * 2);
    let k = 0;
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        out[k++] = box.x0 + ((box.x1 - box.x0) * i) / (n - 1);
        out[k++] = box.y0 + ((box.y1 - box.y0) * j) / (n - 1);
      }
    }
    return out;
  }
  const rand = mulberry32(spec.seed);
  const out = new Float64Array(spec.count * 2);
  for (let k = 0; k < spec.count; k++) {
    out[2 * k] = box.x0 + (box.x1 - box.x0) * rand();
    out[2 * k + 1] = box.y0 + (box.y1 - box.y0) * rand();
  }
  return out;
}

export function sampleShapeSDF(shape: Shape2D, points: Float64Array): Samples {
  const n = points.length / 2;
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = signedDistance(shape, points[2 * i], points[2 * i + 1]);
  return { dim: 2, points: Float64Array.from(points), values };
}
