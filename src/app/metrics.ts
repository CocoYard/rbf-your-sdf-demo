/** Reconstruction error against a known ground-truth shape. */

import { marchingSquares } from '../core/contour2d';
import { evalModel, evalModelGrid2D, type Interpolant } from '../core/interpolant';
import { closestPointOnShape, sampleBoundary, type Shape2D } from '../core/shape2d';
import type { Box2 } from '../core/types';
import type { StageMetric } from './protocol';

function distanceToSegments(segs: Float64Array, px: number, py: number): number {
  let best = Infinity;
  for (let k = 0; k < segs.length; k += 4) {
    const ax = segs[k], ay = segs[k + 1], ex = segs[k + 2] - ax, ey = segs[k + 3] - ay;
    const len2 = ex * ex + ey * ey;
    let t = len2 > 0 ? ((px - ax) * ex + (py - ay) * ey) / len2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const dx = px - ax - t * ex, dy = py - ay - t * ey;
    best = Math.min(best, dx * dx + dy * dy);
  }
  return Math.sqrt(best);
}

export function stageMetric(model: Interpolant, shape: Shape2D, domain: Box2, boundary?: Float64Array): StageMetric {
  const pts = boundary ?? sampleBoundary(shape, 0.004);
  const m = pts.length / 2;
  let meanAbs = 0;
  for (let k = 0; k < m; k++) meanAbs += Math.abs(evalModel(model, pts, 2 * k));
  meanAbs /= m;

  const res = 160;
  const grid = evalModelGrid2D(model, domain.x0, domain.y0, domain.x1, domain.y1, res, res);
  const segs = marchingSquares(grid, res, res, domain.x0, domain.y0, domain.x1, domain.y1);
  if (segs.length === 0) return { meanAbs, chamfer: NaN };

  // Ground truth → level set (subsampled for speed).
  let a = 0, na = 0;
  for (let k = 0; k < m; k += 4, na++) a += distanceToSegments(segs, pts[2 * k], pts[2 * k + 1]);
  // Level set → ground truth.
  let b = 0, nb = 0;
  for (let k = 0; k < segs.length; k += 4, nb++) b += closestPointOnShape(shape, segs[k], segs[k + 1]).distance;
  return { meanAbs, chamfer: 0.5 * (a / na + b / nb) };
}
