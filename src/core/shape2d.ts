/**
 * Ground-truth 2D shape: a set of closed polylines interpreted with the even-odd
 * rule (so holes work). Provides the exact signed distance used to generate samples
 * (negative inside, positive outside, as in the paper).
 */

import type { Box2 } from './types';

export interface Shape2D {
  /** Each loop is a closed polyline [x0,y0, x1,y1, ...] (last vertex ≠ first). */
  loops: Float64Array[];
}

export function shapeBounds(shape: Shape2D): Box2 {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const L of shape.loops) {
    for (let i = 0; i < L.length; i += 2) {
      x0 = Math.min(x0, L[i]);
      x1 = Math.max(x1, L[i]);
      y0 = Math.min(y0, L[i + 1]);
      y1 = Math.max(y1, L[i + 1]);
    }
  }
  return { x0, y0, x1, y1 };
}

/**
 * Uniformly scale and translate so the shape's bounding box is centered and fits in
 * [−extent, extent]²; optionally flip y (SVG's y axis points down).
 */
export function normalizeShape(shape: Shape2D, extent = 0.8, flipY = true): Shape2D {
  const b = shapeBounds(shape);
  const cx = (b.x0 + b.x1) / 2;
  const cy = (b.y0 + b.y1) / 2;
  const s = (2 * extent) / Math.max(b.x1 - b.x0, b.y1 - b.y0, 1e-12);
  return {
    loops: shape.loops.map((L) => {
      const out = new Float64Array(L.length);
      for (let i = 0; i < L.length; i += 2) {
        out[i] = (L[i] - cx) * s;
        out[i + 1] = (flipY ? -1 : 1) * (L[i + 1] - cy) * s;
      }
      return out;
    }),
  };
}

/** Remove consecutive duplicate vertices and loops with fewer than 3 vertices. */
export function cleanShape(shape: Shape2D, eps = 1e-12): Shape2D {
  const loops: Float64Array[] = [];
  for (const L of shape.loops) {
    const pts: number[] = [];
    const m = L.length / 2;
    for (let i = 0; i < m; i++) {
      const x = L[2 * i], y = L[2 * i + 1];
      const k = pts.length;
      if (k >= 2 && Math.hypot(x - pts[k - 2], y - pts[k - 1]) <= eps) continue;
      pts.push(x, y);
    }
    while (pts.length >= 4 && Math.hypot(pts[0] - pts[pts.length - 2], pts[1] - pts[pts.length - 1]) <= eps) {
      pts.length -= 2;
    }
    if (pts.length >= 6) loops.push(Float64Array.from(pts));
  }
  return { loops };
}

export interface ClosestPoint {
  x: number;
  y: number;
  /** Unsigned distance. */
  distance: number;
}

export function closestPointOnShape(shape: Shape2D, px: number, py: number): ClosestPoint {
  let best = Infinity, bx = px, by = py;
  for (const L of shape.loops) {
    const m = L.length / 2;
    for (let i = 0; i < m; i++) {
      const ax = L[2 * i], ay = L[2 * i + 1];
      const j = (i + 1) % m;
      const ex = L[2 * j] - ax, ey = L[2 * j + 1] - ay;
      const len2 = ex * ex + ey * ey;
      let t = len2 > 0 ? ((px - ax) * ex + (py - ay) * ey) / len2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const qx = ax + t * ex, qy = ay + t * ey;
      const d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy);
      if (d2 < best) {
        best = d2;
        bx = qx;
        by = qy;
      }
    }
  }
  return { x: bx, y: by, distance: Math.sqrt(best) };
}

/** Even-odd point-in-shape test. */
export function insideShape(shape: Shape2D, px: number, py: number): boolean {
  let inside = false;
  for (const L of shape.loops) {
    const m = L.length / 2;
    for (let i = 0, j = m - 1; i < m; j = i++) {
      const xi = L[2 * i], yi = L[2 * i + 1];
      const xj = L[2 * j], yj = L[2 * j + 1];
      if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

export function signedDistance(shape: Shape2D, px: number, py: number): number {
  const d = closestPointOnShape(shape, px, py).distance;
  return insideShape(shape, px, py) ? -d : d;
}

export function perimeter(shape: Shape2D): number {
  let total = 0;
  for (const L of shape.loops) {
    const m = L.length / 2;
    for (let i = 0; i < m; i++) {
      const j = (i + 1) % m;
      total += Math.hypot(L[2 * j] - L[2 * i], L[2 * j + 1] - L[2 * i + 1]);
    }
  }
  return total;
}

/** Points spaced (approximately) evenly along the boundary, including every vertex. */
export function sampleBoundary(shape: Shape2D, spacing: number): Float64Array {
  const out: number[] = [];
  for (const L of shape.loops) {
    const m = L.length / 2;
    for (let i = 0; i < m; i++) {
      const j = (i + 1) % m;
      const ax = L[2 * i], ay = L[2 * i + 1];
      const len = Math.hypot(L[2 * j] - ax, L[2 * j + 1] - ay);
      const steps = Math.max(1, Math.ceil(len / spacing));
      for (let s = 0; s < steps; s++) {
        const t = s / steps;
        out.push(ax + t * (L[2 * j] - ax), ay + t * (L[2 * j + 1] - ay));
      }
    }
  }
  return Float64Array.from(out);
}
