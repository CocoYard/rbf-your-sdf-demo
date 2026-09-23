/**
 * 2D power diagram (the dual of the regular triangulation) of weighted sites
 * (c_i, r_i²). Cell i is { x : |x − c_i|² − r_i² ≤ |x − c_j|² − r_j² ∀ j }.
 *
 * Each cell is computed by clipping a large bounding box against the half-planes of
 * all other sites. That is O(n²) overall, which is simple, robust, and plenty fast
 * for the few thousand samples used in the 2D demo.
 *
 * Useful fact: the boundary of the union of the sample disks is exactly
 * ∪_i (circle_i ∩ cell_i), which is how the exposed arcs are computed (arcs2d.ts).
 */

import type { Box2 } from './types';

export interface PowerCell {
  /** Polygon vertices [x0,y0, x1,y1, ...] in counter-clockwise order. */
  polygon: Float64Array;
  /** edgeSite[k] = site across the edge from vertex k to vertex k+1, or −1 for the bounding box. */
  edgeSite: Int32Array;
}

export interface PowerDiagram {
  /** null when the site is hidden (its cell is empty: the disk is covered by others). */
  cells: (PowerCell | null)[];
  /** Sites sharing a cell edge, per site (empty for hidden sites). */
  neighbors: number[][];
  box: Box2;
}

/** Half-plane a·x ≤ b such that the power distance to site i ≤ power distance to site j. */
export function powerHalfPlane(
  cx: number, cy: number, r: number,
  qx: number, qy: number, q: number,
): { ax: number; ay: number; b: number } {
  return {
    ax: 2 * (qx - cx),
    ay: 2 * (qy - cy),
    b: qx * qx + qy * qy - cx * cx - cy * cy - q * q + r * r,
  };
}

function clip(
  xs: number[], ys: number[], labels: number[],
  ax: number, ay: number, b: number, site: number,
): [number[], number[], number[]] {
  const m = xs.length;
  const scale = Math.abs(ax) + Math.abs(ay) + Math.abs(b) + 1e-300;
  const eps = 1e-14 * scale;
  const s = new Array<number>(m);
  let anyOut = false;
  for (let k = 0; k < m; k++) {
    s[k] = ax * xs[k] + ay * ys[k] - b;
    if (s[k] > eps) anyOut = true;
  }
  if (!anyOut) return [xs, ys, labels];

  const ox: number[] = [], oy: number[] = [], ol: number[] = [];
  for (let k = 0; k < m; k++) {
    const k1 = (k + 1) % m;
    const inP = s[k] <= eps;
    const inQ = s[k1] <= eps;
    if (inP) {
      ox.push(xs[k]);
      oy.push(ys[k]);
      ol.push(labels[k]);
      if (!inQ) {
        const t = s[k] / (s[k] - s[k1]);
        ox.push(xs[k] + t * (xs[k1] - xs[k]));
        oy.push(ys[k] + t * (ys[k1] - ys[k]));
        ol.push(site);
      }
    } else if (inQ) {
      const t = s[k] / (s[k] - s[k1]);
      ox.push(xs[k] + t * (xs[k1] - xs[k]));
      oy.push(ys[k] + t * (ys[k1] - ys[k]));
      ol.push(labels[k]);
    }
  }
  return [ox, oy, ol];
}

/**
 * @param points n × 2 site positions
 * @param radii  n radii (weights are radii²)
 * @param box    clipping box; should be much larger than the region of interest
 */
export function powerDiagram2D(points: Float64Array, radii: Float64Array, box: Box2): PowerDiagram {
  const n = radii.length;
  const cells: (PowerCell | null)[] = new Array(n).fill(null);
  const neighbors: number[][] = [];

  for (let i = 0; i < n; i++) {
    const cx = points[2 * i], cy = points[2 * i + 1], r = radii[i];
    let xs = [box.x0, box.x1, box.x1, box.x0];
    let ys = [box.y0, box.y0, box.y1, box.y1];
    let labels = [-1, -1, -1, -1];
    let empty = false;

    for (let j = 0; j < n && !empty; j++) {
      if (j === i) continue;
      const qx = points[2 * j], qy = points[2 * j + 1], q = radii[j];
      if (qx === cx && qy === cy) {
        // Coincident sites: the larger disk owns the whole cell; ties keep both.
        if (q > r) empty = true;
        continue;
      }
      const h = powerHalfPlane(cx, cy, r, qx, qy, q);
      [xs, ys, labels] = clip(xs, ys, labels, h.ax, h.ay, h.b, j);
      if (xs.length < 3) empty = true;
    }

    if (empty) {
      neighbors.push([]);
      continue;
    }
    const polygon = new Float64Array(xs.length * 2);
    for (let k = 0; k < xs.length; k++) {
      polygon[2 * k] = xs[k];
      polygon[2 * k + 1] = ys[k];
    }
    cells[i] = { polygon, edgeSite: Int32Array.from(labels) };
    neighbors.push([...new Set(labels.filter((l) => l >= 0))]);
  }
  return { cells, neighbors, box };
}

/** Index of the (visible) cell containing point p, i.e. the site of minimum power distance. */
export function powerCellContaining(
  points: Float64Array, radii: Float64Array, px: number, py: number, exclude = -1,
  isVisible?: (i: number) => boolean,
): number {
  let best = -1, bestPow = Infinity;
  for (let j = 0; j < radii.length; j++) {
    if (j === exclude || (isVisible && !isVisible(j))) continue;
    const dx = px - points[2 * j], dy = py - points[2 * j + 1];
    const pw = dx * dx + dy * dy - radii[j] * radii[j];
    if (pw < bestPow) {
      bestPow = pw;
      best = j;
    }
  }
  return best;
}
