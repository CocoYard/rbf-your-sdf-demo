/**
 * Exposed arcs of 2D sample circles (paper §4.1, "Small exposed regions").
 *
 * The exposed region of circle i is the part not inside any other disk. A point p on
 * circle i is inside disk j iff pow_j(p) < pow_i(p) = 0, i.e. iff it is on the far side
 * of the radical axis of i and j. So clipping circle i by neighbor j is a half-plane
 * test on the circle: with n = (c_j − c_i)/D and h = (D² + r_i² − r_j²)/(2D) (Eq.
 * cap-params), the kept angles θ satisfy cos(θ − φ_n) ≤ h / r_i.
 *
 * Near-degenerate configurations (tangencies, collinear samples) must not be lost to
 * round-off, so every clip is relaxed by an angular tolerance ε_tan. After all clips,
 * each relaxed interval is shrunk back by ε_tan at both ends; an interval that becomes
 * negative is emitted as a zero-length arc at its midpoint (paper: "Tangent points at
 * near degeneracies").
 */

import { powerCellContaining, type PowerDiagram } from './power2d';

const TWO_PI = 2 * Math.PI;

/** An arc [start, end] of angles (radians) on a sample circle; end ≥ start, end may exceed 2π. */
export interface Arc {
  start: number;
  end: number;
}

export interface ExposedRegion2D {
  arcs: Arc[];
  /** Σ r · (end − start): the perimeter of the exposed region. */
  length: number;
  /** True if the circle is entirely exposed (no neighbor clips it). */
  full: boolean;
  /** Sample has an empty power cell (fully covered up to tangencies). */
  hidden: boolean;
  /** Neighbor samples that were used to clip this circle. */
  clippers: number[];
}

type Interval = [number, number];

/** Intersect a sorted list of disjoint intervals in [0, 2π] with the circular arc [a0, a1]. */
function intersectCircular(set: Interval[], a0: number, a1: number): Interval[] {
  if (a1 - a0 >= TWO_PI) return set;
  let s = a0 % TWO_PI;
  if (s < 0) s += TWO_PI;
  const e = s + (a1 - a0);
  const parts: Interval[] = e <= TWO_PI ? [[s, e]] : [[s, TWO_PI], [0, e - TWO_PI]];
  const out: Interval[] = [];
  for (const [p0, p1] of parts) {
    for (const [q0, q1] of set) {
      const lo = Math.max(p0, q0), hi = Math.min(p1, q1);
      if (hi >= lo) out.push([lo, hi]);
    }
  }
  out.sort((u, v) => u[0] - v[0]);
  return out;
}

/**
 * Compute the exposed arcs of circle i given the set of neighbors that may cover it.
 * @param tol angular tolerance ε_tan (radians)
 */
export function exposedArcs(
  i: number,
  points: Float64Array,
  radii: Float64Array,
  clippers: Iterable<number>,
  tol = 1e-4,
): Omit<ExposedRegion2D, 'hidden'> {
  const cx = points[2 * i], cy = points[2 * i + 1], r = radii[i];
  // A sample on the surface: its "circle" is a single exposed point.
  if (!(r > 1e-12)) return { arcs: [{ start: 0, end: 0 }], length: 0, full: false, clippers: [] };
  const used: number[] = [];
  let set: Interval[] = [[0, TWO_PI]];
  let clipped = false;
  // cos(θ) ≤ k with k slightly below −1 still admits a tangency within tolerance.
  const kSlack = (tol * tol) / 2;

  for (const j of clippers) {
    if (j === i) continue;
    const dx = points[2 * j] - cx, dy = points[2 * j + 1] - cy, q = radii[j];
    const D = Math.hypot(dx, dy);
    used.push(j);
    if (D === 0) {
      if (q > r) {
        set = [];
        clipped = true;
      }
      continue;
    }
    const k = (D * D + r * r - q * q) / (2 * D * r);
    if (k >= 1) continue; // disk j does not reach circle i
    if (k < -1 - kSlack) {
      set = []; // circle i lies strictly inside disk j
      clipped = true;
      break;
    }
    const phi = Math.atan2(dy, dx);
    const a = Math.acos(Math.max(-1, k));
    set = intersectCircular(set, phi + a - tol, phi + TWO_PI - a + tol);
    clipped = true;
    if (set.length === 0) break;
  }

  if (!clipped) {
    return { arcs: [{ start: 0, end: TWO_PI }], length: TWO_PI * r, full: true, clippers: used };
  }

  // Join the interval that ends at 2π with the one that starts at 0 (the seam is artificial).
  if (set.length >= 2 && set[0][0] <= 0 && set[set.length - 1][1] >= TWO_PI) {
    const first = set.shift()!;
    const last = set.pop()!;
    set.push([last[0], first[1] + TWO_PI]);
  } else if (set.length === 1 && set[0][0] <= 0 && set[0][1] >= TWO_PI) {
    // Relaxed clips that together cover everything: treat as full circle.
    return { arcs: [{ start: 0, end: TWO_PI }], length: TWO_PI * r, full: true, clippers: used };
  }

  const arcs: Arc[] = [];
  let length = 0;
  for (const [s, e] of set) {
    if (e - s <= 2 * tol) {
      const mid = (s + e) / 2;
      arcs.push({ start: mid, end: mid });
    } else {
      arcs.push({ start: s + tol, end: e - tol });
      length += r * (e - s - 2 * tol);
    }
  }
  return { arcs, length, full: false, clippers: used };
}

/**
 * Exposed regions of every sample, using the power diagram to find the relevant
 * neighbors: the cell's edge neighbors for visible samples, and — for hidden samples,
 * which are absent from the diagram — the cell containing the sample plus that
 * cell's neighbors.
 */
export function exposedRegions2D(
  points: Float64Array,
  radii: Float64Array,
  pd: PowerDiagram,
  tol = 1e-4,
): ExposedRegion2D[] {
  const n = radii.length;
  const out: ExposedRegion2D[] = [];
  for (let i = 0; i < n; i++) {
    const hidden = pd.cells[i] === null;
    let clippers: number[];
    if (!hidden) {
      clippers = pd.neighbors[i];
    } else {
      const host = powerCellContaining(points, radii, points[2 * i], points[2 * i + 1], i, (j) => pd.cells[j] !== null);
      clippers = host < 0 ? [] : [host, ...pd.neighbors[host].filter((j) => j !== i)];
    }
    out.push({ ...exposedArcs(i, points, radii, clippers, tol), hidden });
  }
  return out;
}

export function arcPoint(cx: number, cy: number, r: number, theta: number): [number, number] {
  return [cx + r * Math.cos(theta), cy + r * Math.sin(theta)];
}

/**
 * Distance (along the chord) from the point at angle θ on circle (c, r) to the exposed
 * region, and the closest angle in it. Zero if θ lies within an arc.
 */
export function distanceToArcs(arcs: Arc[], r: number, theta: number): { distance: number; closest: number } {
  let best = Infinity, closest = theta;
  for (const a of arcs) {
    // Bring θ into [start, start + 2π).
    let t = (theta - a.start) % TWO_PI;
    if (t < 0) t += TWO_PI;
    if (t <= a.end - a.start) return { distance: 0, closest: theta };
    // Angular distance to each endpoint.
    const dEnd = t - (a.end - a.start);
    const dStart = TWO_PI - t;
    const [d, c] = dEnd < dStart ? [dEnd, a.end] : [dStart, a.start];
    const chord = 2 * r * Math.sin(Math.min(d, Math.PI) / 2);
    if (chord < best) {
      best = chord;
      closest = c;
    }
  }
  return { distance: best, closest };
}
