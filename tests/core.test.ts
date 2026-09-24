import { describe, expect, it } from 'vitest';
import { distanceToArcs, exposedArcs } from '../src/core/arcs2d';
import { marchingSquares } from '../src/core/contour2d';
import { evalModel, type Interpolant } from '../src/core/interpolant';
import { denseLU } from '../src/core/linalg';
import { Status, defaultOptions, isConstraint, runPipeline } from '../src/core/pipeline';
import { powerDiagram2D } from '../src/core/power2d';
import { defaultPUOptions, evalPU, evalPUGrad, evalPUGrid2D, fitPU } from '../src/core/pu';
import { evalRBF, evalRBFGrad, fitRBF } from '../src/core/rbf';
import { computeRegions2D, regionOracle2D } from '../src/core/regions2d';
import { samplePositions2D, sampleShapeSDF } from '../src/core/sampling';
import { closestPointOnShape, signedDistance, type Shape2D } from '../src/core/shape2d';
import { refineDirection } from '../src/core/tangent';

const square: Shape2D = { loops: [Float64Array.from([-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5])] };
const domain = { x0: -1, y0: -1, x1: 1, y1: 1 };

function circleShape(r: number, segs = 512): Shape2D {
  const L = new Float64Array(segs * 2);
  for (let k = 0; k < segs; k++) {
    L[2 * k] = r * Math.cos((2 * Math.PI * k) / segs);
    L[2 * k + 1] = r * Math.sin((2 * Math.PI * k) / segs);
  }
  return { loops: [L] };
}

describe('linalg', () => {
  it('solves a pivoting-requiring system', () => {
    const A = Float64Array.from([0, 1, 1, 1, 0, 1, 1, 1, 0]);
    const x = denseLU.solve(A, 3, Float64Array.from([2, 2, 2]));
    for (const v of x) expect(v).toBeCloseTo(1, 12);
  });
});

describe('shape2d', () => {
  it('signed distance of a square', () => {
    expect(signedDistance(square, 0, 0)).toBeCloseTo(-0.5, 12);
    expect(signedDistance(square, 1, 0)).toBeCloseTo(0.5, 12);
    expect(signedDistance(square, 1, 1)).toBeCloseTo(Math.SQRT2 / 2, 12);
    expect(closestPointOnShape(square, 0.7, 0.1).x).toBeCloseTo(0.5, 12);
  });
  it('even-odd holes', () => {
    const ring: Shape2D = { loops: [...circleShape(0.8).loops, ...circleShape(0.3).loops] };
    expect(signedDistance(ring, 0, 0)).toBeGreaterThan(0);
    expect(signedDistance(ring, 0.5, 0)).toBeLessThan(0);
  });
});

describe('rbf', () => {
  const pts = samplePositions2D({ kind: 'scattered', count: 40, seed: 3 }, domain);
  it('interpolates its data', () => {
    const vals = Float64Array.from({ length: 40 }, (_, i) => Math.sin(3 * pts[2 * i]) + pts[2 * i + 1] ** 2);
    const m = fitRBF(2, pts, vals);
    for (let i = 0; i < 40; i++) expect(evalRBF(m, pts, 2 * i)).toBeCloseTo(vals[i], 9);
  });
  it('reproduces linear functions exactly', () => {
    const vals = Float64Array.from({ length: 40 }, (_, i) => 2 * pts[2 * i] - pts[2 * i + 1] + 0.3);
    const m = fitRBF(2, pts, vals);
    for (const a of m.alpha) expect(Math.abs(a)).toBeLessThan(1e-8);
    expect(evalRBF(m, [0.123, -0.77])).toBeCloseTo(2 * 0.123 + 0.77 + 0.3, 9);
  });
  it('gradient matches finite differences', () => {
    const vals = Float64Array.from({ length: 40 }, (_, i) => Math.cos(2 * pts[2 * i + 1]));
    const m = fitRBF(2, pts, vals);
    const g = new Float64Array(2);
    const p = [0.21, -0.34];
    evalRBFGrad(m, p, g);
    const h = 1e-6;
    expect(g[0]).toBeCloseTo((evalRBF(m, [p[0] + h, p[1]]) - evalRBF(m, [p[0] - h, p[1]])) / (2 * h), 5);
    expect(g[1]).toBeCloseTo((evalRBF(m, [p[0], p[1] + h]) - evalRBF(m, [p[0], p[1] - h])) / (2 * h), 5);
  });
});

describe('power diagram', () => {
  it('equal radii give the Voronoi diagram', () => {
    const pts = samplePositions2D({ kind: 'scattered', count: 30, seed: 7 }, domain);
    const radii = new Float64Array(30).fill(0.1);
    const pd = powerDiagram2D(pts, radii, { x0: -5, y0: -5, x1: 5, y1: 5 });
    // Each cell's centroid must be nearest to its own site.
    pd.cells.forEach((cell, i) => {
      expect(cell).not.toBeNull();
      const P = cell!.polygon;
      let cx = 0, cy = 0;
      for (let k = 0; k < P.length; k += 2) { cx += P[k]; cy += P[k + 1]; }
      cx /= P.length / 2; cy /= P.length / 2;
      let best = -1, bd = Infinity;
      for (let j = 0; j < 30; j++) {
        const d = Math.hypot(cx - pts[2 * j], cy - pts[2 * j + 1]);
        if (d < bd) { bd = d; best = j; }
      }
      expect(best).toBe(i);
    });
  });
  it('a small disk surrounded by large disks is hidden', () => {
    const pts = Float64Array.from([0, 0, 1, 0, -1, 0, 0, 1, 0, -1]);
    const pd = powerDiagram2D(pts, Float64Array.from([0.1, 1.2, 1.2, 1.2, 1.2]), { x0: -5, y0: -5, x1: 5, y1: 5 });
    expect(pd.cells[0]).toBeNull();
    for (let i = 1; i < 5; i++) expect(pd.cells[i]).not.toBeNull();
  });
});

describe('exposed arcs', () => {
  it('two unit circles at distance 1', () => {
    const pts = Float64Array.from([0, 0, 1, 0]);
    const radii = Float64Array.from([1, 1]);
    const reg = exposedArcs(0, pts, radii, [1], 1e-9);
    expect(reg.arcs.length).toBe(1);
    // Circle 0 is covered for |θ| < 60°.
    expect(reg.arcs[0].start).toBeCloseTo(Math.PI / 3, 6);
    expect(reg.arcs[0].end).toBeCloseTo(2 * Math.PI - Math.PI / 3, 6);
    expect(reg.length).toBeCloseTo((2 * Math.PI * 2) / 3, 6);
  });
  it('inner tangency leaves a single zero-length arc', () => {
    // Circle 1 (r = 0.5 at x = 0.5) touches circle 0 (r = 1) internally at (1, 0).
    const pts = Float64Array.from([0, 0, 0.5, 0]);
    const radii = Float64Array.from([1, 0.5]);
    const reg = exposedArcs(1, pts, radii, [0]);
    expect(reg.arcs.length).toBe(1);
    expect(reg.length).toBe(0);
    expect(Math.cos(reg.arcs[0].start)).toBeCloseTo(1, 6);
  });
  it('distance to arcs', () => {
    const arcs = [{ start: 0, end: Math.PI / 2 }];
    expect(distanceToArcs(arcs, 1, 0.3).distance).toBe(0);
    expect(distanceToArcs(arcs, 1, Math.PI).closest).toBeCloseTo(Math.PI / 2, 12);
    expect(distanceToArcs(arcs, 1, -0.1).closest).toBeCloseTo(0, 12);
  });
  it('grid samples of a box: corner samples collapse', () => {
    const s = sampleShapeSDF(square, samplePositions2D({ kind: 'grid', resolution: 9 }, domain));
    const r = computeRegions2D(s, domain);
    const oracle = regionOracle2D(s, r, 2e-5);
    let collapsed = 0;
    for (let i = 0; i < 81; i++) if (oracle.collapsedCandidates(i)) collapsed++;
    expect(collapsed).toBeGreaterThan(0);
    // Every region is non-empty (every circle touches the surface somewhere).
    for (const reg of r.regions) expect(reg.arcs.length).toBeGreaterThan(0);
  });
});

describe('tangent search', () => {
  it('converges to the true closest point for an exact circle SDF', () => {
    // Fit to a dense set of exact SDF values of a circle of radius 0.5.
    const pts = samplePositions2D({ kind: 'grid', resolution: 15 }, domain);
    const vals = Float64Array.from({ length: pts.length / 2 }, (_, i) => Math.hypot(pts[2 * i], pts[2 * i + 1]) - 0.5);
    const m = fitRBF(2, pts, vals);
    const x = Float64Array.from([0.8, 0.3]);
    const d = Math.hypot(0.8, 0.3) - 0.5;
    const res = refineDirection(m, x, d, Float64Array.from([0, 1]), { maxIters: 100, maxStep: 1, gradTol: 1e-8 });
    const expected = Math.atan2(0.3, 0.8);
    expect(Math.atan2(res.g[1], res.g[0])).toBeCloseTo(expected, 2);
  });
});

describe('contouring', () => {
  it('extracts a circle', () => {
    const n = 64;
    const v = new Float64Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = -1 + (2 * i) / (n - 1), y = -1 + (2 * j) / (n - 1);
      v[j * n + i] = Math.hypot(x, y) - 0.5;
    }
    const segs = marchingSquares(v, n, n, -1, -1, 1, 1);
    expect(segs.length).toBeGreaterThan(0);
    for (let k = 0; k < segs.length; k += 2) expect(Math.hypot(segs[k], segs[k + 1])).toBeCloseTo(0.5, 2);
  });
});

describe('pipeline', () => {
  function levelSetError(model: Interpolant, shape: Shape2D) {
    // Mean |D̃| along the ground-truth boundary.
    let total = 0, count = 0;
    for (const L of shape.loops) for (let k = 0; k < L.length; k += 2) {
      const j = (k + 2) % L.length;
      for (let t = 0; t < 1; t += 0.1) {
        total += Math.abs(evalModel(model, [L[k] + t * (L[j] - L[k]), L[k + 1] + t * (L[j + 1] - L[k + 1])]));
        count++;
      }
    }
    return total / count;
  }

  for (const [useRegions, method] of [[false, 'global'], [true, 'global'], [true, 'pu']] as const) {
    it(`reduces error on a rotated square (regions: ${useRegions}, ${method})`, () => {
      const c = Math.cos(0.35), s = Math.sin(0.35);
      const L = square.loops[0].map((_, k, a) => (k % 2 === 0 ? c * a[k] - s * a[k + 1] : s * a[k - 1] + c * a[k]));
      const shape: Shape2D = { loops: [Float64Array.from(L)] };
      const samples = sampleShapeSDF(shape, samplePositions2D({ kind: 'grid', resolution: 12 }, domain));
      const opts = { ...defaultOptions(2), iterations: 6 };
      opts.interpolant = { method, pu: { ...defaultPUOptions(), maxLeafPoints: 40, maxPatchPoints: 120 } };
      const oracle = useRegions ? regionOracle2D(samples, computeRegions2D(samples, domain), 2e-5) : null;
      const stages = runPipeline(samples, opts, oracle);
      expect(stages.length).toBe(1 + opts.iterations + (stages[1].kind === 'collapsed' ? 1 : 0));
      const e0 = levelSetError(stages[0].model, shape);
      const eN = levelSetError(stages[stages.length - 1].model, shape);
      expect(eN).toBeLessThan(0.5 * e0);
      const last = stages[stages.length - 1];
      expect([...last.status].filter(isConstraint).length).toBeGreaterThan(20);
      if (useRegions) expect([...stages[1].status].filter((x) => x === Status.Fixed).length).toBeGreaterThan(0);
      if (method === 'pu') expect(last.model.kind === 'pu' && last.model.patches.length).toBeGreaterThan(1);
    });
  }
});

describe('partition of unity', () => {
  const pts = samplePositions2D({ kind: 'scattered', count: 400, seed: 3 }, domain);
  const n = pts.length / 2;
  const opts = { ...defaultPUOptions(), maxLeafPoints: 40, maxPatchPoints: 120 };

  it('splits into several patches and interpolates every constraint', () => {
    const vals = Float64Array.from({ length: n }, (_, i) => Math.hypot(pts[2 * i] - 0.1, pts[2 * i + 1]) - 0.5);
    const m = fitPU(2, pts, vals, 'cubic', opts);
    expect(m.patches.length).toBeGreaterThan(4);
    for (let i = 0; i < n; i++) expect(evalPU(m, pts, 2 * i)).toBeCloseTo(vals[i], 7);
  });

  it('reproduces linear functions', () => {
    const vals = Float64Array.from({ length: n }, (_, i) => 0.7 * pts[2 * i] - 1.3 * pts[2 * i + 1] + 0.2);
    const m = fitPU(2, pts, vals, 'cubic', opts);
    for (const q of [[0.31, -0.42], [-0.9, 0.85], [0, 0]]) expect(evalPU(m, q)).toBeCloseTo(0.7 * q[0] - 1.3 * q[1] + 0.2, 7);
  });

  it('blended gradient (with weight derivatives) matches finite differences', () => {
    const vals = Float64Array.from({ length: n }, (_, i) => Math.sin(3 * pts[2 * i]) * Math.cos(2 * pts[2 * i + 1]));
    const m = fitPU(2, pts, vals, 'cubic', opts);
    const h = 1e-6;
    for (const p of [[0.123, -0.456], [-0.61, 0.2], [0.8, 0.77]]) {
      const g = new Float64Array(2);
      const f = evalPUGrad(m, p, g);
      expect(f).toBeCloseTo(evalPU(m, p), 12);
      expect(g[0]).toBeCloseTo((evalPU(m, [p[0] + h, p[1]]) - evalPU(m, [p[0] - h, p[1]])) / (2 * h), 5);
      expect(g[1]).toBeCloseTo((evalPU(m, [p[0], p[1] + h]) - evalPU(m, [p[0], p[1] - h])) / (2 * h), 5);
    }
  });

  it('stays continuous where it leaves the union of supports', () => {
    const vals = Float64Array.from({ length: n }, (_, i) => Math.hypot(pts[2 * i], pts[2 * i + 1]) - 0.5);
    const m = fitPU(2, pts, vals, 'cubic', opts);
    const covered = (x: number, y: number) => m.patches.some((p) => Math.hypot(x - p.center[0], y - p.center[1]) < p.radius);
    const h = 1e-5;
    let exits = 0;
    for (let a = 0; a < 64; a++) {
      const ux = Math.cos((2 * Math.PI * a) / 64), uy = Math.sin((2 * Math.PI * a) / 64);
      // First exit from the union along the ray, then bisect to the boundary.
      let lo = 0.5, hi = lo;
      while (hi < 3 && covered(hi * ux, hi * uy)) hi += 0.01;
      if (hi >= 3) continue;
      lo = hi - 0.01;
      for (let k = 0; k < 40; k++) {
        const mid = (lo + hi) / 2;
        if (covered(mid * ux, mid * uy)) lo = mid;
        else hi = mid;
      }
      const inside = evalPU(m, [(lo - h) * ux, (lo - h) * uy]);
      const outside = evalPU(m, [(hi + h) * ux, (hi + h) * uy]);
      expect(Math.abs(outside - inside)).toBeLessThan(1e-3);
      exits++;
    }
    expect(exits).toBeGreaterThan(30);
  });

  it('fallback gradient matches finite differences outside the supports', () => {
    const vals = Float64Array.from({ length: n }, (_, i) => Math.sin(3 * pts[2 * i]) * Math.cos(2 * pts[2 * i + 1]));
    const m = fitPU(2, pts, vals, 'cubic', opts);
    const h = 1e-6;
    for (const p of [[1.6, 0.3], [-1.4, -1.5], [0.2, 2.1]]) {
      expect(m.patches.some((q) => Math.hypot(p[0] - q.center[0], p[1] - q.center[1]) < q.radius)).toBe(false);
      const g = new Float64Array(2);
      expect(evalPUGrad(m, p, g)).toBeCloseTo(evalPU(m, p), 12);
      expect(g[0]).toBeCloseTo((evalPU(m, [p[0] + h, p[1]]) - evalPU(m, [p[0] - h, p[1]])) / (2 * h), 4);
      expect(g[1]).toBeCloseTo((evalPU(m, [p[0], p[1] + h]) - evalPU(m, [p[0], p[1] - h])) / (2 * h), 4);
    }
  });

  it('lattice evaluation matches pointwise evaluation', () => {
    const vals = Float64Array.from({ length: n }, (_, i) => pts[2 * i] ** 2 - pts[2 * i + 1]);
    const m = fitPU(2, pts, vals, 'cubic', opts);
    const grid = evalPUGrid2D(m, -1.2, -1.2, 1.2, 1.2, 25, 25);
    for (let j = 0; j < 25; j += 3) for (let i = 0; i < 25; i += 3) {
      expect(grid[j * 25 + i]).toBeCloseTo(evalPU(m, [-1.2 + 0.1 * i, -1.2 + 0.1 * j]), 9);
    }
  });
});
