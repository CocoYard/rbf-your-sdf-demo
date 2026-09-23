/** 2D exposed regions: power diagram + exposed arcs, packaged as an ExposedRegionOracle. */

import { arcPoint, distanceToArcs, exposedRegions2D, type ExposedRegion2D } from './arcs2d';
import { powerDiagram2D, type PowerDiagram } from './power2d';
import type { ExposedRegionOracle } from './regions';
import type { Box2, Samples } from './types';

export interface Regions2D {
  diagram: PowerDiagram;
  regions: ExposedRegion2D[];
}

export function computeRegions2D(samples: Samples, domain: Box2, epsTan = 1e-4): Regions2D {
  if (samples.dim !== 2) throw new Error('computeRegions2D requires 2D samples');
  const radii = samples.values.map(Math.abs);
  // A box far larger than any circle so it never clips an exposed arc.
  const w = domain.x1 - domain.x0, h = domain.y1 - domain.y0;
  const pad = 20 * Math.max(w, h);
  const box = { x0: domain.x0 - pad, y0: domain.y0 - pad, x1: domain.x1 + pad, y1: domain.y1 + pad };
  const diagram = powerDiagram2D(samples.points, radii, box);
  const regions = exposedRegions2D(samples.points, radii, diagram, epsTan);
  return { diagram, regions };
}

export function regionOracle2D(samples: Samples, r: Regions2D, epsDegen: number): ExposedRegionOracle {
  const P = samples.points;
  const radius = (i: number) => Math.abs(samples.values[i]);
  return {
    collapsedCandidates(i) {
      const reg = r.regions[i];
      if (reg.full || reg.arcs.length === 0 || reg.length >= epsDegen) return null;
      const out = new Float64Array(reg.arcs.length * 2);
      reg.arcs.forEach((a, k) => {
        const [x, y] = arcPoint(P[2 * i], P[2 * i + 1], radius(i), (a.start + a.end) / 2);
        out[2 * k] = x;
        out[2 * k + 1] = y;
      });
      return out;
    },
    distance(i, y) {
      const cx = P[2 * i], cy = P[2 * i + 1], rad = radius(i);
      const reg = r.regions[i];
      if (reg.arcs.length === 0) return { distance: Infinity, closest: Float64Array.from(y) };
      const theta = Math.atan2(y[1] - cy, y[0] - cx);
      const { distance, closest } = distanceToArcs(reg.arcs, rad, theta);
      return { distance, closest: Float64Array.from(arcPoint(cx, cy, rad, closest)) };
    },
    size(i) {
      return r.regions[i].length;
    },
  };
}
