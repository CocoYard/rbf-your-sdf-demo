/** Colors used by the figures (kept in one place so the legend in HTML/CSS matches). */

import { Status } from '../core/pipeline';

export const colors = {
  positive: '#d6453d', // sample circles with d > 0 (outside)
  negative: '#2f6db5', // sample circles with d < 0 (inside)
  fieldPos: [244, 165, 130] as [number, number, number],
  fieldNeg: [146, 197, 222] as [number, number, number],
  isoline: 'rgba(40, 40, 40, 0.16)',
  groundTruth: '#555',
  shapeFill: 'rgba(0, 0, 0, 0.07)',
  levelSet: '#1a8a4a',
  previousLevelSet: '#8a3ab9',
  powerCell: 'rgba(60, 60, 60, 0.45)',
  arc: '#e08a00',
  highlight: '#111',
  descent: 'rgba(20, 20, 20, 0.55)',
};

export const statusStyle: Record<number, { color: string; hollow: boolean; label: string }> = {
  [Status.Fixed]: { color: '#b0158c', hollow: false, label: 'fixed by collapsed exposed region' },
  [Status.Projected]: { color: '#12a150', hollow: false, label: 'projected tangent point' },
  [Status.KeptPrevious]: { color: '#12a150', hollow: true, label: 'kept previous (update infeasible)' },
  [Status.Clamped]: { color: '#e08a00', hollow: false, label: 'clamped onto tiny exposed region' },
  [Status.Infeasible]: { color: '#888', hollow: true, label: 'culled: outside exposed region' },
  [Status.Duplicate]: { color: '#888', hollow: false, label: 'dropped: duplicate' },
};
