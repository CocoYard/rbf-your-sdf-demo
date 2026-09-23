/** Drawing primitives for figures (world coordinates → figure screen space). */

import type { ExposedRegion2D } from '../core/arcs2d';
import type { PowerCell } from '../core/power2d';
import type { Shape2D } from '../core/shape2d';
import type { Samples } from '../core/types';
import type { Figure } from './figure';
import { colors } from './theme';

export function drawShape(
  ctx: CanvasRenderingContext2D, fig: Figure, shape: Shape2D,
  opts: { fill?: string; stroke?: string; width?: number; dash?: number[] } = {},
): void {
  ctx.save();
  ctx.beginPath();
  for (const L of shape.loops) {
    for (let k = 0; k < L.length; k += 2) {
      const [x, y] = fig.toScreen(L[k], L[k + 1]);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
  }
  if (opts.fill) {
    ctx.fillStyle = opts.fill;
    ctx.fill('evenodd');
  }
  if (opts.stroke) {
    ctx.strokeStyle = opts.stroke;
    ctx.lineWidth = opts.width ?? 1.5;
    ctx.lineJoin = 'round';
    if (opts.dash) ctx.setLineDash(opts.dash);
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Sample circles, red for d > 0 and blue for d < 0. `fade` lowers the opacity of large
 * circles so dense samplings stay readable (as in the paper's figures).
 */
export function drawCircles(
  ctx: CanvasRenderingContext2D, fig: Figure, s: Samples,
  opts: { alpha?: number; fade?: boolean; width?: number; only?: (i: number) => boolean; dash?: (i: number) => boolean } = {},
): void {
  const n = s.values.length;
  const base = opts.alpha ?? 0.8;
  ctx.save();
  ctx.lineWidth = opts.width ?? 1;
  for (let i = 0; i < n; i++) {
    if (opts.only && !opts.only(i)) continue;
    const d = s.values[i];
    const r = Math.abs(d) * fig.scale;
    const [x, y] = fig.toScreen(s.points[2 * i], s.points[2 * i + 1]);
    if (x + r < 0 || x - r > fig.width || y + r < 0 || y - r > fig.height) continue;
    ctx.globalAlpha = opts.fade ? base * Math.min(1, 0.12 / Math.max(Math.abs(d), 1e-3)) ** 0.7 : base;
    ctx.strokeStyle = d >= 0 ? colors.positive : colors.negative;
    ctx.setLineDash(opts.dash?.(i) ? [3, 3] : []);
    ctx.beginPath();
    ctx.arc(x, y, Math.max(r, 0.5), 0, 2 * Math.PI);
    ctx.stroke();
  }
  ctx.restore();
}

export function drawSampleDots(
  ctx: CanvasRenderingContext2D, fig: Figure, s: Samples, radius = 2.2, alpha = 1,
): void {
  ctx.save();
  ctx.globalAlpha = alpha;
  for (let i = 0; i < s.values.length; i++) {
    const [x, y] = fig.toScreen(s.points[2 * i], s.points[2 * i + 1]);
    ctx.fillStyle = s.values[i] >= 0 ? colors.positive : colors.negative;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.restore();
}

export function drawPoint(
  ctx: CanvasRenderingContext2D, fig: Figure, x: number, y: number,
  color: string, radius = 3.5, hollow = false,
): void {
  const [sx, sy] = fig.toScreen(x, y);
  ctx.beginPath();
  ctx.arc(sx, sy, radius, 0, 2 * Math.PI);
  if (hollow) {
    ctx.fillStyle = 'white';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  } else {
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }
}

export function drawPolyline(
  ctx: CanvasRenderingContext2D, fig: Figure, pts: ArrayLike<number>, color: string, width = 1, closed = false,
): void {
  if (pts.length < 4) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (let k = 0; k < pts.length; k += 2) {
    const [x, y] = fig.toScreen(pts[k], pts[k + 1]);
    if (k === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  if (closed) ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

export function drawCells(
  ctx: CanvasRenderingContext2D, fig: Figure, cells: (PowerCell | null)[], color = colors.powerCell, width = 0.8,
): void {
  for (const c of cells) if (c) drawPolyline(ctx, fig, c.polygon, color, width, true);
}

/** Exposed arcs as thick strokes along each circle; zero-length arcs as dots. */
export function drawArcs(
  ctx: CanvasRenderingContext2D, fig: Figure, s: Samples, regions: ExposedRegion2D[],
  opts: { color?: string; width?: number; only?: (i: number) => boolean } = {},
): void {
  ctx.save();
  ctx.strokeStyle = opts.color ?? colors.arc;
  ctx.fillStyle = opts.color ?? colors.arc;
  ctx.lineWidth = opts.width ?? 2.5;
  ctx.lineCap = 'round';
  regions.forEach((reg, i) => {
    if (opts.only && !opts.only(i)) return;
    const r = Math.abs(s.values[i]);
    const [x, y] = fig.toScreen(s.points[2 * i], s.points[2 * i + 1]);
    for (const a of reg.arcs) {
      if (a.end - a.start < 1e-9 || r * fig.scale * (a.end - a.start) < 1.5) {
        const t = (a.start + a.end) / 2;
        ctx.beginPath();
        ctx.arc(x + r * fig.scale * Math.cos(t), y - r * fig.scale * Math.sin(t), ctx.lineWidth * 0.9, 0, 2 * Math.PI);
        ctx.fill();
        continue;
      }
      ctx.beginPath();
      // Screen y is flipped, so angles are negated and the direction reversed.
      ctx.arc(x, y, r * fig.scale, -a.end, -a.start);
      ctx.stroke();
    }
  });
  ctx.restore();
}

/** Nearest sample center within `maxPx` screen pixels, or −1. */
export function pickSample(fig: Figure, s: Samples | null, world: [number, number], maxPx = 14): number {
  if (!s) return -1;
  let best = -1, bd = (maxPx / fig.scale) ** 2;
  for (let i = 0; i < s.values.length; i++) {
    const dx = s.points[2 * i] - world[0], dy = s.points[2 * i + 1] - world[1];
    const d2 = dx * dx + dy * dy;
    if (d2 < bd) {
      bd = d2;
      best = i;
    }
  }
  return best;
}
