/**
 * Field layer: requests an RBF evaluated over a figure's visible region (from the
 * field worker), renders it as a diverging colormap with iso-contours, and extracts
 * the zero level set by marching squares.
 */

import { marchingSquares } from '../core/contour2d';
import type { RBFModel } from '../core/rbf';
import type { Box2 } from '../core/types';
import type { FieldService } from '../app/state';
import type { Figure } from './figure';
import { colors } from './theme';

/** Diverging colormap: blue (negative) → white → red (positive). */
function colormap(v: number, range: number): [number, number, number] {
  const t = Math.max(-1, Math.min(1, v / range));
  const a = Math.pow(Math.abs(t), 0.8);
  const [r, g, b] = t >= 0 ? colors.fieldPos : colors.fieldNeg;
  return [255 + (r - 255) * a, 255 + (g - 255) * a, 255 + (b - 255) * a];
}

interface Cached {
  model: RBFModel;
  box: Box2;
  nx: number;
  ny: number;
  image: HTMLCanvasElement;
  zero: Float64Array;
  isolines: Float64Array[];
}

export interface FieldStyle {
  showColors: boolean;
  showIsolines: boolean;
  showZero: boolean;
  zeroColor: string;
  zeroWidth: number;
  zeroDash?: number[];
}

export class FieldLayer {
  private cached: Cached | null = null;
  private wanted: { model: RBFModel; box: Box2; nx: number; ny: number } | null = null;
  /** CSS pixels per field sample. */
  pixelsPerSample = 2;
  isoSpacing = 0.05;
  colorRange = 0.6;

  constructor(
    private fig: Figure,
    private service: FieldService,
    private key: string,
  ) {}

  /** Ensure the cached field matches `model` and the current view; request it if not. */
  update(model: RBFModel | null): void {
    if (!model) {
      this.cached = null;
      this.wanted = null;
      return;
    }
    const vb = this.fig.viewBox();
    // Pad the box a little so small pans reuse the current image.
    const padX = 0.1 * (vb.x1 - vb.x0), padY = 0.1 * (vb.y1 - vb.y0);
    const box = { x0: vb.x0 - padX, y0: vb.y0 - padY, x1: vb.x1 + padX, y1: vb.y1 + padY };
    const c = this.cached;
    const covers = c && c.model === model && c.box.x0 <= vb.x0 && c.box.x1 >= vb.x1 && c.box.y0 <= vb.y0 && c.box.y1 >= vb.y1;
    const res = c ? (c.nx - 1) / (c.box.x1 - c.box.x0) : 0; // samples per world unit
    const wantRes = this.fig.scale / this.pixelsPerSample;
    if (covers && res > 0.7 * wantRes && res < 1.5 * wantRes) return;
    const w = this.wanted;
    if (w && w.model === model && Math.abs(w.box.x0 - box.x0) < 1e-9 && Math.abs(w.box.x1 - box.x1) < 1e-9 && Math.abs(w.box.y0 - box.y0) < 1e-9) return;

    const nx = Math.max(8, Math.min(700, Math.round((box.x1 - box.x0) * wantRes) + 1));
    const ny = Math.max(8, Math.min(700, Math.round((box.y1 - box.y0) * wantRes) + 1));
    this.wanted = { model, box, nx, ny };
    this.service.request(this.key, model, box, nx, ny, (values) => {
      if (this.wanted?.model !== model) return;
      this.cached = this.build(model, box, nx, ny, values);
      this.fig.redraw();
    });
  }

  private build(model: RBFModel, box: Box2, nx: number, ny: number, values: Float32Array): Cached {
    const image = document.createElement('canvas');
    image.width = nx;
    image.height = ny;
    const ictx = image.getContext('2d')!;
    const img = ictx.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      const row = ny - 1 - j; // image rows go down, world y goes up
      for (let i = 0; i < nx; i++) {
        const [r, g, b] = colormap(values[j * nx + i], this.colorRange);
        const k = 4 * (row * nx + i);
        img.data[k] = r;
        img.data[k + 1] = g;
        img.data[k + 2] = b;
        img.data[k + 3] = 255;
      }
    }
    ictx.putImageData(img, 0, 0);

    let lo = Infinity, hi = -Infinity;
    for (const v of values) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const isolines: Float64Array[] = [];
    const s = this.isoSpacing;
    const kmin = Math.max(Math.ceil(lo / s), -60), kmax = Math.min(Math.floor(hi / s), 60);
    for (let k = kmin; k <= kmax; k++) {
      if (k === 0) continue;
      isolines.push(marchingSquares(values, nx, ny, box.x0, box.y0, box.x1, box.y1, k * s));
    }
    const zero = marchingSquares(values, nx, ny, box.x0, box.y0, box.x1, box.y1, 0);
    return { model, box, nx, ny, image, zero, isolines };
  }

  /** The zero level set of the currently displayed field (may lag the requested model). */
  zeroSegments(): Float64Array | null {
    return this.cached?.zero ?? null;
  }

  draw(ctx: CanvasRenderingContext2D, style: FieldStyle): void {
    const c = this.cached;
    if (!c) return;
    const fig = this.fig;
    const [sx0, sy1] = fig.toScreen(c.box.x0, c.box.y0);
    const [sx1, sy0] = fig.toScreen(c.box.x1, c.box.y1);
    if (style.showColors) {
      ctx.save();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      // Lattice points are pixel centers; extend by half a sample on each side.
      const hx = (sx1 - sx0) / (c.nx - 1) / 2, hy = (sy1 - sy0) / (c.ny - 1) / 2;
      ctx.drawImage(c.image, sx0 - hx, sy0 - hy, sx1 - sx0 + 2 * hx, sy1 - sy0 + 2 * hy);
      ctx.restore();
    }
    if (style.showIsolines) {
      ctx.save();
      ctx.strokeStyle = colors.isoline;
      ctx.lineWidth = 0.6;
      for (const segs of c.isolines) strokeSegments(ctx, fig, segs);
      ctx.restore();
    }
    if (style.showZero) {
      ctx.save();
      ctx.strokeStyle = style.zeroColor;
      ctx.lineWidth = style.zeroWidth;
      ctx.lineCap = 'round';
      if (style.zeroDash) ctx.setLineDash(style.zeroDash);
      strokeSegments(ctx, fig, c.zero);
      ctx.restore();
    }
  }
}

export function strokeSegments(ctx: CanvasRenderingContext2D, fig: Figure, segs: Float64Array): void {
  ctx.beginPath();
  for (let k = 0; k < segs.length; k += 4) {
    const [ax, ay] = fig.toScreen(segs[k], segs[k + 1]);
    const [bx, by] = fig.toScreen(segs[k + 2], segs[k + 3]);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();
}
