/** A tiny canvas line plot for insets (objective along a circle, error per iteration). */

export interface Series {
  xs: ArrayLike<number>;
  ys: ArrayLike<number>;
  color: string;
  width?: number;
  dash?: number[];
  /** Draw markers at each point. */
  dots?: boolean;
}

export interface PlotSpec {
  series: Series[];
  xRange: [number, number];
  yRange?: [number, number];
  /** Shaded x-intervals (e.g. exposed arcs). */
  bands?: { x0: number; x1: number; color: string }[];
  /** Vertical markers. */
  marks?: { x: number; color: string; label?: string }[];
  /** Point markers. */
  points?: { x: number; y: number; color: string; hollow?: boolean }[];
  xLabel?: string;
  yLabel?: string;
  xTicks?: { x: number; label: string }[];
  /** Horizontal reference line (e.g. y = 0). */
  zeroLine?: boolean;
  logY?: boolean;
}

export class Plot {
  readonly canvas: HTMLCanvasElement;
  private spec: PlotSpec | null = null;
  private w = 1;
  private h = 1;

  constructor(container: HTMLElement, className = 'plot') {
    this.canvas = document.createElement('canvas');
    this.canvas.className = className;
    container.appendChild(this.canvas);
    new ResizeObserver(() => this.resize()).observe(this.canvas);
  }

  private resize(): void {
    const r = this.canvas.getBoundingClientRect();
    this.w = Math.max(1, Math.round(r.width));
    this.h = Math.max(1, Math.round(r.height));
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.draw();
  }

  set(spec: PlotSpec | null): void {
    this.spec = spec;
    this.draw();
  }

  private draw(): void {
    const ctx = this.canvas.getContext('2d')!;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    const spec = this.spec;
    if (!spec) return;

    const padL = 46, padR = 10, padT = 10, padB = spec.xLabel ? 34 : 20;
    const W = this.w - padL - padR, H = this.h - padT - padB;
    if (W <= 0 || H <= 0) return;
    const ty = (v: number) => (spec.logY ? Math.log10(Math.max(v, 1e-12)) : v);

    let [y0, y1] = spec.yRange ?? [Infinity, -Infinity];
    if (!spec.yRange) {
      for (const s of spec.series) for (let k = 0; k < s.ys.length; k++) {
        const v = ty(s.ys[k]);
        if (Number.isFinite(v)) {
          y0 = Math.min(y0, v);
          y1 = Math.max(y1, v);
        }
      }
      if (!Number.isFinite(y0)) [y0, y1] = [0, 1];
      if (y1 - y0 < 1e-12) [y0, y1] = [y0 - 0.5, y1 + 0.5];
      const m = 0.08 * (y1 - y0);
      y0 -= m;
      y1 += m;
    }
    const [x0, x1] = spec.xRange;
    const X = (x: number) => padL + ((x - x0) / (x1 - x0)) * W;
    const Y = (y: number) => padT + (1 - (ty(y) - y0) / (y1 - y0)) * H;

    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = '#666';
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, W, H);

    for (const b of spec.bands ?? []) {
      ctx.fillStyle = b.color;
      const a = Math.max(padL, X(b.x0)), c = Math.min(padL + W, X(b.x1));
      ctx.fillRect(a, padT, Math.max(1.5, c - a), H);
    }

    // y ticks
    ctx.fillStyle = '#666';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yt = spec.logY ? logTicks(y0, y1) : niceTicks(y0, y1, 4);
    for (const v of yt) {
      const py = padT + (1 - (v - y0) / (y1 - y0)) * H;
      ctx.fillText(fmt(spec.logY ? Math.pow(10, v) : v), padL - 4, py);
      ctx.strokeStyle = '#eee';
      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(padL + W, py);
      ctx.stroke();
    }
    // x ticks
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xt = spec.xTicks ?? niceTicks(x0, x1, 6).map((x) => ({ x, label: fmt(x) }));
    for (const t of xt) ctx.fillText(t.label, X(t.x), padT + H + 3);
    if (spec.xLabel) ctx.fillText(spec.xLabel, padL + W / 2, padT + H + 17);
    if (spec.yLabel) {
      ctx.save();
      ctx.translate(11, padT + H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'middle';
      ctx.fillText(spec.yLabel, 0, 0);
      ctx.restore();
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, padT, W, H);
    ctx.clip();
    if (spec.zeroLine && !spec.logY && y0 < 0 && y1 > 0) {
      ctx.strokeStyle = '#999';
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, Y(0));
      ctx.lineTo(padL + W, Y(0));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const m of spec.marks ?? []) {
      ctx.strokeStyle = m.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(X(m.x), padT);
      ctx.lineTo(X(m.x), padT + H);
      ctx.stroke();
    }
    for (const s of spec.series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width ?? 1.5;
      ctx.setLineDash(s.dash ?? []);
      ctx.beginPath();
      let pen = false;
      for (let k = 0; k < s.xs.length; k++) {
        const v = s.ys[k];
        if (!Number.isFinite(v)) {
          pen = false;
          continue;
        }
        if (pen) ctx.lineTo(X(s.xs[k]), Y(v));
        else ctx.moveTo(X(s.xs[k]), Y(v));
        pen = true;
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (s.dots) {
        ctx.fillStyle = s.color;
        for (let k = 0; k < s.xs.length; k++) {
          if (!Number.isFinite(s.ys[k])) continue;
          ctx.beginPath();
          ctx.arc(X(s.xs[k]), Y(s.ys[k]), 2.5, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }
    for (const p of spec.points ?? []) {
      ctx.beginPath();
      ctx.arc(X(p.x), Y(p.y), 4, 0, 2 * Math.PI);
      if (p.hollow) {
        ctx.fillStyle = 'white';
        ctx.fill();
        ctx.strokeStyle = p.color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      } else {
        ctx.fillStyle = p.color;
        ctx.fill();
      }
    }
    ctx.restore();
  }
}

function fmt(v: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.001) return v.toExponential(0);
  return String(+v.toPrecision(3));
}

/** Ticks for a log10 axis spanning [lo, hi] (in log units): 1·10^k, plus 2 and 5 if the span is short. */
function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  const mults = hi - lo < 1.5 ? [1, 2, 5] : [1];
  for (let k = Math.floor(lo); k <= Math.ceil(hi); k++) {
    for (const m of mults) {
      const v = k + Math.log10(m);
      if (v >= lo && v <= hi) out.push(v);
    }
  }
  return out;
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
