/**
 * A pannable/zoomable 2D canvas figure in world coordinates (y up).
 * Drawing is delegated to an `onDraw` callback; hover/click report world positions.
 *
 * Figures narrower than MIN_LOGICAL_WIDTH (phones) draw in a wider logical pixel space
 * scaled down to fit, so line widths and marker sizes stay in proportion to the figure.
 * Mouse: drag to pan, wheel to zoom, double-click to reset. Touch: one finger scrolls
 * the page, two fingers pinch/pan, tap = hover + click, double-tap to reset.
 */

const MIN_LOGICAL_WIDTH = 560;

import type { Box2 } from '../core/types';

export interface FigureOptions {
  /** Default visible world box. */
  home: Box2;
  onDraw: (fig: Figure, ctx: CanvasRenderingContext2D) => void;
  onHover?: (world: [number, number] | null, fig: Figure) => void;
  onClick?: (world: [number, number], fig: Figure) => void;
  /** Called after the view changes (pan/zoom/resize). */
  onView?: (fig: Figure) => void;
}

export class Figure {
  readonly root: HTMLDivElement;
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  width = 1;
  height = 1;
  cx = 0;
  cy = 0;
  /** Logical pixels per world unit (logical = CSS pixels unless the figure is narrow). */
  scale = 1;
  /** CSS pixels per logical pixel (≤ 1). */
  private k = 1;
  /** True while the last interaction was touch (hit targets are enlarged). */
  touchInput = false;
  private frame = 0;
  private dragging: { x: number; y: number; cx: number; cy: number; moved: boolean } | null = null;

  constructor(container: HTMLElement, readonly opts: FigureOptions) {
    this.root = document.createElement('div');
    this.root.className = 'fig-canvas';
    this.canvas = document.createElement('canvas');
    this.root.appendChild(this.canvas);
    const reset = document.createElement('button');
    reset.className = 'fig-reset';
    reset.type = 'button';
    reset.title = 'Reset view (or double-click)';
    reset.textContent = '⟲';
    reset.addEventListener('click', () => this.resetView());
    this.root.appendChild(reset);
    container.appendChild(this.root);
    this.ctx = this.canvas.getContext('2d')!;

    new ResizeObserver(() => this.resize()).observe(this.root);
    this.resize();
    this.resetView();
    this.bindEvents();
  }

  resetView(): void {
    const h = this.opts.home;
    this.cx = (h.x0 + h.x1) / 2;
    this.cy = (h.y0 + h.y1) / 2;
    this.scale = Math.min(this.width / (h.x1 - h.x0), this.height / (h.y1 - h.y0));
    this.viewChanged();
  }

  private resize(): void {
    const r = this.root.getBoundingClientRect();
    const cw = Math.max(1, Math.round(r.width));
    const ch = Math.max(1, Math.round(r.height));
    const first = this.width === 1;
    const oldMin = Math.min(this.width, this.height);
    this.k = Math.min(1, cw / MIN_LOGICAL_WIDTH);
    this.width = cw / this.k;
    this.height = ch / this.k;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(cw * dpr);
    this.canvas.height = Math.round(ch * dpr);
    this.canvas.style.width = `${cw}px`;
    this.canvas.style.height = `${ch}px`;
    if (first) this.resetView();
    else {
      this.scale *= Math.min(this.width, this.height) / oldMin;
      this.viewChanged();
    }
  }

  private viewChanged(): void {
    this.opts.onView?.(this);
    this.redraw();
  }

  toScreen(x: number, y: number): [number, number] {
    return [this.width / 2 + (x - this.cx) * this.scale, this.height / 2 - (y - this.cy) * this.scale];
  }

  toWorld(sx: number, sy: number): [number, number] {
    return [this.cx + (sx - this.width / 2) / this.scale, this.cy - (sy - this.height / 2) / this.scale];
  }

  /** Visible world rectangle. */
  viewBox(): Box2 {
    const [x0, y1] = this.toWorld(0, 0);
    const [x1, y0] = this.toWorld(this.width, this.height);
    return { x0, y0, x1, y1 };
  }

  redraw(): void {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      const s = (window.devicePixelRatio || 1) * this.k;
      const ctx = this.ctx;
      ctx.setTransform(s, 0, 0, s, 0, 0);
      ctx.clearRect(0, 0, this.width, this.height);
      this.opts.onDraw(this, ctx);
    });
  }

  /** Hit-test radius multiplier: fingers are less precise than a mouse. */
  get hitScale(): number {
    return this.touchInput ? 1.8 / this.k : 1;
  }

  /** Client coordinates → logical figure pixels. */
  private local(ev: { clientX: number; clientY: number }): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) / this.k, (ev.clientY - r.top) / this.k];
  }

  /** Zoom to `scale`, keeping world point (wx, wy) at logical screen point (sx, sy). */
  private zoomAround(wx: number, wy: number, sx: number, sy: number, scale: number): void {
    this.scale = Math.min(1e6, Math.max(10, scale));
    this.cx = wx - (sx - this.width / 2) / this.scale;
    this.cy = wy + (sy - this.height / 2) / this.scale;
    this.viewChanged();
  }

  private bindEvents(): void {
    const c = this.canvas;
    c.addEventListener(
      'wheel',
      (ev) => {
        ev.preventDefault();
        const [sx, sy] = this.local(ev);
        const [wx, wy] = this.toWorld(sx, sy);
        const factor = Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0015));
        this.zoomAround(wx, wy, sx, sy, this.scale * factor);
      },
      { passive: false },
    );
    c.addEventListener('pointerdown', (ev) => {
      if (ev.pointerType === 'touch') return;
      this.touchInput = false;
      const [x, y] = this.local(ev);
      this.dragging = { x, y, cx: this.cx, cy: this.cy, moved: false };
      c.setPointerCapture(ev.pointerId);
    });
    c.addEventListener('pointermove', (ev) => {
      if (ev.pointerType === 'touch') return;
      const [x, y] = this.local(ev);
      if (this.dragging) {
        const dx = x - this.dragging.x, dy = y - this.dragging.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) this.dragging.moved = true;
        if (this.dragging.moved) {
          this.cx = this.dragging.cx - dx / this.scale;
          this.cy = this.dragging.cy + dy / this.scale;
          c.classList.add('panning');
          this.viewChanged();
          return;
        }
      }
      this.opts.onHover?.(this.toWorld(x, y), this);
    });
    c.addEventListener('pointerup', (ev) => {
      if (ev.pointerType === 'touch') return;
      const d = this.dragging;
      this.dragging = null;
      c.classList.remove('panning');
      if (d && !d.moved) this.opts.onClick?.(this.toWorld(...this.local(ev)), this);
    });
    c.addEventListener('pointerleave', (ev) => {
      if (ev.pointerType !== 'touch' && !this.dragging) this.opts.onHover?.(null, this);
    });
    c.addEventListener('dblclick', () => {
      if (!this.touchInput) this.resetView();
    });
    this.bindTouch();
  }

  private bindTouch(): void {
    const c = this.canvas;
    let pinch: { wx: number; wy: number; dist: number; scale: number } | null = null;
    let tap: { x: number; y: number; t: number } | null = null;
    let lastTap: { x: number; y: number; t: number } | null = null;
    const two = (ev: TouchEvent) => {
      const [a, b] = [ev.touches[0], ev.touches[1]];
      const [ax, ay] = this.local(a), [bx, by] = this.local(b);
      return { mx: (ax + bx) / 2, my: (ay + by) / 2, dist: Math.max(1, Math.hypot(ax - bx, ay - by)) };
    };
    const startPinch = (ev: TouchEvent) => {
      const g = two(ev);
      const [wx, wy] = this.toWorld(g.mx, g.my);
      pinch = { wx, wy, dist: g.dist, scale: this.scale };
    };

    c.addEventListener('touchstart', (ev) => {
      this.touchInput = true;
      if (ev.touches.length >= 2) {
        ev.preventDefault();
        tap = null;
        startPinch(ev);
      } else {
        const [x, y] = this.local(ev.touches[0]);
        tap = { x, y, t: performance.now() };
      }
    }, { passive: false });

    c.addEventListener('touchmove', (ev) => {
      if (ev.touches.length >= 2) {
        ev.preventDefault(); // two fingers belong to the figure, not the page
        if (!pinch) startPinch(ev);
        const g = two(ev), p = pinch!;
        this.zoomAround(p.wx, p.wy, g.mx, g.my, (p.scale * g.dist) / p.dist);
        return;
      }
      if (tap) {
        const [x, y] = this.local(ev.touches[0]);
        if (Math.hypot(x - tap.x, y - tap.y) * this.k > 10) tap = null;
      }
    }, { passive: false });

    const end = (ev: TouchEvent) => {
      if (ev.touches.length >= 2) {
        startPinch(ev);
        return;
      }
      pinch = null;
      if (ev.type !== 'touchend' || !tap || ev.touches.length > 0) return;
      const now = performance.now();
      const t = tap;
      tap = null;
      if (now - t.t > 500) return;
      ev.preventDefault(); // no synthetic mouse events / click
      if (lastTap && now - lastTap.t < 320 && Math.hypot(t.x - lastTap.x, t.y - lastTap.y) * this.k < 30) {
        lastTap = null;
        this.resetView();
        return;
      }
      lastTap = { ...t, t: now };
      const w = this.toWorld(t.x, t.y);
      this.opts.onHover?.(w, this);
      this.opts.onClick?.(w, this);
    };
    c.addEventListener('touchend', end, { passive: false });
    c.addEventListener('touchcancel', end);
  }
}
