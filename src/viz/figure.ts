/**
 * A pannable/zoomable 2D canvas figure in world coordinates (y up).
 * Drawing is delegated to an `onDraw` callback; hover/click report world positions.
 */

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
  /** CSS pixels per world unit. */
  scale = 1;
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
    const w = Math.max(1, Math.round(r.width));
    const h = Math.max(1, Math.round(r.height));
    const first = this.width === 1;
    const oldMin = Math.min(this.width, this.height);
    this.width = w;
    this.height = h;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    if (first) this.resetView();
    else {
      this.scale *= Math.min(w, h) / oldMin;
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
      const dpr = window.devicePixelRatio || 1;
      const ctx = this.ctx;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.width, this.height);
      this.opts.onDraw(this, ctx);
    });
  }

  private local(ev: MouseEvent): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [ev.clientX - r.left, ev.clientY - r.top];
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
        this.scale = Math.min(1e6, Math.max(10, this.scale * factor));
        // Keep the point under the cursor fixed.
        this.cx = wx - (sx - this.width / 2) / this.scale;
        this.cy = wy + (sy - this.height / 2) / this.scale;
        this.viewChanged();
      },
      { passive: false },
    );
    c.addEventListener('pointerdown', (ev) => {
      const [x, y] = this.local(ev);
      this.dragging = { x, y, cx: this.cx, cy: this.cy, moved: false };
      c.setPointerCapture(ev.pointerId);
    });
    c.addEventListener('pointermove', (ev) => {
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
      const d = this.dragging;
      this.dragging = null;
      c.classList.remove('panning');
      if (d && !d.moved) this.opts.onClick?.(this.toWorld(...this.local(ev)), this);
    });
    c.addEventListener('pointerleave', () => {
      if (!this.dragging) this.opts.onHover?.(null, this);
    });
    c.addEventListener('dblclick', () => this.resetView());
  }
}
