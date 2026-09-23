/** Application state shared by all figures, plus the worker clients that fill it in. */

import type { ExposedRegion2D } from '../core/arcs2d';
import type { Stage } from '../core/pipeline';
import type { PowerCell } from '../core/power2d';
import type { RBFModel } from '../core/rbf';
import type { Shape2D } from '../core/shape2d';
import type { Box2, Samples } from '../core/types';
import type { FieldRequest, FieldResponse, PipelineMessage, RunRequest, StageMetric } from './protocol';

export const DOMAIN: Box2 = { x0: -1, y0: -1, x1: 1, y1: 1 };

export interface DemoState {
  shape: Shape2D | null;
  samples: Samples | null;
  cells: (PowerCell | null)[];
  regions: ExposedRegion2D[];
  collapsed: Set<number>;
  stages: Stage[];
  metrics: StageMetric[];
  /** Whether the run used the exposed-region step. */
  usedRegions: boolean;
  running: boolean;
  progress: string;
  error: string | null;
  ms: number | null;
}

type Listener = (s: DemoState) => void;

export class Store {
  state: DemoState = {
    shape: null, samples: null, cells: [], regions: [], collapsed: new Set(),
    stages: [], metrics: [], usedRegions: false, running: false, progress: '', error: null, ms: null,
  };
  private listeners: Listener[] = [];
  subscribe(fn: Listener): void {
    this.listeners.push(fn);
  }
  update(patch: Partial<DemoState>): void {
    this.state = { ...this.state, ...patch };
    for (const fn of this.listeners) fn(this.state);
  }
}

/** Runs the pipeline in a worker; restarts the worker if a new run arrives mid-computation. */
export class PipelineClient {
  private worker: Worker | null = null;
  private runId = 0;
  private busy = false;

  constructor(private store: Store) {}

  private spawn(): Worker {
    const w = new Worker(new URL('./pipeline.worker.ts', import.meta.url), { type: 'module' });
    w.onmessage = (ev: MessageEvent<PipelineMessage>) => this.handle(ev.data);
    w.onerror = (ev) => {
      this.busy = false;
      this.store.update({ running: false, error: ev.message || 'Worker error' });
    };
    return w;
  }

  run(shape: Shape2D, req: Omit<RunRequest, 'type' | 'runId' | 'loops'>): void {
    if (this.busy && this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.worker ??= this.spawn();
    const runId = ++this.runId;
    this.busy = true;
    this.store.update({
      shape, samples: null, cells: [], regions: [], collapsed: new Set(), stages: [], metrics: [],
      usedRegions: req.useRegions, running: true, progress: 'Sampling', error: null, ms: null,
    });
    const msg: RunRequest = { type: 'run', runId, loops: shape.loops, ...req };
    this.worker.postMessage(msg);
  }

  private handle(m: PipelineMessage): void {
    if (m.runId !== this.runId) return;
    const s = this.store.state;
    switch (m.type) {
      case 'init':
        this.store.update({ samples: m.samples, cells: m.cells, regions: m.regions, collapsed: new Set(m.collapsed) });
        break;
      case 'progress':
        this.store.update({ progress: m.message });
        break;
      case 'stage': {
        const stages = s.stages.slice();
        const metrics = s.metrics.slice();
        stages[m.index] = m.stage;
        metrics[m.index] = m.metric;
        this.store.update({ stages, metrics });
        break;
      }
      case 'done':
        this.busy = false;
        this.store.update({ running: false, progress: '', ms: m.ms });
        break;
      case 'error':
        this.busy = false;
        this.store.update({ running: false, progress: '', error: m.message });
        break;
    }
  }
}

/**
 * Evaluates fields for display in a separate worker. Each consumer (figure) has at
 * most one request in flight; newer requests replace queued ones.
 */
export class FieldService {
  private worker = new Worker(new URL('./field.worker.ts', import.meta.url), { type: 'module' });
  private nextId = 1;
  private callbacks = new Map<number, { key: string; cb: (v: Float32Array) => void }>();
  private inFlight = new Set<string>();
  private pending = new Map<string, { req: Omit<FieldRequest, 'reqId'>; cb: (v: Float32Array) => void }>();

  constructor() {
    this.worker.onmessage = (ev: MessageEvent<FieldResponse>) => {
      const entry = this.callbacks.get(ev.data.reqId);
      if (!entry) return;
      this.callbacks.delete(ev.data.reqId);
      this.inFlight.delete(entry.key);
      entry.cb(ev.data.values);
      const next = this.pending.get(entry.key);
      if (next) {
        this.pending.delete(entry.key);
        this.send(entry.key, next.req, next.cb);
      }
    };
  }

  request(key: string, model: RBFModel, box: Box2, nx: number, ny: number, cb: (v: Float32Array) => void): void {
    const req = { model, box, nx, ny };
    if (this.inFlight.has(key)) this.pending.set(key, { req, cb });
    else this.send(key, req, cb);
  }

  private send(key: string, req: Omit<FieldRequest, 'reqId'>, cb: (v: Float32Array) => void): void {
    const reqId = this.nextId++;
    this.inFlight.add(key);
    this.callbacks.set(reqId, { key, cb });
    this.worker.postMessage({ reqId, ...req } satisfies FieldRequest);
  }
}
