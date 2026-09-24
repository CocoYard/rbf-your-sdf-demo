/// <reference lib="webworker" />
/** Evaluates fitted interpolants on lattices for display. Stateless. */

import { evalModelGrid2D } from '../core/interpolant';
import type { FieldRequest, FieldResponse } from './protocol';

self.onmessage = (ev: MessageEvent<FieldRequest>) => {
  const { reqId, model, box, nx, ny } = ev.data;
  const values = Float32Array.from(evalModelGrid2D(model, box.x0, box.y0, box.x1, box.y1, nx, ny));
  const res: FieldResponse = { reqId, values };
  (self as DedicatedWorkerGlobalScope).postMessage(res, [values.buffer]);
};
