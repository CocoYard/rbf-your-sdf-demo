/// <reference lib="webworker" />
/** Runs the full pipeline off the main thread and streams stages back as they finish. */

import { runPipeline } from '../core/pipeline';
import { computeRegions2D, regionOracle2D } from '../core/regions2d';
import { samplePositions2D, sampleShapeSDF } from '../core/sampling';
import { sampleBoundary, type Shape2D } from '../core/shape2d';
import { stageMetric } from './metrics';
import type { PipelineMessage, RunRequest } from './protocol';

const post = (m: PipelineMessage) => (self as DedicatedWorkerGlobalScope).postMessage(m);

self.onmessage = (ev: MessageEvent<RunRequest>) => {
  const req = ev.data;
  const { runId } = req;
  const t0 = performance.now();
  try {
    const shape: Shape2D = { loops: req.loops };
    const samples = sampleShapeSDF(shape, samplePositions2D(req.sampling, req.domain));
    post({ type: 'progress', runId, message: 'Computing power diagram' });
    const regions = computeRegions2D(samples, req.domain, req.epsTan);
    const oracle = regionOracle2D(samples, regions, req.epsDegen);
    const collapsed: number[] = [];
    for (let i = 0; i < samples.values.length; i++) if (oracle.collapsedCandidates(i)) collapsed.push(i);
    post({ type: 'init', runId, samples, cells: regions.diagram.cells, regions: regions.regions, collapsed });

    const boundary = sampleBoundary(shape, 0.004);
    runPipeline(samples, req.options, req.useRegions ? oracle : null, {
      onProgress: (message) => post({ type: 'progress', runId, message }),
      onStage: (stage, index) => {
        post({ type: 'stage', runId, index, stage, metric: stageMetric(stage.model, shape, req.domain, boundary) });
      },
    });
    post({ type: 'done', runId, ms: performance.now() - t0 });
  } catch (e) {
    post({ type: 'error', runId, message: e instanceof Error ? e.message : String(e) });
  }
};
