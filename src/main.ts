import katex from 'katex';
import 'katex/dist/katex.min.css';
import './style.css';

import type { Arc } from './core/arcs2d';
import { Status, defaultOptions, isConstraint, type Stage } from './core/pipeline';
import { evalModel, type Interpolant } from './core/interpolant';
import type { SamplingSpec } from './core/sampling';
import { closestPointOnShape, type Shape2D } from './core/shape2d';
import type { Samples } from './core/types';
import { DOMAIN, FieldService, PipelineClient, Store, type DemoState } from './app/state';
import { svgToShape } from './io/svg';
import { drawArcs, drawCells, drawCircles, drawPatches, drawPoint, drawPolyline, drawSampleDots, drawShape, pickSample } from './viz/draw';
import { FieldLayer } from './viz/field';
import { Figure } from './viz/figure';
import { Plot } from './viz/plot';
import { colors, statusStyle } from './viz/theme';

// ---------------------------------------------------------------------------
// Math typesetting
// ---------------------------------------------------------------------------
for (const el of document.querySelectorAll<HTMLElement>('.m, .md')) {
  katex.render(el.textContent ?? '', el, { displayMode: el.classList.contains('md'), throwOnError: false });
}

// ---------------------------------------------------------------------------
// State and workers
// ---------------------------------------------------------------------------
const store = new Store();
const pipeline = new PipelineClient(store);
const fields = new FieldService();
const HOME = { x0: -1.04, y0: -1.04, x1: 1.04, y1: 1.04 };
const TWO_PI = 2 * Math.PI;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------------------------------------------------------------------------
// Global controls
// ---------------------------------------------------------------------------
const ui = {
  shape: $<HTMLSelectElement>('shape'),
  upload: $<HTMLInputElement>('upload'),
  gridN: $<HTMLInputElement>('grid-n'),
  gridOut: $<HTMLOutputElement>('grid-n-out'),
  count: $<HTMLInputElement>('count'),
  countOut: $<HTMLOutputElement>('count-out'),
  seed: $<HTMLInputElement>('seed'),
  reseed: $<HTMLButtonElement>('reseed'),
  iters: $<HTMLInputElement>('iters'),
  itersOut: $<HTMLOutputElement>('iters-out'),
  useRegions: $<HTMLInputElement>('use-regions'),
  usePU: $<HTMLInputElement>('use-pu'),
  puOverlap: $<HTMLInputElement>('pu-overlap'),
  puLeaf: $<HTMLInputElement>('pu-leaf'),
  puPatch: $<HTMLInputElement>('pu-patch'),
  puMin: $<HTMLInputElement>('pu-min'),
  puRepair: $<HTMLInputElement>('pu-repair'),
  kernel: $<HTMLSelectElement>('kernel'),
  descentIters: $<HTMLInputElement>('descent-iters'),
  epsDegen: $<HTMLInputElement>('eps-degen'),
  dedup: $<HTMLInputElement>('dedup'),
  deferIsolated: $<HTMLInputElement>('defer-isolated'),
  filterInfeasible: $<HTMLInputElement>('filter-infeasible'),
  monotone: $<HTMLInputElement>('monotone'),
  clamp: $<HTMLInputElement>('clamp'),
  status: $('status'),
};

// Advanced settings live in a side panel.
const advToggle = $<HTMLButtonElement>('adv-toggle');
const advPanel = $('adv-panel');
function setAdvancedOpen(open: boolean): void {
  advPanel.hidden = !open;
  advToggle.setAttribute('aria-expanded', String(open));
  document.body.classList.toggle('adv-open', open);
}
advToggle.addEventListener('click', () => setAdvancedOpen(advPanel.hidden));
$('adv-close').addEventListener('click', () => setAdvancedOpen(false));
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && !advPanel.hidden) setAdvancedOpen(false);
});

let shape: Shape2D | null = null;
const uploaded = new Map<string, string>();

function samplingKind(): 'grid' | 'scattered' {
  return (document.querySelector<HTMLInputElement>('input[name=sampling]:checked')?.value ?? 'grid') as 'grid' | 'scattered';
}

function syncControlLabels(): void {
  const grid = samplingKind() === 'grid';
  $('ctl-grid').hidden = !grid;
  $('ctl-count').hidden = grid;
  $('ctl-seed').hidden = grid;
  ui.gridOut.textContent = `${ui.gridN.value}²`;
  ui.countOut.textContent = ui.count.value;
  ui.itersOut.textContent = ui.iters.value;
}

function run(): void {
  syncControlLabels();
  if (!shape) return;
  const sampling: SamplingSpec =
    samplingKind() === 'grid'
      ? { kind: 'grid', resolution: +ui.gridN.value }
      : { kind: 'scattered', count: +ui.count.value, seed: Math.max(0, Math.floor(+ui.seed.value || 0)) };
  const options = defaultOptions(DOMAIN.x1 - DOMAIN.x0);
  options.kernel = ui.kernel.value as typeof options.kernel;
  options.iterations = +ui.iters.value;
  options.descent.maxIters = Math.max(1, Math.round(+ui.descentIters.value || 10));
  options.deferIsolated = ui.deferIsolated.checked;
  options.filterInfeasible = ui.filterInfeasible.checked;
  options.monotoneFeasibility = ui.monotone.checked;
  options.clamp.enabled = ui.clamp.checked;
  if (ui.dedup.value !== '' && Number.isFinite(+ui.dedup.value)) options.dedupRadius = Math.max(0, +ui.dedup.value);
  const int = (el: HTMLInputElement, fallback: number, min: number) =>
    Math.max(min, el.value.trim() !== '' && Number.isFinite(+el.value) ? Math.round(+el.value) : fallback);
  options.interpolant.method = ui.usePU.checked ? 'pu' : 'global';
  options.interpolant.pu = {
    overlap: Math.max(0, Number.isFinite(+ui.puOverlap.value) ? +ui.puOverlap.value : 0.25),
    maxLeafPoints: int(ui.puLeaf, 50, 4),
    maxPatchPoints: int(ui.puPatch, 100, 4),
    minPatchPoints: int(ui.puMin, 0, 0),
    rankRepair: ui.puRepair.checked,
  };
  pipeline.run(shape, {
    domain: DOMAIN,
    sampling,
    useRegions: ui.useRegions.checked,
    epsDegen: Math.max(0, +ui.epsDegen.value || 0),
    epsTan: 1e-4,
    options,
  });
}

let runTimer = 0;
function runSoon(delay = 120): void {
  syncControlLabels();
  clearTimeout(runTimer);
  runTimer = window.setTimeout(run, delay);
}

async function loadShape(value: string): Promise<void> {
  try {
    let text = uploaded.get(value);
    if (text === undefined) {
      const r = await fetch(value);
      if (!r.ok) throw new Error(`Could not load ${value}`);
      text = await r.text();
    }
    shape = svgToShape(text);
    run();
  } catch (e) {
    store.update({ error: e instanceof Error ? e.message : String(e) });
  }
}

let lastShape = ui.shape.value;
ui.shape.addEventListener('change', () => {
  if (ui.shape.value === 'upload') {
    ui.shape.value = lastShape;
    ui.upload.click();
    return;
  }
  lastShape = ui.shape.value;
  void loadShape(ui.shape.value);
});
ui.upload.addEventListener('change', async () => {
  const file = ui.upload.files?.[0];
  if (!file) return;
  const key = `uploaded:${uploaded.size}`;
  uploaded.set(key, await file.text());
  const opt = new Option(`Uploaded: ${file.name}`, key);
  ui.shape.insertBefore(opt, ui.shape.querySelector('option[value=upload]'));
  ui.shape.value = key;
  lastShape = key;
  ui.upload.value = '';
  void loadShape(key);
});
for (const r of document.querySelectorAll<HTMLInputElement>('input[name=sampling]')) r.addEventListener('change', () => runSoon(0));
for (const el of [ui.gridN, ui.count, ui.iters]) el.addEventListener('input', () => runSoon());
for (const el of [ui.seed, ui.kernel, ui.descentIters, ui.epsDegen, ui.dedup, ui.useRegions, ui.usePU, ui.puOverlap, ui.puLeaf, ui.puPatch, ui.puMin, ui.puRepair, ui.deferIsolated, ui.filterInfeasible, ui.monotone, ui.clamp]) {
  el.addEventListener('change', () => runSoon(0));
}
ui.reseed.addEventListener('click', () => {
  ui.seed.value = String(Math.floor(Math.random() * 100000));
  runSoon(0);
});

store.subscribe((s) => {
  ui.status.classList.toggle('busy', s.running);
  ui.status.classList.toggle('error', !!s.error);
  if (s.error) ui.status.textContent = `Error: ${s.error}`;
  else if (s.running) ui.status.textContent = s.progress || 'Working…';
  else if (s.samples && s.ms !== null) {
    ui.status.textContent = `${s.samples.values.length} samples · ${s.stages.length} solves · ${(s.ms / 1000).toFixed(2)} s`;
  } else ui.status.textContent = '';
});

// ---------------------------------------------------------------------------
// Figure helpers
// ---------------------------------------------------------------------------
function toggle(toolbar: HTMLElement, label: string, initial: boolean, onChange: () => void): () => boolean {
  const wrap = document.createElement('label');
  wrap.className = 'check';
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = initial;
  box.addEventListener('change', onChange);
  wrap.append(box, label);
  toolbar.appendChild(wrap);
  return () => box.checked;
}

function legend(items: [string, string, string?][]): string {
  return `<div class="legend">${items.map(([c, label, kind]) => `<span><i class="${kind ?? ''}" style="--c:${c}"></i>${label}</span>`).join('')}</div>`;
}

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? v.toFixed(digits) : '—');
/** Error values: fixed-point normally, scientific once they would round to zeros. */
const fmtErr = (v: number) => (!Number.isFinite(v) ? '—' : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(4));

/** " · k PU patches (sizes a–b)" for a partition-of-unity fit, else "". */
function puSummary(model: Interpolant | null | undefined): string {
  if (!model || model.kind !== 'pu') return '';
  const s = model.stats;
  const extra = [s.repaired ? `${s.repaired} rank-repaired` : '', s.skipped ? `${s.skipped} skipped` : ''].filter(Boolean).join(', ');
  return ` · ${s.patches} PU patch${s.patches === 1 ? '' : 'es'} over ${s.constraints} constraints (${s.minSize}–${s.maxSize} each${extra ? `; ${extra}` : ''})`;
}
const patchLegend = (model: Interpolant | null | undefined): [string, string, string][] =>
  model?.kind === 'pu' ? [['#6a5acd', 'PU patch support', 'line']] : [];

function highlightCircle(ctx: CanvasRenderingContext2D, fig: Figure, s: Samples, i: number, width = 2.5): void {
  const d = s.values[i];
  const [x, y] = fig.toScreen(s.points[2 * i], s.points[2 * i + 1]);
  ctx.save();
  ctx.strokeStyle = d >= 0 ? colors.positive : colors.negative;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.arc(x, y, Math.max(1, Math.abs(d) * fig.scale), 0, TWO_PI);
  ctx.stroke();
  ctx.fillStyle = ctx.strokeStyle;
  ctx.beginPath();
  ctx.arc(x, y, 3.5, 0, TWO_PI);
  ctx.fill();
  ctx.restore();
}

function drawTangents(ctx: CanvasRenderingContext2D, fig: Figure, stage: Stage, opts: { onlyConstraints?: boolean; radius?: number } = {}): void {
  const n = stage.status.length;
  for (let i = 0; i < n; i++) {
    const st = stage.status[i];
    const style = statusStyle[st];
    if (!style || (opts.onlyConstraints && !isConstraint(st))) continue;
    const x = stage.tangents[2 * i], y = stage.tangents[2 * i + 1];
    if (!Number.isFinite(x)) continue;
    drawPoint(ctx, fig, x, y, style.color, opts.radius ?? 3, style.hollow);
  }
}

function statusCounts(stage: Stage): string {
  const counts = new Map<number, number>();
  for (const s of stage.status) counts.set(s, (counts.get(s) ?? 0) + 1);
  const parts: string[] = [];
  const names: [number, string][] = [
    [Status.Fixed, 'fixed'], [Status.Projected, 'projected'], [Status.KeptPrevious, 'kept'],
    [Status.Clamped, 'clamped'], [Status.Infeasible, 'culled'], [Status.Deferred, 'deferred'], [Status.Duplicate, 'duplicates'],
  ];
  for (const [code, name] of names) if (counts.get(code)) parts.push(`${counts.get(code)} ${name}`);
  return parts.join(' · ');
}

const tangentLegend = (codes: number[]) =>
  codes.map((c) => [statusStyle[c].color, statusStyle[c].label, statusStyle[c].hollow ? 'hollow' : ''] as [string, string, string]);

/** Index of the projection stage for outer iteration k, or −1. */
function projectionIndex(s: DemoState, k: number): number {
  return s.stages.findIndex((st) => st && st.kind === 'projection' && st.iteration === k);
}

function stageShortLabel(st: Stage): string {
  return st.kind === 'samples' ? '0' : st.kind === 'collapsed' ? 'C' : String(st.iteration);
}

/** Split arcs (which may wrap past 2π) into [0, 2π] intervals. */
function arcBands(arcs: Arc[]): [number, number][] {
  const out: [number, number][] = [];
  for (const a of arcs) {
    let s = a.start % TWO_PI;
    if (s < 0) s += TWO_PI;
    const e = s + (a.end - a.start);
    if (e <= TWO_PI) out.push([s, e]);
    else out.push([s, TWO_PI], [0, e - TWO_PI]);
  }
  return out;
}

const allFigures: Figure[] = [];
const redrawAll = () => allFigures.forEach((f) => f.redraw());

// ---------------------------------------------------------------------------
// §1 Samples
// ---------------------------------------------------------------------------
let hover1 = -1;
const tb1 = $('tb-samples');
const f1Shape = toggle(tb1, 'ground truth', true, () => fig1.redraw());
const f1Circles = toggle(tb1, 'circles', true, () => fig1.redraw());
const f1Fade = toggle(tb1, 'fade large circles', true, () => fig1.redraw());
const fig1 = new Figure($('fig-samples'), {
  home: HOME,
  onDraw: (fig, ctx) => {
    const s = store.state;
    if (s.shape && f1Shape()) drawShape(ctx, fig, s.shape, { fill: colors.shapeFill, stroke: colors.groundTruth, width: 1.5 });
    if (!s.samples) return;
    if (f1Circles()) drawCircles(ctx, fig, s.samples, { fade: f1Fade(), alpha: 0.7 });
    drawSampleDots(ctx, fig, s.samples);
    if (hover1 >= 0 && s.shape) {
      highlightCircle(ctx, fig, s.samples, hover1);
      const px = s.samples.points[2 * hover1], py = s.samples.points[2 * hover1 + 1];
      const cp = closestPointOnShape(s.shape, px, py);
      drawPolyline(ctx, fig, [px, py, cp.x, cp.y], colors.highlight, 1.2);
      drawPoint(ctx, fig, cp.x, cp.y, colors.highlight, 4.5);
    }
  },
  onHover: (w) => {
    const i = w ? pickSample(fig1, store.state.samples, w) : -1;
    if (i !== hover1) {
      hover1 = i;
      fig1.redraw();
      info1();
    }
  },
});
allFigures.push(fig1);

function info1(): void {
  const s = store.state;
  const el = $('info-samples');
  if (!s.samples) {
    el.textContent = '';
    return;
  }
  const n = s.samples.values.length;
  let head = `${n} samples. Hover over a sample to see its circle and its true tangent point (the closest point on the curve).`;
  if (hover1 >= 0) {
    const i = hover1;
    head = `Sample ${i}: <b>x</b> = (${fmt(s.samples.points[2 * i])}, ${fmt(s.samples.points[2 * i + 1])}), <b>d</b> = ${fmtErr(s.samples.values[i])}`;
  }
  el.innerHTML = head + legend([
    [colors.positive, 'd > 0 (outside)', 'ring'], [colors.negative, 'd < 0 (inside)', 'ring'], [colors.highlight, 'true tangent point'],
  ]);
}

// ---------------------------------------------------------------------------
// §2 Power diagram and exposed arcs
// ---------------------------------------------------------------------------
let hover2 = -1;
const tb2 = $('tb-power');
const f2Cells = toggle(tb2, 'power diagram', true, () => fig2.redraw());
const f2Circles = toggle(tb2, 'circles', true, () => fig2.redraw());
const f2Arcs = toggle(tb2, 'exposed arcs', true, () => fig2.redraw());
const f2Shape = toggle(tb2, 'ground truth', false, () => fig2.redraw());

function collapsedStage(s: DemoState): Stage | null {
  return s.stages.find((st) => st && st.kind === 'collapsed') ?? null;
}

const fig2 = new Figure($('fig-power'), {
  home: HOME,
  onDraw: (fig, ctx) => {
    const s = store.state;
    if (s.shape && f2Shape()) drawShape(ctx, fig, s.shape, { stroke: colors.groundTruth, width: 1, dash: [4, 4] });
    if (!s.samples) return;
    const hidden = (i: number) => s.cells[i] === null;
    if (hover2 >= 0 && s.cells[hover2]) {
      ctx.save();
      ctx.fillStyle = 'rgba(224, 138, 0, 0.12)';
      ctx.beginPath();
      const P = s.cells[hover2]!.polygon;
      for (let k = 0; k < P.length; k += 2) {
        const [x, y] = fig.toScreen(P[k], P[k + 1]);
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.fill();
      ctx.restore();
    }
    if (f2Circles()) drawCircles(ctx, fig, s.samples, { alpha: 0.35, fade: true, dash: hidden });
    if (f2Cells()) drawCells(ctx, fig, s.cells);
    drawSampleDots(ctx, fig, s.samples, 2);
    if (f2Arcs()) drawArcs(ctx, fig, s.samples, s.regions, { width: 2.5 });
    // Collapsed regions: fixed tangent points (filled), or candidates that were not used (hollow).
    const cs = collapsedStage(s);
    for (const i of s.collapsed) {
      if (cs && cs.status[i] === Status.Fixed) {
        drawPoint(ctx, fig, cs.tangents[2 * i], cs.tangents[2 * i + 1], statusStyle[Status.Fixed].color, 4);
        continue;
      }
      const r = Math.abs(s.samples.values[i]);
      for (const a of s.regions[i].arcs) {
        const t = (a.start + a.end) / 2;
        drawPoint(ctx, fig, s.samples.points[2 * i] + r * Math.cos(t), s.samples.points[2 * i + 1] + r * Math.sin(t), statusStyle[Status.Fixed].color, 3.5, true);
      }
    }
    if (hover2 >= 0) {
      highlightCircle(ctx, fig, s.samples, hover2, 2);
      drawArcs(ctx, fig, s.samples, s.regions, { width: 5, color: '#000', only: (i) => i === hover2 });
      drawArcs(ctx, fig, s.samples, s.regions, { width: 3, color: colors.arc, only: (i) => i === hover2 });
    }
  },
  onHover: (w) => {
    const i = w ? pickSample(fig2, store.state.samples, w) : -1;
    if (i !== hover2) {
      hover2 = i;
      fig2.redraw();
      info2();
    }
  },
});
allFigures.push(fig2);

function info2(): void {
  const s = store.state;
  const el = $('info-power');
  if (!s.samples || s.regions.length === 0) {
    el.textContent = '';
    return;
  }
  const hidden = s.cells.filter((c) => c === null).length;
  const cs = collapsedStage(s);
  const fixed = cs ? [...cs.status].filter((x) => x === Status.Fixed).length : 0;
  let head = `${hidden} hidden samples · ${s.collapsed.size} collapsed exposed regions · ${fixed} fixed tangent points` +
    (s.usedRegions ? '' : ' <i>(step disabled: not used by the pipeline)</i>');
  if (hover2 >= 0) {
    const i = hover2, reg = s.regions[i];
    head = `Sample ${i}: ${s.cells[i] ? `${new Set(s.cells[i]!.edgeSite).size} cell edges` : '<b>hidden</b> (empty power cell)'} · ` +
      `${reg.arcs.length} exposed arc${reg.arcs.length === 1 ? '' : 's'}, total length ${reg.full ? 'full circle' : reg.length.toExponential(2)}` +
      (s.collapsed.has(i) ? ' · <b>collapsed</b>' : '');
  }
  el.innerHTML = head + legend([
    [colors.powerCell, 'power cells', 'line'], [colors.arc, 'exposed arcs', 'line'],
    [statusStyle[Status.Fixed].color, 'fixed tangent point'], [statusStyle[Status.Fixed].color, 'collapsed but rejected (|D̃₀| too large)', 'hollow'],
  ]);
}

// ---------------------------------------------------------------------------
// §3 Initial RBF
// ---------------------------------------------------------------------------
const tb3 = $('tb-rbf0');
const f3Colors = toggle(tb3, 'field', true, () => fig3.redraw());
const f3Iso = toggle(tb3, 'isolines', true, () => fig3.redraw());
const f3Shape = toggle(tb3, 'ground truth', true, () => fig3.redraw());
const f3Circles = toggle(tb3, 'circles', false, () => fig3.redraw());
const f3Patches = toggle(tb3, 'PU patches', false, () => fig3.redraw());
let layer3: FieldLayer;
const fig3 = new Figure($('fig-rbf0'), {
  home: HOME,
  onDraw: (fig, ctx) => {
    const s = store.state;
    layer3.update(s.stages[0]?.model ?? null);
    layer3.draw(ctx, { showColors: f3Colors(), showIsolines: f3Iso(), showZero: false, zeroColor: colors.levelSet, zeroWidth: 2.5 });
    if (s.shape && f3Shape()) drawShape(ctx, fig, s.shape, { stroke: colors.groundTruth, width: 1.2, dash: [5, 4] });
    if (s.samples && f3Circles()) drawCircles(ctx, fig, s.samples, { alpha: 0.5, fade: true });
    if (f3Patches()) drawPatches(ctx, fig, s.stages[0]?.model);
    layer3.draw(ctx, { showColors: false, showIsolines: false, showZero: true, zeroColor: colors.levelSet, zeroWidth: 2.5 });
    if (s.samples) drawSampleDots(ctx, fig, s.samples, 2);
  },
});
layer3 = new FieldLayer(fig3, fields, 'rbf0');
allFigures.push(fig3);

function info3(): void {
  const s = store.state;
  const m = s.metrics[0];
  $('info-rbf0').innerHTML = (m ? `Samples only: Chamfer distance to ground truth ${fmtErr(m.chamfer)}, mean |D̃| on ground truth ${fmtErr(m.meanAbs)}${puSummary(s.stages[0]?.model)}` : '') +
    legend([...patchLegend(s.stages[0]?.model), [colors.levelSet, 'zero level set of D̃', 'line'], [colors.groundTruth, 'ground truth', 'line dashed'],
      ['rgb(244,165,130)', 'D̃ > 0'], ['rgb(146,197,222)', 'D̃ < 0']]);
}

// ---------------------------------------------------------------------------
// §4 Tangent points by projected descent
// ---------------------------------------------------------------------------
const tb4 = $('tb-descent');
let iter4 = 1;
let step4 = Infinity;
let hover4 = -1;
let selected4 = -1;
let userSelected4 = false;
const iterLabel = document.createElement('label');
iterLabel.innerHTML = 'Iteration <select></select>';
const iterSelect = iterLabel.querySelector('select')!;
tb4.appendChild(iterLabel);
const stepWrap = document.createElement('label');
stepWrap.innerHTML = 'Descent step <input type="range" min="0" max="10" value="10"> <output></output>';
const stepRange = stepWrap.querySelector('input')!;
const stepOut = stepWrap.querySelector('output')!;
tb4.appendChild(stepWrap);
const playBtn4 = document.createElement('button');
playBtn4.type = 'button';
playBtn4.textContent = '▶ Play';
tb4.appendChild(playBtn4);
const f4Colors = toggle(tb4, 'field', true, () => fig4.redraw());
const f4Circles = toggle(tb4, 'circles', true, () => fig4.redraw());
const f4Paths = toggle(tb4, 'descent paths', true, () => fig4.redraw());

iterSelect.addEventListener('change', () => {
  iter4 = +iterSelect.value;
  step4 = Infinity;
  update4();
});
stepRange.addEventListener('input', () => {
  step4 = +stepRange.value >= +stepRange.max ? Infinity : +stepRange.value;
  stepOut.textContent = String(stepRange.value);
  fig4.redraw();
  plot4Update();
});
let anim4 = 0;
playBtn4.addEventListener('click', () => {
  cancelAnimationFrame(anim4);
  const max = +stepRange.max;
  const t0 = performance.now();
  const dur = 350 * Math.max(3, max);
  const tick = (t: number) => {
    const u = Math.min(1, (t - t0) / dur);
    const v = Math.round(u * max);
    stepRange.value = String(v);
    step4 = u >= 1 ? Infinity : v;
    stepOut.textContent = String(v);
    fig4.redraw();
    plot4Update();
    if (u < 1) anim4 = requestAnimationFrame(tick);
  };
  anim4 = requestAnimationFrame(tick);
});

function descentContext(s: DemoState): { stage: Stage; model: Interpolant } | null {
  const idx = projectionIndex(s, iter4);
  if (idx < 1 || !s.stages[idx - 1]) return null;
  return { stage: s.stages[idx], model: s.stages[idx - 1].model };
}

let layer4: FieldLayer;
const fig4 = new Figure($('fig-descent'), {
  home: HOME,
  onDraw: (fig, ctx) => {
    const s = store.state;
    const dc = descentContext(s);
    layer4.update(dc?.model ?? null);
    layer4.draw(ctx, { showColors: f4Colors(), showIsolines: false, showZero: true, zeroColor: colors.levelSet, zeroWidth: 2 });
    if (!s.samples) return;
    if (f4Circles()) drawCircles(ctx, fig, s.samples, { alpha: 0.28, fade: true });
    drawSampleDots(ctx, fig, s.samples, 1.8, 0.8);
    if (!dc) return;
    const { stage } = dc;
    const focus = hover4 >= 0 ? hover4 : selected4;
    const paths = stage.paths ?? [];
    for (let i = 0; i < paths.length; i++) {
      const p = paths[i];
      if (!p) continue;
      const len = p.length / 2;
      const upto = Math.min(len - 1, step4);
      if (f4Paths() && upto > 0) drawPolyline(ctx, fig, p.subarray(0, 2 * (upto + 1)), colors.descent, i === focus ? 2.2 : 1);
      if (upto < len - 1) drawPoint(ctx, fig, p[2 * upto], p[2 * upto + 1], '#333', 2.6);
    }
    // Final points (and fixed ones) once the animation reaches the end of each path.
    for (let i = 0; i < stage.status.length; i++) {
      const p = paths[i];
      if (p && Math.min(p.length / 2 - 1, step4) < p.length / 2 - 1) continue;
      const st = stage.status[i];
      const style = statusStyle[st];
      if (!style || !Number.isFinite(stage.tangents[2 * i])) continue;
      drawPoint(ctx, fig, stage.tangents[2 * i], stage.tangents[2 * i + 1], style.color, i === focus ? 4.5 : 3, style.hollow);
    }
    if (focus >= 0) highlightCircle(ctx, fig, s.samples, focus, 2);
  },
  onHover: (w) => {
    const i = w ? pickSample(fig4, store.state.samples, w) : -1;
    if (i !== hover4) {
      hover4 = i;
      fig4.redraw();
      info4();
      plot4Update();
    }
  },
  onClick: (w) => {
    selected4 = pickSample(fig4, store.state.samples, w);
    userSelected4 = selected4 >= 0;
    fig4.redraw();
    info4();
    plot4Update();
  },
});
layer4 = new FieldLayer(fig4, fields, 'descent');
allFigures.push(fig4);

const plot4 = new Plot($('plot-objective'));

function plot4Update(): void {
  const s = store.state;
  const i = hover4 >= 0 ? hover4 : selected4;
  const dc = descentContext(s);
  const title = $('objective-title');
  if (!s.samples || !dc || i < 0) {
    title.textContent = "Objective around the selected sample's circle (click a sample)";
    plot4.set(null);
    return;
  }
  const d = s.samples.values[i];
  const cx = s.samples.points[2 * i], cy = s.samples.points[2 * i + 1];
  const r = Math.abs(d);
  const N = 361;
  const xs = new Float64Array(N), ys = new Float64Array(N);
  const p = new Float64Array(2);
  for (let k = 0; k < N; k++) {
    const t = (TWO_PI * k) / (N - 1);
    p[0] = cx + r * Math.cos(t);
    p[1] = cy + r * Math.sin(t);
    xs[k] = t;
    ys[k] = evalModel(dc.model, p) / d;
  }
  const angleOf = (x: number, y: number) => {
    let t = Math.atan2(y - cy, x - cx);
    if (t < 0) t += TWO_PI;
    return t;
  };
  const points: { x: number; y: number; color: string; hollow?: boolean }[] = [];
  const path = dc.stage.paths?.[i];
  if (path) {
    const len = path.length / 2;
    const upto = Math.min(len - 1, step4);
    for (let k = 0; k <= upto; k++) {
      p[0] = path[2 * k];
      p[1] = path[2 * k + 1];
      points.push({ x: angleOf(p[0], p[1]), y: evalModel(dc.model, p) / d, color: k === 0 ? '#333' : colors.descent, hollow: k === 0 });
    }
  }
  const st = dc.stage.status[i];
  if (statusStyle[st] && Number.isFinite(dc.stage.tangents[2 * i]) && (!path || step4 >= path.length / 2 - 1)) {
    p[0] = dc.stage.tangents[2 * i];
    p[1] = dc.stage.tangents[2 * i + 1];
    points.push({ x: angleOf(p[0], p[1]), y: evalModel(dc.model, p) / d, color: statusStyle[st].color, hollow: statusStyle[st].hollow });
  }
  const bands = s.usedRegions && s.regions[i]
    ? arcBands(s.regions[i].arcs).map(([a, b]) => ({ x0: a, x1: b, color: 'rgba(224, 138, 0, 0.18)' }))
    : [];
  title.textContent = `Sample ${i}: f(θ) = D̃(point at angle θ on its circle) / d, with d = ${fmtErr(d)}` + (bands.length ? ' (orange: exposed arcs)' : '');
  plot4.set({
    series: [{ xs, ys, color: '#333', width: 1.5 }],
    xRange: [0, TWO_PI],
    bands,
    points,
    zeroLine: true,
    xLabel: 'angle θ around the circle',
    xTicks: [{ x: 0, label: '0' }, { x: Math.PI / 2, label: 'π/2' }, { x: Math.PI, label: 'π' }, { x: 1.5 * Math.PI, label: '3π/2' }, { x: TWO_PI, label: '2π' }],
  });
}

function info4(): void {
  const s = store.state;
  const dc = descentContext(s);
  const el = $('info-descent');
  if (!dc || !s.samples) {
    el.textContent = s.running ? 'Computing…' : '';
    return;
  }
  const i = hover4 >= 0 ? hover4 : selected4;
  let head = `Iteration ${iter4}, against the RBF from the previous stage: ${statusCounts(dc.stage)}`;
  if (i >= 0) {
    const st = dc.stage.status[i];
    const path = dc.stage.paths?.[i];
    const name = statusStyle[st]?.label ?? (st === Status.Deferred ? 'deferred to the next iteration' : st === Status.OnSurface ? 'on the surface' : 'none');
    head = `Sample ${i}: d = ${fmtErr(s.samples.values[i])} · ${name}` + (path ? ` · ${path.length / 2 - 1} descent steps` : '');
  }
  el.innerHTML = head + legend([
    ...tangentLegend([Status.Projected, Status.Fixed, Status.Infeasible, Status.KeptPrevious, Status.Clamped]),
    [colors.descent, 'descent path', 'line'], [colors.levelSet, 'current zero level set', 'line'],
  ]);
}

function update4(): void {
  const s = store.state;
  const iters = s.stages.filter((st) => st && st.kind === 'projection').map((st) => st.iteration);
  const cur = [...iterSelect.options].map((o) => +o.value);
  if (cur.join() !== iters.join()) {
    iterSelect.innerHTML = iters.map((k) => `<option value="${k}">${k}</option>`).join('');
  }
  if (!iters.includes(iter4)) iter4 = iters.length ? Math.min(Math.max(1, iter4), iters[iters.length - 1]) : 1;
  iterSelect.value = String(iter4);
  const dc = descentContext(s);
  let maxLen = 1;
  for (const p of dc?.stage.paths ?? []) if (p) maxLen = Math.max(maxLen, p.length / 2 - 1);
  // Until the user picks a sample, show an interesting one: the accepted projection with the longest descent.
  if (!userSelected4 && dc) {
    let best = -1, bestLen = -1;
    dc.stage.paths?.forEach((p, i) => {
      if (p && dc.stage.status[i] === Status.Projected && p.length > bestLen) {
        bestLen = p.length;
        best = i;
      }
    });
    selected4 = best;
  }
  stepRange.max = String(maxLen);
  if (step4 === Infinity) stepRange.value = String(maxLen);
  stepOut.textContent = stepRange.value;
  fig4.redraw();
  info4();
  plot4Update();
}

// ---------------------------------------------------------------------------
// §5 Refit
// ---------------------------------------------------------------------------
const tb5 = $('tb-refit');
let iter5 = 1;
const iterLabel5 = document.createElement('label');
iterLabel5.innerHTML = 'Iteration <select></select>';
const iterSelect5 = iterLabel5.querySelector('select')!;
tb5.appendChild(iterLabel5);
iterSelect5.addEventListener('change', () => {
  iter5 = +iterSelect5.value;
  update5();
});
const f5Colors = toggle(tb5, 'field', true, () => fig5.redraw());
const f5Iso = toggle(tb5, 'isolines', false, () => fig5.redraw());
const f5Prev = toggle(tb5, 'previous level set', true, () => fig5.redraw());
const f5Shape = toggle(tb5, 'ground truth', true, () => fig5.redraw());
const f5Circles = toggle(tb5, 'circles', false, () => fig5.redraw());
const f5Patches = toggle(tb5, 'PU patches', false, () => fig5.redraw());
let layer5: FieldLayer;
let layer5prev: FieldLayer;
const fig5 = new Figure($('fig-refit'), {
  home: HOME,
  onDraw: (fig, ctx) => {
    const s = store.state;
    const idx = projectionIndex(s, iter5);
    const stage = idx >= 0 ? s.stages[idx] : null;
    layer5.update(stage?.model ?? null);
    layer5prev.update(idx >= 1 && f5Prev() ? s.stages[idx - 1]?.model ?? null : null);
    layer5.draw(ctx, { showColors: f5Colors(), showIsolines: f5Iso(), showZero: false, zeroColor: colors.levelSet, zeroWidth: 2.5 });
    if (s.shape && f5Shape()) drawShape(ctx, fig, s.shape, { stroke: colors.groundTruth, width: 1.2, dash: [5, 4] });
    if (s.samples && f5Circles()) drawCircles(ctx, fig, s.samples, { alpha: 0.4, fade: true });
    if (f5Patches()) drawPatches(ctx, fig, stage?.model);
    if (f5Prev()) layer5prev.draw(ctx, { showColors: false, showIsolines: false, showZero: true, zeroColor: colors.previousLevelSet, zeroWidth: 1.8, zeroDash: [6, 4] });
    layer5.draw(ctx, { showColors: false, showIsolines: false, showZero: true, zeroColor: colors.levelSet, zeroWidth: 2.5 });
    if (s.samples) drawSampleDots(ctx, fig, s.samples, 1.8, 0.8);
    if (stage) drawTangents(ctx, fig, stage, { onlyConstraints: true });
  },
});
layer5 = new FieldLayer(fig5, fields, 'refit');
layer5prev = new FieldLayer(fig5, fields, 'refit-prev');
allFigures.push(fig5);

function info5(): void {
  const s = store.state;
  const idx = projectionIndex(s, iter5);
  const el = $('info-refit');
  if (idx < 1 || !s.metrics[idx]) {
    el.textContent = s.running ? 'Computing…' : '';
    return;
  }
  const before = s.metrics[idx - 1], after = s.metrics[idx];
  const constraints = [...s.stages[idx].status].filter(isConstraint).length;
  const present = new Set(s.stages[idx].status);
  el.innerHTML = `Iteration ${iter5}: ${constraints} tangent points as zero-valued constraints. Chamfer distance to ground truth: ${fmtErr(before.chamfer)} → <b>${fmtErr(after.chamfer)}</b>${puSummary(s.stages[idx].model)}` +
    legend([
      [colors.levelSet, 'new level set', 'line'], [colors.previousLevelSet, `previous (${s.stages[idx - 1].label.toLowerCase()})`, 'line dashed'],
      ...tangentLegend([Status.Projected, Status.KeptPrevious, Status.Clamped, Status.Fixed].filter((c) => c === Status.Projected || present.has(c))),
    ]);
}

function update5(): void {
  const s = store.state;
  const iters = s.stages.filter((st) => st && st.kind === 'projection').map((st) => st.iteration);
  if ([...iterSelect5.options].map((o) => +o.value).join() !== iters.join()) {
    iterSelect5.innerHTML = iters.map((k) => `<option value="${k}">${k}</option>`).join('');
  }
  if (!iters.includes(iter5)) iter5 = iters.length ? Math.min(Math.max(1, iter5), iters[iters.length - 1]) : 1;
  iterSelect5.value = String(iter5);
  fig5.redraw();
  info5();
}

// ---------------------------------------------------------------------------
// §6 Iterate
// ---------------------------------------------------------------------------
const tb6 = $('tb-iterate');
let stage6 = -1; // −1 = follow the latest stage
const stageWrap = document.createElement('label');
stageWrap.innerHTML = 'Stage <input type="range" min="0" max="0" value="0"> <output></output>';
const stageRange = stageWrap.querySelector('input')!;
const stageOut = stageWrap.querySelector('output')!;
tb6.appendChild(stageWrap);
const playBtn6 = document.createElement('button');
playBtn6.type = 'button';
playBtn6.textContent = '▶ Play';
tb6.appendChild(playBtn6);
const f6Colors = toggle(tb6, 'field', true, () => fig6.redraw());
const f6Initial = toggle(tb6, 'initial level set', true, () => fig6.redraw());
const f6Shape = toggle(tb6, 'ground truth', true, () => fig6.redraw());
const f6Tangents = toggle(tb6, 'tangent points', true, () => fig6.redraw());
const f6Circles = toggle(tb6, 'circles', false, () => fig6.redraw());
const f6Patches = toggle(tb6, 'PU patches', false, () => fig6.redraw());

function currentStage6(s: DemoState): number {
  const n = s.stages.length;
  if (n === 0) return -1;
  return stage6 < 0 || stage6 >= n ? n - 1 : stage6;
}

stageRange.addEventListener('input', () => {
  const n = store.state.stages.length;
  stage6 = +stageRange.value >= n - 1 && !store.state.running ? -1 : +stageRange.value;
  update6();
});
let anim6 = 0;
playBtn6.addEventListener('click', () => {
  clearInterval(anim6);
  let k = 0;
  stage6 = 0;
  update6();
  anim6 = window.setInterval(() => {
    const n = store.state.stages.length;
    k++;
    if (k >= n) {
      clearInterval(anim6);
      stage6 = -1;
    } else stage6 = k;
    update6();
  }, 700);
});

let layer6: FieldLayer;
let layer6init: FieldLayer;
const fig6 = new Figure($('fig-iterate'), {
  home: HOME,
  onDraw: (fig, ctx) => {
    const s = store.state;
    const k = currentStage6(s);
    const stage = k >= 0 ? s.stages[k] : null;
    layer6.update(stage?.model ?? null);
    layer6init.update(f6Initial() && k > 0 ? s.stages[0]?.model ?? null : null);
    layer6.draw(ctx, { showColors: f6Colors(), showIsolines: f6Colors(), showZero: false, zeroColor: colors.levelSet, zeroWidth: 2.5 });
    if (s.shape && f6Shape()) drawShape(ctx, fig, s.shape, { fill: f6Colors() ? undefined : colors.shapeFill, stroke: colors.groundTruth, width: 1.2 });
    if (s.samples && f6Circles()) drawCircles(ctx, fig, s.samples, { alpha: 0.4, fade: true });
    if (f6Patches()) drawPatches(ctx, fig, stage?.model);
    if (f6Initial() && k > 0) layer6init.draw(ctx, { showColors: false, showIsolines: false, showZero: true, zeroColor: colors.previousLevelSet, zeroWidth: 1.5, zeroDash: [6, 4] });
    layer6.draw(ctx, { showColors: false, showIsolines: false, showZero: true, zeroColor: colors.levelSet, zeroWidth: 2.5 });
    if (stage && f6Tangents()) drawTangents(ctx, fig, stage, { onlyConstraints: true, radius: 2.6 });
  },
});
layer6 = new FieldLayer(fig6, fields, 'iterate');
layer6init = new FieldLayer(fig6, fields, 'iterate-init');
allFigures.push(fig6);

const plot6 = new Plot($('plot-error'));

function update6(): void {
  const s = store.state;
  const n = s.stages.length;
  const k = currentStage6(s);
  stageRange.max = String(Math.max(0, n - 1));
  stageRange.value = String(Math.max(0, k));
  stageOut.textContent = k >= 0 ? s.stages[k].label : '';
  fig6.redraw();

  const el = $('info-iterate');
  if (k < 0) {
    el.textContent = '';
    plot6.set(null);
    return;
  }
  const m = s.metrics[k];
  const counts = statusCounts(s.stages[k]);
  el.innerHTML = `<b>${s.stages[k].label}</b> · Chamfer ${fmtErr(m?.chamfer ?? NaN)} · mean |D̃| on ground truth ${fmtErr(m?.meanAbs ?? NaN)}${counts ? ` · ${counts}` : ''}${puSummary(s.stages[k].model)}` +
    legend([
      [colors.levelSet, 'zero level set', 'line'], [colors.previousLevelSet, 'initial level set (samples only)', 'line dashed'],
      [colors.groundTruth, 'ground truth', 'line'], ...tangentLegend([Status.Projected, Status.Fixed, Status.Clamped]),
    ]);

  const xs = s.stages.map((_, j) => j);
  plot6.set({
    series: [
      { xs, ys: s.metrics.map((mm) => mm?.chamfer ?? NaN), color: colors.levelSet, width: 2, dots: true },
      { xs, ys: s.metrics.map((mm) => mm?.meanAbs ?? NaN), color: '#555', width: 1.5, dash: [5, 4] },
    ],
    xRange: [0, Math.max(1, n - 1)],
    marks: [{ x: k, color: 'rgba(0,0,0,0.25)' }],
    xTicks: s.stages.map((st, j) => ({ x: j, label: stageShortLabel(st) })),
    xLabel: 'stage (0 = samples only, C = collapsed-region points, k = iteration k)',
    logY: true,
  });
}

// ---------------------------------------------------------------------------
// React to state changes
// ---------------------------------------------------------------------------
let lastSamples: Samples | null = null;
store.subscribe((s) => {
  if (s.samples !== lastSamples) {
    lastSamples = s.samples;
    hover1 = hover2 = hover4 = selected4 = -1;
    userSelected4 = false;
    step4 = Infinity;
  }
  info1();
  info2();
  info3();
  update4();
  update5();
  update6();
  redrawAll();
});

syncControlLabels();
void loadShape(ui.shape.value);
