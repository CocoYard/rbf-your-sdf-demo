/**
 * Load a ground-truth shape from SVG text (browser only: uses the DOM for transforms
 * and curve flattening). Polygons, rects, and straight-segment paths are converted
 * exactly so that sharp corners stay sharp; curved paths, circles, and ellipses are
 * flattened by arc length.
 */

import { cleanShape, normalizeShape, type Shape2D } from '../core/shape2d';

type Loop = number[];

/** Parse a path made only of M/L/H/V/Z commands; returns null if it has curves. */
function parseStraightPath(d: string): Loop[] | null {
  const tokens = d.match(/[a-zA-Z]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?/g);
  if (!tokens) return [];
  const loops: Loop[] = [];
  let cur: Loop = [];
  let x = 0, y = 0, sx = 0, sy = 0;
  let cmd = '';
  let i = 0;
  const num = () => parseFloat(tokens[i++]);
  const flush = () => {
    if (cur.length >= 6) loops.push(cur);
    cur = [];
  };
  while (i < tokens.length) {
    if (/[a-zA-Z]/.test(tokens[i])) cmd = tokens[i++];
    switch (cmd) {
      case 'M': case 'm': {
        flush();
        const nx = num(), ny = num();
        x = cmd === 'm' ? x + nx : nx;
        y = cmd === 'm' ? y + ny : ny;
        sx = x; sy = y;
        cur.push(x, y);
        cmd = cmd === 'm' ? 'l' : 'L'; // implicit lineto for following pairs
        break;
      }
      case 'L': x = num(); y = num(); cur.push(x, y); break;
      case 'l': x += num(); y += num(); cur.push(x, y); break;
      case 'H': x = num(); cur.push(x, y); break;
      case 'h': x += num(); cur.push(x, y); break;
      case 'V': y = num(); cur.push(x, y); break;
      case 'v': y += num(); cur.push(x, y); break;
      case 'Z': case 'z':
        flush();
        x = sx; y = sy;
        cmd = '';
        break;
      default:
        return null;
    }
    if (Number.isNaN(x) || Number.isNaN(y)) return null;
  }
  flush();
  return loops;
}

function sampleGeometry(el: SVGGeometryElement, spacing: number): Loop[] {
  const total = el.getTotalLength();
  if (!(total > 0)) return [];
  const count = Math.min(20000, Math.max(64, Math.ceil(total / spacing)));
  const step = total / count;
  const loops: Loop[] = [];
  let cur: Loop = [];
  let px = NaN, py = NaN;
  for (let k = 0; k < count; k++) {
    const p = el.getPointAtLength(k * step);
    // A moveto has zero length, so a new subpath shows up as a jump between samples.
    if (cur.length && Math.hypot(p.x - px, p.y - py) > 3 * step) {
      loops.push(cur);
      cur = [];
    }
    cur.push(p.x, p.y);
    px = p.x;
    py = p.y;
  }
  loops.push(cur);
  return loops.filter((L) => L.length >= 6);
}

function ellipseLoop(cx: number, cy: number, rx: number, ry: number, segs = 256): Loop {
  const L: Loop = [];
  for (let k = 0; k < segs; k++) {
    const t = (2 * Math.PI * k) / segs;
    L.push(cx + rx * Math.cos(t), cy + ry * Math.sin(t));
  }
  return L;
}

function num(el: Element, attr: string): number {
  return parseFloat(el.getAttribute(attr) ?? '0') || 0;
}

export function svgToShape(text: string, extent = 0.75): Shape2D {
  const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
  const parsed = doc.documentElement;
  if (parsed.nodeName !== 'svg') throw new Error('Not an SVG file.');

  // Transforms (getCTM) and flattening need the SVG to be in the live document.
  const host = document.createElement('div');
  host.style.cssText = 'position:absolute;left:-100000px;top:0;width:1000px;height:1000px;visibility:hidden;overflow:hidden';
  const svg = document.importNode(parsed, true) as unknown as SVGSVGElement;
  svg.setAttribute('width', '1000');
  svg.setAttribute('height', '1000');
  host.appendChild(svg);
  document.body.appendChild(host);

  try {
    const els = svg.querySelectorAll<SVGGraphicsElement>('path, polygon, polyline, rect, circle, ellipse');
    let diag = 0;
    for (const el of els) {
      const b = el.getBBox();
      diag = Math.max(diag, Math.hypot(b.width, b.height));
    }
    const spacing = Math.max(diag, 1e-9) / 1500;
    const loops: Float64Array[] = [];

    for (const el of els) {
      if (el.closest('defs, clipPath, mask, pattern, symbol')) continue;
      let local: Loop[] = [];
      switch (el.tagName.toLowerCase()) {
        case 'polygon':
        case 'polyline': {
          const pts = (el.getAttribute('points') ?? '').trim().split(/[\s,]+/).map(parseFloat);
          local = [pts.filter((v) => !Number.isNaN(v))];
          break;
        }
        case 'rect': {
          const x = num(el, 'x'), y = num(el, 'y'), w = num(el, 'width'), h = num(el, 'height');
          if (el.getAttribute('rx') || el.getAttribute('ry')) local = sampleGeometry(el as SVGGeometryElement, spacing);
          else local = [[x, y, x + w, y, x + w, y + h, x, y + h]];
          break;
        }
        case 'circle': {
          const r = num(el, 'r');
          local = [ellipseLoop(num(el, 'cx'), num(el, 'cy'), r, r)];
          break;
        }
        case 'ellipse':
          local = [ellipseLoop(num(el, 'cx'), num(el, 'cy'), num(el, 'rx'), num(el, 'ry'))];
          break;
        case 'path':
          local = parseStraightPath(el.getAttribute('d') ?? '') ?? sampleGeometry(el as SVGGeometryElement, spacing);
          break;
      }
      const m = el.getCTM();
      for (const L of local) {
        if (L.length < 6) continue;
        const out = new Float64Array(L.length);
        for (let k = 0; k < L.length; k += 2) {
          const x = L[k], y = L[k + 1];
          out[k] = m ? m.a * x + m.c * y + m.e : x;
          out[k + 1] = m ? m.b * x + m.d * y + m.f : y;
        }
        loops.push(out);
      }
    }
    const shape = cleanShape({ loops });
    if (shape.loops.length === 0) throw new Error('No closed shapes found in the SVG.');
    return normalizeShape(shape, extent, true);
  } finally {
    host.remove();
  }
}
