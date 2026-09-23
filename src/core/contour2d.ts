/**
 * Marching squares: extract the iso-contour of a scalar field sampled on a regular
 * lattice. Returns line segments as a flat array [ax, ay, bx, by, ...] in world units.
 */

export function marchingSquares(
  values: ArrayLike<number>,
  nx: number,
  ny: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  iso = 0,
): Float64Array {
  const out: number[] = [];
  const sx = (x1 - x0) / (nx - 1);
  const sy = (y1 - y0) / (ny - 1);
  const lerp = (a: number, b: number) => {
    const d = b - a;
    return d === 0 ? 0.5 : (iso - a) / d;
  };

  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const v00 = values[j * nx + i];
      const v10 = values[j * nx + i + 1];
      const v11 = values[(j + 1) * nx + i + 1];
      const v01 = values[(j + 1) * nx + i];
      const c =
        (v00 > iso ? 1 : 0) | (v10 > iso ? 2 : 0) | (v11 > iso ? 4 : 0) | (v01 > iso ? 8 : 0);
      if (c === 0 || c === 15) continue;

      const X = x0 + i * sx;
      const Y = y0 + j * sy;
      // Edge crossing points: bottom (00-10), right (10-11), top (01-11), left (00-01).
      const bx = X + lerp(v00, v10) * sx, by = Y;
      const rx = X + sx, ry = Y + lerp(v10, v11) * sy;
      const tx = X + lerp(v01, v11) * sx, ty = Y + sy;
      const lx = X, ly = Y + lerp(v00, v01) * sy;

      const seg = (ax: number, ay: number, cx: number, cy: number) => out.push(ax, ay, cx, cy);
      switch (c) {
        case 1: case 14: seg(lx, ly, bx, by); break;
        case 2: case 13: seg(bx, by, rx, ry); break;
        case 3: case 12: seg(lx, ly, rx, ry); break;
        case 4: case 11: seg(rx, ry, tx, ty); break;
        case 6: case 9: seg(bx, by, tx, ty); break;
        case 7: case 8: seg(lx, ly, tx, ty); break;
        case 5: case 10: {
          // Saddle: disambiguate with the cell-center value.
          const center = (v00 + v10 + v11 + v01) / 4;
          const centerAbove = center > iso;
          if ((c === 5) === centerAbove) {
            seg(lx, ly, tx, ty);
            seg(bx, by, rx, ry);
          } else {
            seg(lx, ly, bx, by);
            seg(rx, ry, tx, ty);
          }
          break;
        }
      }
    }
  }
  return Float64Array.from(out);
}
