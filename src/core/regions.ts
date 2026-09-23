/**
 * Dimension-independent view of "exposed regions" used by the pipeline. The 2D
 * implementation (arcs on circles, from a power diagram) lives in regions2d.ts; a 3D
 * implementation (patches on spheres bounded by circular arcs) would implement the
 * same interface.
 */

export interface ExposedRegionOracle {
  /**
   * If sample i's exposed region has collapsed to (nearly) a point, the candidate
   * tangent points (k × dim, e.g. midpoints of its tiny arcs); otherwise null.
   */
  collapsedCandidates(i: number): Float64Array | null;
  /** Distance from y (a point on sample i's sphere) to its exposed region, and the closest point in it. */
  distance(i: number, y: Float64Array): { distance: number; closest: Float64Array };
  /** Size of sample i's exposed region (perimeter of the arcs in 2D). */
  size(i: number): number;
}
