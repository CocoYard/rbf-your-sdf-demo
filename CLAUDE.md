Build an interactive, step-by-step web demo of the paper "RBF Your SDF: Radial Basis Function Interpolation of Signed Distance Fields with Implied Tangent Points". The paper lives in `../main.tex`. The steps to show are described below. A fantastic example of this for a different paper is here: <https://clementjambon.github.io/wods/index.html>.

Key points:

* Demonstrate a simplified 2D version of the algorithm.

    1. Given a ground truth closed path (from an SVG for easily changing the example), sample it to create an SDF. Show a circle for each sample whose radius equals the sample's distance value (red for positive, blue for negative). Allow the user to choose whether samples are scattered or on a grid (and the grid dimension or number of samples).
    2. From the SDF samples, create an RBF using a cubic kernel and polynomial term.
    3. For each sample, find the point on its circle with the minimum "RBF function value times the sign of the distance value" using projected gradient descent. This point is called the tangent point.
    4. Re-compute the RBF with the additional tangent points as samples whose value is 0.
    5. Repeat steps 3 and 4, updating the tangent point locations each time.

* As an optional step to be inserted before step 2, compute the power diagram (regular triangulation) for the sample circles. Compute the exposed arcs of each circle. Exposed arcs are the portion of the circle not covered by other circles. This can be computed by clipping each circle against the cell's edges. If only a very small exposed arc remains, then the tangent point of that sample is fixed to the small exposed arc. If a sample disappears in the power diagram, that means it is completely covered by other circles. Find its cell in the power diagram and clip it against all neighbors of that cell. An infinitessimal arc should remain.

* Design the code in a modular fashion so that we could in the future make a standalone web application that runs the algorithm for a user-provided 2D data.

* Architect the code so that we could do a 3D version in the future.

* Should the logic be implemented in TypeScript/JavaScript or Rust via WASM?
