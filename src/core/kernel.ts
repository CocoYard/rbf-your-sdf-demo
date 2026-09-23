/**
 * Radial kernels φ(r). The paper uses the cubic polyharmonic kernel φ(r) = r³ with
 * a degree-1 polynomial term; a few alternatives are provided for comparison.
 * Kernels are referenced by name so RBF models stay plain serializable data.
 */

export type KernelName = 'cubic' | 'thinplate' | 'linear';

export interface Kernel {
  name: KernelName;
  label: string;
  phi(r: number): number;
  /** φ'(r) / r, so that ∇_x φ(|x − c|) = (φ'(r)/r) (x − c). Must be finite at r = 0. */
  dphiOverR(r: number): number;
}

export const kernels: Record<KernelName, Kernel> = {
  cubic: {
    name: 'cubic',
    label: 'cubic  r³',
    phi: (r) => r * r * r,
    dphiOverR: (r) => 3 * r,
  },
  thinplate: {
    name: 'thinplate',
    label: 'thin plate  r² log r',
    phi: (r) => (r > 0 ? r * r * Math.log(r) : 0),
    // The true limit diverges logarithmically but is multiplied by (x − c) → 0.
    dphiOverR: (r) => (r > 0 ? 2 * Math.log(r) + 1 : 0),
  },
  linear: {
    name: 'linear',
    label: 'linear  r',
    phi: (r) => r,
    dphiOverR: (r) => (r > 0 ? 1 / r : 0),
  },
};
