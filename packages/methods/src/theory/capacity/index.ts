/**
 * `aifn-methods/theory/capacity`: shattering by half-planes, rectangles and intervals, enumerated exactly
 * (`realisable`, `shatteringTable`), and the empirical Rademacher complexity of the realised labellings
 * (`empiricalRademacher`).
 *
 * - Shattering: `realisable` tests one $\pm 1$ labelling against a `ShatterClass` (half-planes by a feasibility linear
 *   program, rectangles and intervals by a bounding box); `shatteringTable` tests all $2^n$ of them, which shows the
 *   VC dimension.
 * - Rademacher complexity: `empiricalRademacher` estimates it for a finite set of labellings by Monte Carlo, beside
 *   Massart's bound $\sqrt{2 \ln \lvert \Hcal \rvert / n}$.
 *
 * Points are $n \times 2$ matrices. `capacityFunctions` registers the functions.
 */

export { empiricalRademacher, realisable, shatteringTable, type ShatterClass } from './capacity'
export { capacityFunctions } from './registry'
