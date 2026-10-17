/**
 * `aifn-compute/nn/equivariant`: layers that commute with a group of transformations of their input.
 *
 * - Rotations by quarter turns: `rot90` (as `numpy.rot90` on the last two axes) and `p4Act`, the action of a rotation
 *   on a $p4$ feature map `[N, C, 4, H, W]` (each slice rotated, the orientations shifted).
 * - Group convolution on $p4$ (Cohen and Welling 2016): the lifting correlation `p4Lift` (image to $p4$ map), the
 *   group correlation `p4GroupConv` ($p4$ map to $p4$ map) and `orientationPool`, the invariant maximum over
 *   orientations.
 *
 * Everything is built on `conv2d` and `gather`, so it is differentiable and batched. The correlations are exactly
 * equivariant on square images with symmetric padding. Shapes that do not fit throw `ShapeError`.
 */

export { orientationPool, p4Act, p4GroupConv, p4Lift, rot90, type P4Options } from './p4'
export { equivariantFunctions } from './registry'
