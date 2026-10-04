/**
 * Private helpers of the samplers: evaluation of a target's log-density and gradient, per-coordinate options and
 * standard normal draws as working arrays. The dense working-array helpers themselves (`vec`, `mat`, `data`, `dot`,
 * `allFinite`) are the one definition in `aifn-compute/foundation/tensor`'s `dense`. Every helper returns new arrays.
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { normals, type Stream } from 'aifn-compute/foundation/random'
import { dense, item, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { LogDensity, VectorLike } from './types'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** A float64 working array. */
export type F64 = dense.F64
/** The dense working-array helpers of `aifn-compute/foundation/tensor` (one definition), under their own names. */
export const { allFinite, data, dot, mat, vec } = dense

/** A copy of a vector argument; a number is a vector of length 1. */
export function toF64(v: VectorLike | number, where: string): F64 {
  return typeof v === 'number' ? Float64Array.of(v) : dense.toF64(v, where)
}

/** A number from a number, a one-element tensor or a traced value. */
export function toNumber(v: Value): number {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : item(raw)
}

/** The target's log-density at x as a number (−Infinity outside the support; NaN is reported as NaN). */
export function logDensityAt(target: LogDensity, x: F64): number {
  return toNumber(target.logDensity(dense.vec(x)))
}

/**
 * The log-density and its gradient at x: `target.grad` when given, otherwise reverse-mode autodiff of
 * `target.logDensity` (which must then be written with aifn primitives).
 */
export function logDensityAndGrad(target: LogDensity, x: F64): { value: number; grad: F64 } {
  if (target.grad) {
    const value = logDensityAt(target, x)
    const g = toF64(target.grad(dense.vec(x)) as Tensor, 'mcmc: target.grad')
    if (g.length !== x.length)
      throw new ShapeError('mcmc', `mcmc: target.grad returned ${g.length} values for dimension ${x.length}`)
    return { value, grad: g }
  }
  const { value, grad } = valueAndGrad((theta: Tensor) => target.logDensity(theta) as Value)(dense.vec(x))
  return { value: toNumber(value as Value), grad: toF64(grad as Tensor, 'mcmc: autodiff gradient') }
}

/** n standard normal draws as a working array. */
export const standardNormals = (s: Stream, n: number): F64 => dense.data(normals(s, n))

/** Per-coordinate values from a number or an array of length n. */
export function perCoordinate(v: number | ArrayLike<number>, n: number, where: string): F64 {
  if (typeof v === 'number') return new Float64Array(n).fill(v)
  if (v.length !== n) throw new ShapeError(where, `${where}: expected ${n} values, got ${v.length}`)
  return Float64Array.from(v)
}

/** Stack equal-length rows into an m×n tensor. */
export function stackRows(rows: readonly ArrayLike<number>[], n: number): Tensor {
  const out = new Float64Array(rows.length * n)
  rows.forEach((r, i) => out.set(r, i * n))
  return dense.mat(out, rows.length, n)
}
