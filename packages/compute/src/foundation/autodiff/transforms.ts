/**
 * The function transforms: `grad`, `valueAndGrad`, `vjp`, `jvp`, `linearize`, `hvp`, `jacobian`, `hessian`, `vmap`
 * and `stopGradient`. They follow the interface of JAX (Bradbury et al., 2018): a transform takes a function and
 * returns a function. Each starts one interpreter (reverse, forward or batch; design K §4.1) at a fresh level, runs the
 * function on tracers of it, and reads the answer off the tracers. Transforms nest in any order, because each
 * interpreter treats tracers of the others as constants.
 */

import { AifnError, DTypeError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  avalOf,
  batchToFront,
  broadcastTo,
  definePrimitive,
  fromData,
  isTraced,
  ones,
  permute,
  reshape,
  slice,
  sum,
  zerosOf,
  type Aval,
  type Tensor,
  type Traced,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { treeFlatten, treeUnflatten, type Flat } from 'aifn-compute/foundation/pytree'
import { BatchInterpreter } from './batch'
import { ForwardInterpreter } from './forward'
import { ReverseInterpreter, type ReverseTracer } from './reverse'

const count = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1)

// ── Types ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A result that may be traced (design K A10). A transform's result is raw when nothing it computes on is traced, but
 * inside another transform (or when f closes over a traced value) it is traced by the enclosing transform. The type
 * says so at every level: a number leaf becomes `number | Traced`, a tensor `Tensor | Traced`, and the structure of
 * arrays and objects is kept. At the outermost level, narrow with `as` or read it through `unwrap`.
 */
export type Lifted<T> = T extends number
  ? number | Traced
  : T extends Tensor
    ? Tensor | Traced
    : T extends Traced
      ? Value
      : T extends readonly unknown[]
        ? { -readonly [K in keyof T]: Lifted<T[K]> }
        : T extends object
          ? { -readonly [K in keyof T]: Lifted<T[K]> }
          : T

/** A tree of the structure of `T` with every leaf replaced by `L`. */
export type TreeOf<T, L> = T extends Value
  ? L
  : T extends readonly unknown[]
    ? { -readonly [K in keyof T]: TreeOf<T[K], L> }
    : T extends object
      ? { -readonly [K in keyof T]: TreeOf<T[K], L> }
      : T

// ── Helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Seed cotangent for a scalar output: 1, or a rank-0 tensor of ones. */
function scalarSeed(where: string, y: Value): Value {
  const aval = avalOf(y)
  if (aval.dtype === 'complex128')
    throw new DTypeError(
      where,
      `${where}: the function must return a real value, not complex128 (differentiate a real loss such as ` +
        `realPart, abs or the squared modulus of the output)`,
      ['complex128'],
    )
  if (aval.number) return 1
  if (aval.shape.length === 0) return ones([])
  throw new ShapeError(
    where,
    `${where}: the function must return a number or a rank-0 tensor, not shape [${aval.shape.join(', ')}]`,
  )
}

type Argnums = number | readonly number[]

function argnumList(argnums: Argnums, n: number, where: string): number[] {
  const list = typeof argnums === 'number' ? [argnums] : [...argnums]
  for (const k of list) {
    if (!Number.isInteger(k) || k < 0 || k >= n)
      throw new AifnError(where, `${where}: argnum ${k} is out of range for ${n} args`)
  }
  // A repeated argnum would trace the argument twice, and the first copy's gradient would read as zero.
  if (new Set(list).size !== list.length) throw new AifnError(where, `${where}: argnums [${list.join(', ')}] repeat`)
  return list
}

/** Make the arguments `argnums` input leaves of a reverse interpreter; returns the new arguments and their trees. */
function traceArgs(rev: ReverseInterpreter, args: readonly unknown[], argnums: readonly number[]) {
  const out = [...args]
  const flats: Flat[] = []
  const inputs: ReverseTracer[] = []
  for (const k of argnums) {
    const flat = treeFlatten(args[k], `arg${k}`)
    const traced = flat.leaves.map((leaf, i) => rev.input(leaf, flat.paths[i]))
    out[k] = treeUnflatten(flat.treedef, traced)
    flats.push(flat)
    inputs.push(...traced)
  }
  return { args: out, flats, inputs }
}

/** Split a flat list of leaves back into one tree per argument. */
function splitLeaves(flats: readonly Flat[], leaves: readonly unknown[]): unknown[] {
  let at = 0
  return flats.map((flat) => {
    const part = leaves.slice(at, at + flat.leaves.length)
    at += flat.leaves.length
    return treeUnflatten(flat.treedef, part)
  })
}

/** Cotangents with the missing ones (outputs independent of that leaf) replaced by zeros of the leaf's kind. */
function filled(cotangents: readonly (Value | null)[], leaves: readonly Value[]): Value[] {
  return cotangents.map((g, i) => g ?? zerosOf(avalOf(leaves[i])))
}

// ── grad and valueAndGrad ────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `grad` and `valueAndGrad`. */
export type GradOptions = {
  /** Which argument(s) to differentiate with respect to (default 0). An array gives an array of gradients. */
  argnums?: Argnums
}

/** The result of `valueAndGrad`: the function's value and its gradient. */
export type ValueAndGrad<V, G> = { value: V; grad: G }

/**
 * `valueAndGrad(f)` is a function computing f(...args) and the gradient of f with respect to argument `argnums`
 * (default 0) in one forward and one reverse pass. f must return a number or a rank-0 tensor. Arguments may be
 * numbers, tensors or pytrees (nested arrays and objects) of them; each gradient has its argument's structure, with
 * numbers for numbers and tensors of the same shape for tensors. With `argnums` an array, `grad` is an array of
 * gradients. Inside another transform the results are traced by it (so nested `grad` gives second derivatives).
 *
 * @example valueAndGrad((w: Value) => sum(square(w)))(tensor([1, 2])) // { value: 5, grad: [2, 4] }
 */
export function valueAndGrad<A extends unknown[]>(
  f: (...args: A) => Value,
): (...args: A) => ValueAndGrad<Value, Lifted<A[0]>>
export function valueAndGrad<A extends unknown[], N extends number>(
  f: (...args: A) => Value,
  options: { argnums: N },
): (...args: A) => ValueAndGrad<Value, Lifted<A[N]>>
export function valueAndGrad<A extends unknown[]>(
  f: (...args: A) => Value,
  options: { argnums: readonly number[] },
): (...args: A) => ValueAndGrad<Value, Lifted<A[number]>[]>
export function valueAndGrad<A extends unknown[]>(
  f: (...args: A) => Value,
  options?: GradOptions,
): (...args: A) => ValueAndGrad<Value, unknown>
export function valueAndGrad<A extends unknown[]>(f: (...args: A) => Value, { argnums = 0 }: GradOptions = {}) {
  return (...args: A) => {
    const nums = argnumList(argnums, args.length, 'grad')
    const rev = new ReverseInterpreter()
    const traced = traceArgs(rev, args, nums)
    const y = f(...(traced.args as A))
    const seed = scalarSeed('grad', y)
    const { cotangents } = rev.backward([y], [seed], traced.inputs)
    const grads = splitLeaves(traced.flats, filled(cotangents, traced.inputs))
    return { value: rev.lower(y), grad: typeof argnums === 'number' ? grads[0] : grads }
  }
}

/**
 * `grad(f)` is the gradient of a scalar function: `grad(f)(...args)` has the structure of argument `argnums`
 * (default 0). See `valueAndGrad` for the conventions. Nest it for higher derivatives: `grad(grad(f))`.
 *
 * @example grad((x: Value) => mul(x, sin(x)))(1) // sin 1 + cos 1
 */
export function grad<A extends unknown[]>(f: (...args: A) => Value): (...args: A) => Lifted<A[0]>
export function grad<A extends unknown[], N extends number>(
  f: (...args: A) => Value,
  options: { argnums: N },
): (...args: A) => Lifted<A[N]>
export function grad<A extends unknown[]>(
  f: (...args: A) => Value,
  options: { argnums: readonly number[] },
): (...args: A) => Lifted<A[number]>[]
export function grad<A extends unknown[]>(f: (...args: A) => Value, options?: GradOptions): (...args: A) => unknown
export function grad<A extends unknown[]>(f: (...args: A) => Value, options: GradOptions = {}) {
  const both = valueAndGrad(f, options)
  return (...args: A) => both(...args).grad
}

// ── vjp, jvp, linearize, hvp ─────────────────────────────────────────────────────────────────────────────────────────

/** The result of `vjp`: f's value and its pullback. */
export type VjpResult<X, Y> = {
  value: Lifted<Y>
  /** Map a cotangent of the output (the structure of `value`) to one of the input (the structure of x). */
  pullback: (cotangent: TreeOf<Y, Value>) => Lifted<X>
}

/**
 * The value of f at x and its pullback u ↦ uᵀJ, where J is the Jacobian of f at x. x and f(x) are numbers, tensors or
 * pytrees of them. One forward pass, recorded; each call of `pullback` is one reverse sweep over the record.
 */
export function vjp<X, Y>(f: (x: X) => Y, x: X): VjpResult<X, Y> {
  const rev = new ReverseInterpreter()
  const traced = traceArgs(rev, [x], [0])
  const y = f(traced.args[0] as X)
  const out = treeFlatten(y)
  return {
    value: treeUnflatten(
      out.treedef,
      out.leaves.map((v) => rev.lower(v)),
    ),
    pullback: (u) => {
      const seeds = treeFlatten(u).leaves
      if (seeds.length !== out.leaves.length)
        throw new ShapeError('vjp', 'vjp: the cotangent must have the structure of the output')
      sameShapes('vjp', 'the cotangent', out.leaves, seeds)
      const { cotangents } = rev.backward(out.leaves, seeds, traced.inputs)
      return splitLeaves(traced.flats, filled(cotangents, traced.inputs))[0] as Lifted<X>
    },
  }
}

/** Throw unless each direction leaf has the shape of its primal leaf (a number never stands for a whole vector). */
function sameShapes(where: string, what: string, primals: readonly Value[], directions: readonly Value[]): void {
  primals.forEach((p, i) => {
    const a = avalOf(p).shape
    const b = avalOf(directions[i]).shape
    if (a.length !== b.length || a.some((d, k) => d !== b[k]))
      throw new ShapeError(where, `${where}: ${what} leaf ${i} has shape [${b.join(', ')}], expected [${a.join(', ')}]`)
  })
}

/** The result of `jvp`: f's value and the directional derivative J·v, both of the structure of f(x). */
export type JvpResult<Y> = { value: Lifted<Y>; tangent: TreeOf<Y, Value> }

/**
 * The value of f at x and the Jacobian–vector product J·v (the derivative of f at x in direction v), where v has the
 * structure of x and the tangent that of f(x). One forward pass with dual numbers: every primitive computes its value
 * and its tangent together.
 */
export function jvp<X, Y>(f: (x: X) => Y, x: X, v: X): JvpResult<Y> {
  const fwd = new ForwardInterpreter()
  const flat = treeFlatten(x)
  const directions = treeFlatten(v).leaves
  if (directions.length !== flat.leaves.length) throw new ShapeError('jvp', 'jvp: v must have the structure of x')
  sameShapes('jvp', 'the tangent', flat.leaves, directions)
  const seeded = flat.leaves.map((leaf, i) => fwd.seed(leaf, directions[i]))
  const y = f(treeUnflatten(flat.treedef, seeded))
  const out = treeFlatten(y)
  return {
    value: treeUnflatten(
      out.treedef,
      out.leaves.map((leaf) => fwd.primalOf(leaf)),
    ),
    tangent: treeUnflatten(
      out.treedef,
      out.leaves.map((leaf) => fwd.tangentOf(leaf) ?? zerosOf(avalOf(leaf))),
    ),
  }
}

/** The result of `linearize`: f's value and the linear map v ↦ J·v at x. */
export type Linearized<X, Y> = { value: Lifted<Y>; jvp: (v: X) => TreeOf<Y, Value> }

/**
 * f's value at x and its linearisation v ↦ J·v, the best linear approximation of f near x. Each call of `jvp` is one
 * forward pass (eager evaluation keeps no staged linear program, so the primal is recomputed alongside).
 */
export function linearize<X, Y>(f: (x: X) => Y, x: X): Linearized<X, Y> {
  const out = treeFlatten(f(x))
  return {
    value: treeUnflatten(out.treedef, out.leaves),
    jvp: (v) => jvp(f, x, v).tangent,
  }
}

/**
 * The Hessian–vector product H·v of a scalar function f at x, where v has the structure of x and so has the result:
 * forward over reverse, the jvp of `grad(f)` in direction v (Pearlmutter, 1994). One forward-mode pass through one
 * gradient evaluation, without forming H.
 */
export function hvp<X>(f: (x: X) => Value, x: X, v: X): TreeOf<X, Value> {
  return jvp((y: X) => grad(f)(y) as unknown, x, v).tangent as TreeOf<X, Value>
}

// ── vmap ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `vmap`. */
export type VmapOptions = {
  /**
   * The batch axis of each argument (applied to every leaf of it), or `null` for an argument shared by every example.
   * One number applies to every argument. Default 0.
   */
  inAxes?: number | null | readonly (number | null)[]
  /** Where the batch axis goes in every output leaf. Default 0. */
  outAxes?: number
}

/**
 * `vmap(f)` runs f, written for one example, on a batch: each argument carries one more axis (`inAxes`, default the
 * first), and every output leaf gains the batch axis at `outAxes`. Each primitive runs once on the whole batch by its
 * batching rule (a loop over examples where it has none). Arguments and outputs may be pytrees. Nest it with the
 * derivative transforms: `vmap(grad(loss))` gives per-example gradients.
 *
 * @example vmap((x: Value) => dot(x, x))(tensor([[1, 2], [3, 4]])) // [5, 25]
 */
export function vmap<A extends unknown[], R>(
  f: (...args: A) => R,
  options: VmapOptions = {},
): (...args: A) => TreeOf<R, Value> {
  const { inAxes = 0, outAxes = 0 } = options
  return (...args: A) => {
    const axes = Array.isArray(inAxes)
      ? (inAxes as readonly (number | null)[])
      : args.map(() => inAxes as number | null)
    if (axes.length !== args.length)
      throw new ShapeError('vmap', `vmap: inAxes has ${axes.length} entries for ${args.length} arguments`)
    let size: number | null = null
    const flats = args.map((a) => treeFlatten(a))
    flats.forEach((flat, k) => {
      const axis = axes[k]
      if (axis === null) return
      for (const leaf of flat.leaves) {
        const shape = avalOf(leaf).shape
        const n = shape[axis < 0 ? shape.length + axis : axis]
        if (n === undefined) throw new ShapeError('vmap', `vmap: argument ${k} has no axis ${axis}`)
        if (size !== null && n !== size) throw new ShapeError('vmap', `vmap: batch sizes differ (${size} and ${n})`)
        size = n
      }
    })
    if (size === null) throw new AifnError('vmap', 'vmap: no argument is batched')
    const batch = new BatchInterpreter(size)
    const wrapped = args.map((a, k) => {
      const axis = axes[k]
      if (axis === null) return a
      const flat = flats[k]
      return treeUnflatten(
        flat.treedef,
        flat.leaves.map((leaf) => {
          const rank = avalOf(leaf).shape.length
          return batch.wrap(leaf, axis < 0 ? rank + axis : axis)
        }),
      )
    })
    const out = treeFlatten(f(...(wrapped as A)))
    const n = size
    const leaves = out.leaves.map((leaf) => {
      // An output that does not depend on the batch is the same for every example: broadcast it.
      const front = batch.owns(leaf)
        ? batchToFront(leaf.value, leaf.axis)
        : broadcastTo(leaf, [n, ...avalOf(leaf).shape])
      return moveFront(front, outAxes)
    })
    return treeUnflatten(out.treedef, leaves)
  }
}

/** Move axis 0 of `v` to position `to`. */
function moveFront(v: Value, to: number): Value {
  const rank = avalOf(v).shape.length
  const at = to < 0 ? rank + to : to
  if (at === 0) return v
  const order = Array.from({ length: rank - 1 }, (_, k) => k + 1)
  order.splice(at, 0, 0)
  return permute(v, order)
}

// ── jacobian and hessian ─────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `jacobian`. */
export type JacobianOptions = {
  /**
   * `reverse`: one pullback per output element, batched by `vmap` (suits few outputs); `forward`: one jvp per input
   * element, batched by `vmap` (suits few inputs); `auto` (default): forward when x has no more elements than f(x).
   */
  mode?: 'auto' | 'forward' | 'reverse'
}

/**
 * The Jacobian of f at x. For a single input and output leaf it is a tensor of shape [...shape of f(x), ...shape of
 * x] with entry [i, j] = ∂fᵢ/∂xⱼ (a number when both are numbers). For pytrees it is a tree of the structure of f(x)
 * whose leaves are trees of the structure of x (JAX's convention).
 */
export function jacobian<X, Y>(f: (x: X) => Y, options: JacobianOptions = {}): (x: X) => TreeOf<Y, TreeOf<X, Value>> {
  const { mode = 'auto' } = options
  return (x: X) => {
    const inFlat = treeFlatten(x)
    const inAvals = inFlat.leaves.map(avalOf)
    const n = inAvals.reduce((s, a) => s + count(a.shape), 0)
    if (mode === 'forward') return forwardJacobian(f, x, inFlat, inAvals, n)
    if (mode === 'auto') {
      // The output size decides; read it from a plain evaluation, which records no tape for the forward case.
      const m0 = treeFlatten(f(x)).leaves.reduce((s: number, v) => s + count(avalOf(v as Value).shape), 0)
      if (n <= m0) return forwardJacobian(f, x, inFlat, inAvals, n)
    }
    const { value, pullback } = vjp(f, x)
    const outFlat = treeFlatten(value)
    const outAvals = outFlat.leaves.map(avalOf)
    const m = outAvals.reduce((s, a) => s + count(a.shape), 0)
    // Row block of output leaf j: the pullbacks of the basis cotangents of its elements, all at once under vmap.
    const basis = outAvals.map((_, j) => basisBlock(outAvals, j, m))
    const rows = vmap((u: unknown) => pullback(u as TreeOf<Y, Value>))(treeUnflatten(outFlat.treedef, basis))
    const rowLeaves = treeFlatten(rows).leaves
    let off = 0
    const blocks = outAvals.map((o) => {
      const size = count(o.shape)
      const entries = rowLeaves.map((r, i) => entry(slice(r, [off, off + size]), o, inAvals[i], false))
      off += size
      return treeUnflatten(inFlat.treedef, entries)
    })
    return treeUnflatten(outFlat.treedef, blocks) as TreeOf<Y, TreeOf<X, Value>>
  }
}

function forwardJacobian<X, Y>(f: (x: X) => Y, x: X, inFlat: Flat, inAvals: readonly Aval[], n: number) {
  // Column block of input leaf i: the jvps along the basis tangents of its elements, all at once under vmap.
  const basis = inAvals.map((_, i) => basisBlock(inAvals, i, n))
  // One example's output avals, read inside the batch (they say which outputs are numbers).
  let outAvals: Aval[] = []
  const cols = vmap((t: unknown) => {
    const tangent = jvp(f, x, t as X).tangent
    outAvals = treeFlatten(tangent).leaves.map(avalOf)
    return tangent
  })(treeUnflatten(inFlat.treedef, basis))
  const colFlat = treeFlatten(cols)
  const blocks = colFlat.leaves.map((c, j) => {
    const out = outAvals[j]
    let off = 0
    const entries = inAvals.map((a) => {
      const size = count(a.shape)
      const e = entry(slice(c, [off, off + size]), a, out, true)
      off += size
      return e
    })
    return treeUnflatten(inFlat.treedef, entries)
  })
  return treeUnflatten(colFlat.treedef, blocks) as TreeOf<Y, TreeOf<X, Value>>
}

/**
 * The block of basis vectors for leaf `j` of a flat list of leaves with `total` elements: shape [total, ...shape of
 * leaf j], one-hot in the rows of leaf j's own elements.
 */
function basisBlock(avals: readonly Aval[], j: number, total: number): Tensor {
  let off = 0
  for (let k = 0; k < j; k++) off += count(avals[k].shape)
  const size = count(avals[j].shape)
  const data = new Float64Array(total * size)
  for (let r = 0; r < size; r++) data[(off + r) * size + r] = 1
  return fromData(data, [total, ...avals[j].shape])
}

/**
 * One Jacobian entry from a block of rows [size of `lead`, ...shape of `trail`] (reverse mode) or of columns [size
 * of `lead`, ...shape of `trail`] to be transposed (forward mode): the result has shape [...out, ...in], and is a
 * number when both the output and the input are numbers.
 */
function entry(block: Value, lead: Aval, trail: Aval, transposed: boolean): Value {
  const [out, inp] = transposed ? [trail, lead] : [lead, trail]
  let e = block
  if (transposed) e = permute(reshape(e, [count(lead.shape), count(trail.shape)]), [1, 0])
  if (out.number && inp.number) return sum(e)
  return reshape(e, [...out.shape, ...inp.shape])
}

/**
 * The Hessian of a scalar function f at x: forward over reverse (the forward-mode Jacobian of the gradient). For a
 * single leaf x it has shape [...shape of x, ...shape of x]; for a pytree it is a tree of trees.
 */
export function hessian<X>(f: (x: X) => Value): (x: X) => TreeOf<X, TreeOf<X, Value>> {
  return jacobian((y: X) => grad(f)(y) as unknown as X, { mode: 'forward' }) as (x: X) => TreeOf<X, TreeOf<X, Value>>
}

// ── stopGradient ─────────────────────────────────────────────────────────────────────────────────────────────────────

const stopGradientOp = definePrimitive<undefined>({
  id: 'foundation/autodiff/stopGradient',
  arity: 1,
  impl: ([x]) => x,
  zeroDerivative: true,
  shape: ([x]) => x,
  batch: ([x], [axis]) => [x, axis ?? 0],
  differentiable: [false],
  doc: { summary: 'x itself, treated as a constant by every derivative transform.' },
  test: { cases: (draw) => [{ inputs: [draw([2, 3])] }] },
})

/** x itself, treated as a constant by every derivative transform: its derivative is zero. Applies to each leaf. */
export function stopGradient<T>(x: T): T {
  const flat = treeFlatten(x)
  return treeUnflatten(
    flat.treedef,
    flat.leaves.map((leaf) => (isTraced(leaf) ? stopGradientOp([leaf], undefined) : leaf)),
  ) as T
}
