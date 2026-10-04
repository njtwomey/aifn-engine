/**
 * The generated primitive suite (design K §10.1): one block per primitive in `registry.list()`, with inputs drawn from
 * the primitive's declared domains (elementwise) or cases (general) on a stream keyed by its id, so every failure is
 * reproducible. Per primitive and case:
 *
 * - values: elementwise results equal the scalar rule applied element by element (numbers and rank-0 tensors), and
 *   contiguous, transposed and sliced layouts agree;
 * - broadcasting over [] [3] [2,3] [2,1] equals the explicitly broadcast inputs;
 * - dtypes: bool, int32, float32, float64 and (where declared) complex128 inputs give the dtype of the primitive's
 *   rule, and complex inputs agree with the complex scalar loop;
 * - the vjp and the jvp against central differences, in every differentiable input (complex inputs perturbed in their
 *   real and imaginary parts separately, the ℝ² convention);
 * - the dot-product test ⟨u, J v⟩ = ⟨Jᵀ u, v⟩ between the jvp and the vjp;
 * - `vmap` against a loop over examples (every input batched on axis 0; the first input batched on its last axis with
 *   the others shared);
 * - for a primitive with its own batching rule, `vmap` of its vjp and the vjp and jvp of its `vmap`, against loops;
 * - the shape rule against the output of `impl`;
 * - second order (the gradient of ⟨u, ∇loss⟩ against central differences) where the primitive declares it;
 * - inputs unchanged after every call.
 *
 * Checks that fail today are listed in `KNOWN` with a reason and run as expected failures (`it.fails`), so the suite
 * is green, a fix is noticed (the expected failure then fails), and the gaps are printed at the end with the rules
 * each primitive lacks (a missing jvp falls back to the transpose trick, a missing batch rule to a loop).
 */
import fs from 'node:fs'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { grad, jvp, vjp, vmap } from 'aifn-compute/foundation/autodiff'
import { NotDifferentiableError } from 'aifn-compute/foundation/errors'
import { stream, units } from 'aifn-compute/foundation/random'
import {
  add,
  allclose,
  astype,
  avalOf,
  broadcastShapes,
  broadcastTo,
  complex,
  conj,
  copy,
  fromData,
  isTensor,
  mul,
  permute,
  promoteTypes,
  realPart,
  registry,
  reshape,
  resultType,
  scalar,
  shapeOfValue,
  slice,
  stack,
  sum,
  toFlat,
  transpose,
  unwrap,
  zeros,
  type Domain,
  type Draw,
  type Primitive,
  type PrimitiveCase,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { DType } from 'aifn-compute/foundation/contracts'

// Load every compute module (the modules of the tree in modules.json) so that the registry holds all of compute's
// primitives.
type Tree = { compute: { families: { family: string; modules: { module: string; status?: string }[] }[] } }
const spec = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, '../../../modules.json'), 'utf8')) as Tree
const computeModules = spec.compute.families.flatMap((f) =>
  f.modules.length
    ? f.modules.filter((m) => m.status !== 'gap').map((m) => `aifn-compute/${f.family}/${m.module}`)
    : [`aifn-compute/${f.family}`],
)
await Promise.all(computeModules.map((m) => import(m)))

/** Checks that fail today, by primitive id and check (the complex variant of a check is `<check> (complex)`). */
const KNOWN: Readonly<Record<string, Partial<Record<string, string>>>> = {}

const SHAPES: readonly (readonly number[])[] = [[], [3], [2, 3], [2, 1]]
const DEFAULT: Domain = { lo: -2, hi: 2 }
const UNIT: Domain = { lo: -1, hi: 1 }
const EXAMPLES = 3

// ── Drawing inputs ───────────────────────────────────────────────────────────────────────────────────────────────────

function drawFor(id: string): Draw {
  const s = stream(`primitives/${id}`)
  return (shape, { lo, hi, integer } = DEFAULT) => {
    const n = shape.reduce((a, b) => a * b, 1)
    const u = units(s, n)
    const data = new Float64Array(n)
    for (let k = 0; k < n; k++)
      data[k] = integer ? Math.floor(lo + (Math.floor(hi) - lo + 1) * u[k]) : lo + (hi - lo) * u[k]
    return fromData(data, shape)
  }
}

const isComplexValue = (x: Value) => avalOf(x).dtype === 'complex128'
const isFloatInput = (x: Raw) => typeof x === 'number' || x.dtype === 'float64' || x.dtype === 'float32'

/** The same case with every differentiable floating input made complex (imaginary parts in [−1, 1]). */
function complexCase(p: Primitive, c: PrimitiveCase, draw: Draw): PrimitiveCase {
  return {
    inputs: c.inputs.map((x, i) =>
      isDiff(p, i) && typeof x !== 'number' && isFloatInput(x) ? (complex(x, draw([...x.shape], UNIT)) as Tensor) : x,
    ),
    params: c.params,
  }
}

const domainOf = (p: Primitive, i: number): Domain => {
  const d = p.test.domain
  return (Array.isArray(d) ? (d as Domain[])[i] : (d as Domain | undefined)) ?? DEFAULT
}
const isDiff = (p: Primitive, i: number) =>
  Array.isArray(p.differentiable) ? (p.differentiable[i] ?? true) : p.differentiable !== false

// ── Comparing values ─────────────────────────────────────────────────────────────────────────────────────────────────

/** The real coordinates of a value: complex entries interleaved (re, im), the ℝ² view. */
const flat = (x: Value): number[] => {
  const r = unwrap(x)
  return typeof r === 'number' ? [r] : toFlat(r)
}
function same(a: Value, b: Value, rtol = 1e-12, atol = 1e-12): void {
  expect(shapeOfValue(a)).toEqual(shapeOfValue(b))
  const [x, y] = [unwrap(a), unwrap(b)]
  if (typeof x !== 'number' && typeof y !== 'number') expect(x.dtype).toBe(y.dtype)
  expect(allclose(x, y, { rtol, atol, equalNan: true })).toBe(true)
}
/** Central-difference agreement: |a − n| ≤ atol·scale + rtol·|n|, with scale the largest |n| (at least 1). */
function close(analytic: readonly number[], numeric: readonly number[], rtol: number): void {
  expect(analytic.length).toBe(numeric.length)
  const scale = Math.max(1, ...numeric.map(Math.abs))
  numeric.forEach((n, k) => expect(Math.abs(analytic[k] - n)).toBeLessThanOrEqual(1e-6 * scale + rtol * Math.abs(n)))
}
/** The ℝ² inner product Re Σ conj(a)·b of two values of one kind, from their real coordinates. */
const inner = (a: Value, b: Value): number => {
  const [x, y] = [flat(a), flat(b)]
  expect(x.length).toBe(y.length)
  return x.reduce((acc, v, k) => acc + v * y[k], 0)
}
/** The same inner product, traced: Re Σ conj(w)·y. */
const tracedInner = (w: Value, y: Value): Value =>
  isComplexValue(y) || isComplexValue(w) ? sum(realPart(mul(conj(w), y))) : sum(mul(w, y))

/** A value of x's kind with every real coordinate moved by `d` times the direction's. */
const along = (x: Raw, v: Raw, d: number): Raw =>
  typeof x === 'number' ? x + d * (v as number) : (add(x, mul(d, v)) as Tensor)

/** A random value of x's kind (number, real tensor or complex tensor), for directions and cotangents. */
function like(x: Value, draw: Draw, fallback = 1.3): Raw {
  const a = avalOf(x)
  if (a.number) return fallback
  const re = draw([...a.shape], UNIT)
  return a.dtype === 'complex128' ? (complex(re, draw([...a.shape], UNIT)) as Tensor) : re
}

/** Element k (row-major) of a tensor, as a rank-0 tensor of its dtype. */
const elementAt = (t: Tensor, k: number): Tensor =>
  reshape(slice(reshape(t, [t.shape.reduce((a, b) => a * b, 1)]), [k, k + 1]), []) as Tensor

/** The same values as `t` in a non-contiguous layout: a transpose of the reversed layout. */
function transposed(t: Raw): Raw {
  if (typeof t === 'number' || t.shape.length < 2) return t
  return transpose(copy(transpose(t)))
}
/** The same values as `t` read with stride 2 along the last axis. */
function sliced(t: Raw): Raw {
  if (typeof t === 'number' || t.shape.length === 0) return t
  const s = t.shape
  const wide = reshape(stack([t, zeros(s, t.dtype)], -1), [...s.slice(0, -1), 2 * s[s.length - 1]])
  return slice(wide, ...s.slice(0, -1).map(() => null), [0, null, 2])
}

/** Apply, checking that the inputs are unchanged by the call. */
function run(p: Primitive, inputs: readonly Raw[], params: unknown): Raw {
  const before = inputs.map((x) => (typeof x === 'number' ? x : Float64Array.from(x.data)))
  const out = unwrap(p.apply(inputs, params))
  inputs.forEach((x, i) => {
    if (typeof x !== 'number') expect(Float64Array.from(x.data)).toEqual(before[i])
  })
  return out
}

// ── Checks ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** f restricted to input i, the others held at the case's values. */
const partial = (p: Primitive, c: PrimitiveCase, i: number) => (x: Value) =>
  p.apply(
    c.inputs.map((v, j) => (j === i ? x : v)),
    c.params,
  )

/** Real outputs of a floating dtype (or complex) carry derivatives; bool and index outputs do not. */
const differentiableOutput = (y: Value) => {
  const a = avalOf(y)
  return a.number || a.dtype === 'float64' || a.dtype === 'float32' || a.dtype === 'complex128'
}

/** The vjp of ⟨w, f⟩ against central differences in every differentiable input, and second order if asked. */
function checkVjp(p: Primitive, c: PrimitiveCase, draw: Draw, order: 1 | 2): void {
  const out = run(p, c.inputs, c.params)
  const w = like(out, draw, 0.7)
  const rtol = p.test.rtol ?? 1e-5
  c.inputs.forEach((x0, i) => {
    if (!isDiff(p, i)) return
    const loss = (x: Value) => tracedInner(w, partial(p, c, i)(x))
    const u = like(x0, draw)
    // Order 1 differentiates the loss; order 2 differentiates ⟨u, ∇loss⟩, so its gradient is a Hessian–vector product.
    const f = order === 1 ? loss : (x: Value) => tracedInner(u, grad(loss)(x) as Value)
    const analytic = flat(grad(f)(x0) as Value)
    const base = flat(x0)
    const rebuild = (e: number[]): Raw =>
      typeof x0 === 'number'
        ? e[0]
        : fromData(Float64Array.from(e), x0.shape, x0.dtype === 'complex128' ? 'complex128' : undefined)
    const numeric = base.map((v, k) => {
      const h = 1e-5 * Math.max(1, Math.abs(v))
      const at = (d: number) => {
        const e = base.slice()
        e[k] = v + d
        return flat(f(rebuild(e)))[0]
      }
      return (at(h) - at(-h)) / (2 * h)
    })
    close(analytic, numeric, rtol)
  })
}

/** The jvp in each differentiable input against central differences along a random direction. */
function checkJvp(p: Primitive, c: PrimitiveCase, draw: Draw): void {
  const rtol = p.test.rtol ?? 1e-5
  c.inputs.forEach((x0, i) => {
    if (!isDiff(p, i)) return
    const f = partial(p, c, i)
    const v = like(x0, draw)
    const tangent = flat(jvp(f, x0, v).tangent as Value)
    const scale = Math.max(1, ...flat(x0).map(Math.abs))
    const h = 1e-5 * scale
    const plus = flat(f(along(x0, v, h)))
    const minus = flat(f(along(x0, v, -h)))
    close(
      tangent,
      plus.map((a, k) => (a - minus[k]) / (2 * h)),
      rtol,
    )
  })
}

/** ⟨u, J v⟩ = ⟨Jᵀ u, v⟩ in each differentiable input, with J v from `jvp` and Jᵀ u from `vjp`. */
function checkDot(p: Primitive, c: PrimitiveCase, draw: Draw): void {
  const out = run(p, c.inputs, c.params)
  c.inputs.forEach((x0, i) => {
    if (!isDiff(p, i)) return
    const f = partial(p, c, i)
    const v = like(x0, draw)
    const u = like(out, draw, 0.7)
    const left = inner(u, jvp(f, x0, v).tangent as Value)
    const right = inner(vjp(f, x0).pullback(u as never) as Value, v)
    expect(Math.abs(left - right)).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(left)))
  })
}

/** Stack examples along a new first axis (numbers become rank-0 tensors). */
const stackExamples = (xs: readonly Value[]): Tensor =>
  stack(xs.map((x) => (typeof x === 'number' ? scalar(x) : x)) as Tensor[], 0) as Tensor

/** Parameters equal between cases (so examples drawn separately can share one call's parameters). */
function sameParams(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (isTensor(a) && isTensor(b)) return allclose(a, b, { rtol: 0, atol: 0, equalNan: true })
  if (ArrayBuffer.isView(a) && ArrayBuffer.isView(b))
    return Array.from(a as unknown as ArrayLike<number>).every((v, k) =>
      Object.is(v, (b as unknown as ArrayLike<number>)[k]),
    )
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  const ka = Object.keys(a)
  if (ka.length !== Object.keys(b).length) return false
  return ka.every((k) => sameParams((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]))
}

/**
 * `vmap` against a loop over examples. `examples[e]` is one case's inputs; every input is batched on axis 0, or with
 * `last` the first input on its last axis and the rest shared from example 0.
 */
function checkVmap(p: Primitive, examples: readonly PrimitiveCase[], last: boolean): void {
  const params = examples[0].params
  const n = examples[0].inputs.length
  const batchedInputs = last ? [0] : Array.from({ length: n }, (_, i) => i)
  const inputs = examples[0].inputs.map((x, i) => {
    if (!batchedInputs.includes(i)) return x
    const b = stackExamples(examples.map((e) => e.inputs[i]))
    return last ? permute(b, [...Array.from({ length: b.shape.length - 1 }, (_, k) => k + 1), 0]) : b
  })
  const inAxes = examples[0].inputs.map((_x, i) =>
    batchedInputs.includes(i) ? (last ? shapeOfValue(inputs[i]).length - 1 : 0) : null,
  )
  const batched = vmap((...xs: Value[]) => p.apply(xs, params), { inAxes })(...inputs) as Value
  const loop = examples.map((e) =>
    run(
      p,
      examples[0].inputs.map((x, i) => (batchedInputs.includes(i) ? e.inputs[i] : x)),
      params,
    ),
  )
  same(batched, stackExamples(loop), 1e-10, 1e-10)
}

/** The derivative checks through a primitive's own batching rule, each against a loop over the examples. */
type BatchedDerivative = 'vmap of vjp' | 'vjp of vmap' | 'jvp of vmap'

/**
 * Derivatives and batching together, against a loop over examples: `vmap` of the vjp (per-example gradients: the
 * batching rules of the primitive's derivative rules), the vjp of `vmap` and the jvp of `vmap` (the derivative rules of
 * the batched primitive). Every input is batched on axis 0; the differentiable ones are differentiated.
 */
function checkBatchedDerivative(p: Primitive, examples: readonly PrimitiveCase[], draw: Draw, mode: BatchedDerivative) {
  const params = examples[0].params
  const n = examples[0].inputs.length
  const diff = Array.from({ length: n }, (_, i) => i).filter((i) => isDiff(p, i))
  // The inputs with the differentiable ones replaced by `ys`.
  const merge = (xs: readonly Value[], ys: readonly Value[]) =>
    xs.map((x, i) => (diff.includes(i) ? ys[diff.indexOf(i)] : x))
  const f = (xs: readonly Value[]) => (ys: Value[]) => p.apply(merge(xs, ys), params)
  const pick = (xs: readonly Value[]) => diff.map((i) => xs[i])
  const cts = examples.map((e) => like(run(p, e.inputs, params), draw, 0.7))
  const dirs = examples.map((e) => pick(e.inputs).map((x) => like(x, draw)))
  const stacked = Array.from({ length: n }, (_, i) => stackExamples(examples.map((e) => e.inputs[i])))
  const stackedDirs = diff.map((_, j) => stackExamples(dirs.map((d) => d[j])))
  const batchedF = (ys: Value[]) => vmap((...xs: Value[]) => p.apply(xs, params))(...merge(stacked, ys)) as Value
  if (mode === 'jvp of vmap') {
    const loop = examples.map((e, k) => jvp(f(e.inputs), pick(e.inputs), dirs[k]).tangent as Value)
    same(jvp(batchedF, pick(stacked), stackedDirs).tangent as Value, stackExamples(loop), 1e-9, 1e-9)
    return
  }
  const loop = examples.map((e, k) => vjp(f(e.inputs), pick(e.inputs)).pullback(cts[k] as never) as Value[])
  const batched =
    mode === 'vmap of vjp'
      ? (vmap((ct: Value, ...xs: Value[]) => vjp(f(xs), pick(xs)).pullback(ct as never))(
          stackExamples(cts),
          ...stacked,
        ) as Value[])
      : (vjp(batchedF, pick(stacked)).pullback(stackExamples(cts) as never) as Value[])
  diff.forEach((_, j) => same(batched[j], stackExamples(loop.map((l) => l[j])), 1e-9, 1e-9))
}

/** The shape rule's abstract value against the output of the call. */
function checkShape(p: Primitive, c: PrimitiveCase): void {
  if (!p.shape) throw new Error(`${p.id} has no shape rule`)
  const out = run(p, c.inputs, c.params)
  const aval = p.shape(c.inputs.map(avalOf), c.params)
  const real = avalOf(out)
  expect(aval.shape).toEqual(real.shape)
  expect(aval.dtype).toBe(real.dtype)
  expect(aval.number).toBe(real.number)
}

/** The derivative, batching and shape checks of one case (and of its complex variant when the primitive takes one). */
function caseChecks(
  p: Primitive,
  c: PrimitiveCase,
  examples: readonly PrimitiveCase[],
  draw: Draw,
  tag = '',
): [string, () => void][] {
  const out: [string, () => void][] = []
  const derivatives =
    p.differentiable !== false && !p.zeroDerivative && differentiableOutput(run(p, c.inputs, c.params))
  if (derivatives) {
    out.push([`vjp${tag}`, () => checkVjp(p, c, draw, 1)])
    out.push([`jvp${tag}`, () => checkJvp(p, c, draw)])
    out.push([`dot product${tag}`, () => checkDot(p, c, draw)])
    if (p.test.secondOrder) out.push([`second order${tag}`, () => checkVjp(p, c, draw, 2)])
    if (p.kind === 'general' && p.rules.batch === 'own')
      for (const mode of ['vmap of vjp', 'vjp of vmap', 'jvp of vmap'] as const)
        out.push([`${mode}${tag}`, () => checkBatchedDerivative(p, examples, draw, mode)])
  }
  out.push([`vmap${tag}`, () => checkVmap(p, examples, false)])
  if (!examples[0].inputs.some((x) => typeof x === 'number') || typeof examples[0].inputs[0] !== 'number')
    out.push([`vmap last axis${tag}`, () => checkVmap(p, examples, true)])
  if (p.shape) out.push([`shape rule${tag}`, () => checkShape(p, c)])
  return out
}

/** The elementwise value, broadcasting and dtype checks. */
function elementwiseChecks(p: Primitive, draw: Draw): [string, () => void][] {
  const n = p.arity as number
  const out: [string, () => void][] = []
  const args = Array.from({ length: n }, (_, i) => draw([2, 3], domainOf(p, i)))
  out.push([
    'values',
    () => {
      const ref = run(p, args, undefined) as Tensor
      same(run(p, args.map(transposed), undefined), ref)
      same(run(p, args.map(sliced), undefined), ref)
      const values = args.map(toFlat)
      for (let k = 0; k < 6; k++) {
        const at = elementAt(ref, k)
        // Numbers give a number (a rank-0 tensor for a complex result); rank-0 tensors give a rank-0 tensor.
        const scalarOut = run(
          p,
          values.map((v) => v[k]),
          undefined,
        )
        if (typeof scalarOut === 'number') expect(scalarOut).toBeCloseTo(flat(at)[0], 12)
        else same(scalarOut, at)
        same(
          run(
            p,
            values.map((v) => fromData(Float64Array.of(v[k]), [])),
            undefined,
          ),
          at,
        )
      }
    },
  ])
  if (n > 1)
    out.push([
      'broadcasting',
      () => {
        for (const a of SHAPES)
          for (const b of SHAPES) {
            const ins = args.map((_, i) => draw(i === 0 ? a : i === 1 ? b : [], domainOf(p, i)))
            const target = broadcastShapes(...ins.map((t) => t.shape))
            const expanded = ins.map((t) => copy(broadcastTo(t, target)))
            same(run(p, ins, undefined), run(p, expanded, undefined))
          }
      },
    ])
  out.push([
    'dtype',
    () => {
      const ints = args.map((_, i) => domainOf(p, i)).map((d) => ({ ...d, lo: Math.ceil(d.lo), integer: true }))
      const canInt = ints.every((d) => d.lo <= Math.floor(d.hi))
      const canBool = ints.every((d) => d.lo <= 1 && Math.floor(d.hi) >= 0)
      const variants: [DType, Tensor[]][] = [
        ['float64', args],
        ['float32', args.map((t) => astype(t, 'float32'))],
      ]
      if (canInt) variants.push(['int32', ints.map((d) => astype(draw([2, 3], d), 'int32'))])
      if (canBool)
        variants.push(['bool', args.map(() => astype(draw([2, 3], { lo: 0, hi: 1, integer: true }), 'bool'))])
      for (const [dtype, ins] of variants) {
        const y = run(p, ins, undefined) as Tensor
        if (p.dtype) expect(y.dtype, dtype).toBe(resultType(p.dtype, dtype))
        if (p.shape) expect(p.shape(ins.map(avalOf), undefined).dtype, dtype).toBe(y.dtype)
      }
    },
  ])
  if (p.test.complex) {
    // Complex values of every argument that takes them (integer domains stay real), against the complex scalar loop.
    const cargs = args.map((t, i) => (domainOf(p, i).integer ? t : (complex(t, draw([2, 3], UNIT)) as Tensor)))
    out.push([
      'complex values',
      () => {
        const ref = run(p, cargs, undefined) as Tensor
        const promoted = cargs.map((t) => t.dtype).reduce(promoteTypes)
        if (p.dtype) expect(ref.dtype).toBe(resultType(p.dtype, promoted))
        if (p.shape) expect(p.shape(cargs.map(avalOf), undefined).dtype).toBe(ref.dtype)
        same(run(p, cargs.map(transposed), undefined), ref)
        same(run(p, cargs.map(sliced), undefined), ref)
        for (let k = 0; k < 6; k++) {
          const at = cargs.map((t) => {
            const v = flat(t)
            return t.dtype === 'complex128'
              ? fromData(Float64Array.of(v[2 * k], v[2 * k + 1]), [], 'complex128')
              : fromData(Float64Array.of(v[k]), [], t.dtype)
          })
          same(run(p, at, undefined), elementAt(ref, k))
        }
        if (n > 1) {
          const a = cargs[0]
          const b = cargs.slice(1).map((t) => slice(t, 0, null) as Tensor)
          same(run(p, [a, ...b], undefined), run(p, [a, ...b.map((t) => copy(broadcastTo(t, [2, 3])))], undefined))
        }
      },
    ])
  }
  return out
}

/** Every check of one primitive, each a function that throws on failure. */
function checksOf(p: Primitive): [string, () => void][] {
  const draw = drawFor(p.id)
  const out: [string, () => void][] = []
  // Cases (elementwise: one of shape [2, 3]) and EXAMPLES further draws of each, for vmap.
  const casesOf = (): PrimitiveCase[] =>
    p.kind === 'elementwise'
      ? [{ inputs: Array.from({ length: p.arity as number }, (_, i) => draw([2, 3], domainOf(p, i))) }]
      : [...(p.test.cases?.(draw) ?? [])]
  if (p.kind === 'elementwise') out.push(...elementwiseChecks(p, draw))
  const cases = casesOf()
  const draws = Array.from({ length: EXAMPLES - 1 }, casesOf)
  cases.forEach((c, j) => {
    // Examples for vmap: further draws when they share this case's parameters (parameters computed from the inputs,
    // such as a cached factor, would differ), otherwise this case repeated.
    const others = draws
      .map((d) => d[j])
      .filter((e) => e && e.inputs.length === c.inputs.length && sameParams(e.params, c.params))
    const examples = others.length === EXAMPLES - 1 ? [c, ...others] : Array.from({ length: EXAMPLES }, () => c)
    const tag = cases.length > 1 ? ` #${j}` : ''
    if (p.kind === 'general')
      out.push([
        `values${tag}`,
        () => {
          const ref = run(p, c.inputs, c.params)
          same(run(p, c.inputs.map(transposed), c.params), ref)
          same(run(p, c.inputs.map(sliced), c.params), ref)
        },
      ])
    out.push(...caseChecks(p, c, examples, draw, tag))
    if (p.test.complex && c.inputs.some((x, i) => isDiff(p, i) && typeof x !== 'number' && isFloatInput(x))) {
      const cc = complexCase(p, c, draw)
      out.push(
        ...caseChecks(
          p,
          cc,
          examples.map((e) => complexCase(p, e, draw)),
          draw,
          `${tag} (complex)`,
        ),
      )
    }
  })
  if (p.differentiable === false && !p.zeroDerivative) {
    const c = cases[0]
    if (c)
      out.push([
        'no rule',
        () =>
          expect(() => grad((x: Value) => sum(p.apply([x, ...c.inputs.slice(1)], c.params)))(c.inputs[0])).toThrow(
            NotDifferentiableError,
          ),
      ])
  }
  return out
}

// ── The suite ────────────────────────────────────────────────────────────────────────────────────────────────────────

const fallback = { jvp: 'the transpose trick', batch: 'a loop', shape: 'none' } as const
const gaps: string[] = []
const expected: string[] = []

describe('generated primitive suite', () => {
  it('has registered primitives with module-qualified ids', () => {
    const list = registry.list()
    expect(list.length).toBeGreaterThan(50)
    for (const p of list) expect(p.id).toBe(`${p.module}/${p.name}`)
  })

  for (const p of registry.list()) {
    if (p.kind === 'general' && !p.test.cases) gaps.push(`untested       ${p.id}: no test cases declared`)
    const missing = (['jvp', 'batch', 'shape'] as const).filter((r) => p.rules[r] === 'missing')
    if (missing.length && p.differentiable !== false)
      gaps.push(
        `missing rule   ${p.id}: ${missing.join(', ')} (falls back to ${missing.map((r) => fallback[r]).join(', ')})`,
      )
    else if (missing.includes('shape')) gaps.push(`missing rule   ${p.id}: shape`)
    let checks: [string, () => void][]
    try {
      checks = checksOf(p)
    } catch (e) {
      gaps.push(`setup error    ${p.id}: ${(e as Error).message}`)
      it.fails(`${p.id}: setup`, () => {
        throw e
      })
      continue
    }
    if (checks.length === 0) continue
    describe(p.id, () => {
      const seen = new Map<string, number>()
      for (const [check, body] of checks) {
        const k = (seen.get(check) ?? 0) + 1
        seen.set(check, k)
        const reason = KNOWN[p.id]?.[check.replace(/ #\d+/, '')]
        if (reason) expected.push(`known failure  ${p.id} (${check}): ${reason}`)
        ;(reason ? it.fails : it)(k > 1 ? `${check} [${k}]` : check, body)
      }
    })
  }

  afterAll(() => {
    const lines = [...new Set([...expected, ...gaps])]
    const head = `primitive suite: ${registry.list().length} primitives, ${lines.length} known gaps`
    process.stderr.write(`${[head, ...lines].join('\n  ')}\n`)
  })
})
