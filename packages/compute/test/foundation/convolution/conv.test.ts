/**
 * The N-d convolution family against a naive reference (values against scipy are in reference.test.ts): every
 * geometry (stride, dilation, groups, padding, flip) and method (direct, fft, overlap-add) agree; convTranspose is
 * the adjoint of conv; numpy.pad's modes; linearFilter's recursion, initial state and continuation.
 */
import { describe, expect, it } from 'vitest'
import { grad, vjp, vmap } from 'aifn-compute/foundation/autodiff'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { normal, stream } from 'aifn-compute/foundation/random'
import {
  conv,
  convOutputSize,
  convTranspose,
  correlate,
  linearFilter,
  pad,
  type ConvOptions,
} from 'aifn-compute/foundation/convolution'
import {
  add,
  allclose,
  complex,
  concat,
  dot,
  flatten,
  imagPart,
  realPart,
  slice,
  sub,
  sum,
  tensor,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

const s = stream('convolution tests')
const draw = (shape: number[]) => normal(s, 0, 1, { shape }) as Tensor
const same = (a: Value, b: Value, tol = 1e-10) =>
  expect(allclose(a as Tensor, b as Tensor, { rtol: tol, atol: tol })).toBe(true)

/** Naive N-d grouped convolution: y[n, o, p] = Σ_{c, k} x[n, g·Cg + c, p·s + k·d − lo] · w[o, c, k] (flipped when asked). */
function naive(
  x: Tensor,
  w: Tensor,
  o: { stride: number[]; dilation: number[]; lo: number[]; hi: number[]; groups: number; flip: boolean },
): Tensor {
  const [N, , ...S] = x.shape
  const [O, Cg, ...K] = w.shape
  const Y = S.map((n, i) => Math.floor((n + o.lo[i] + o.hi[i] - o.dilation[i] * (K[i] - 1) - 1) / o.stride[i]) + 1)
  const xs = toFlat(x)
  const ws = toFlat(w)
  const strides = (shape: number[]) => shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1))
  const sx = strides(x.shape as number[])
  const sw = strides(w.shape as number[])
  const out: number[] = []
  const Og = O / o.groups
  const index = (dims: number[], k: number) =>
    dims.map((_, i) => Math.floor(k / dims.slice(i + 1).reduce((a, b) => a * b, 1)) % dims[i])
  const Ysize = Y.reduce((a, b) => a * b, 1)
  const Ksize = K.reduce((a, b) => a * b, 1)
  for (let n = 0; n < N; n++)
    for (let oc = 0; oc < O; oc++)
      for (let yk = 0; yk < Ysize; yk++) {
        const p = index(Y, yk)
        const g = Math.floor(oc / Og)
        let acc = 0
        for (let c = 0; c < Cg; c++)
          for (let kk = 0; kk < Ksize; kk++) {
            const k = index(K, kk)
            const pos = p.map((pi, i) => pi * o.stride[i] + k[i] * o.dilation[i] - o.lo[i])
            if (pos.some((q, i) => q < 0 || q >= S[i])) continue
            const kIdx = o.flip ? k.map((ki, i) => K[i] - 1 - ki) : k
            let xi = n * sx[0] + (g * Cg + c) * sx[1]
            pos.forEach((q, i) => (xi += q * sx[2 + i]))
            let wi = oc * sw[0] + c * sw[1]
            kIdx.forEach((q, i) => (wi += q * sw[2 + i]))
            acc += xs[xi] * ws[wi]
          }
        out.push(acc)
      }
  return tensor(out, [N, O, ...Y])
}

type Case = {
  x: number[]
  w: number[]
  o: ConvOptions
  stride: number[]
  dilation: number[]
  lo: number[]
  hi: number[]
}
const cases: Case[] = [
  { x: [2, 2, 11], w: [3, 2, 3], o: {}, stride: [1], dilation: [1], lo: [0], hi: [0] },
  {
    x: [1, 4, 13],
    w: [6, 2, 4],
    o: { stride: 2, dilation: 2, groups: 2, padding: [[1, 2]] },
    stride: [2],
    dilation: [2],
    lo: [1],
    hi: [2],
  },
  {
    x: [2, 1, 6, 5],
    w: [2, 1, 3, 2],
    o: { padding: [1, 0], flip: false },
    stride: [1, 1],
    dilation: [1, 1],
    lo: [1, 0],
    hi: [1, 0],
  },
  {
    x: [1, 2, 7, 6],
    w: [4, 1, 2, 3],
    o: { stride: [2, 1], dilation: [1, 2], groups: 2, padding: 1 },
    stride: [2, 1],
    dilation: [1, 2],
    lo: [1, 1],
    hi: [1, 1],
  },
  {
    x: [1, 1, 4, 5, 3],
    w: [2, 1, 2, 2, 2],
    o: { padding: 'valid' },
    stride: [1, 1, 1],
    dilation: [1, 1, 1],
    lo: [0, 0, 0],
    hi: [0, 0, 0],
  },
]

describe('conv against a naive reference', () => {
  cases.forEach((c, j) => {
    it(`case ${j}: every method`, () => {
      const x = draw(c.x)
      const w = draw(c.w)
      const ref = naive(x, w, { ...c, groups: c.o.groups ?? 1, flip: c.o.flip ?? true })
      for (const method of ['direct', 'fft', 'overlapAdd', 'auto'] as const) same(conv(x, w, { ...c.o, method }), ref)
      const out = c.x
        .slice(2)
        .map((n, i) => Math.floor((n + c.lo[i] + c.hi[i] - c.dilation[i] * (c.w[2 + i] - 1) - 1) / c.stride[i]) + 1)
      expect((conv(x, w, c.o) as Tensor).shape.slice(2)).toEqual(out)
      c.lo.forEach((lo, i) => {
        if (lo === c.hi[i]) expect(convOutputSize(c.x[2 + i], c.w[2 + i], c.stride[i], lo, c.dilation[i])).toBe(out[i])
      })
    })
  })
  it('same and full padding follow scipy for odd and even kernels', () => {
    const x = draw([1, 1, 9])
    for (const k of [3, 4]) {
      const w = draw([1, 1, k])
      expect((conv(x, w, { padding: 'same' }) as Tensor).shape).toEqual([1, 1, 9])
      const full = conv(x, w, { padding: 'full' }) as Tensor
      expect(full.shape).toEqual([1, 1, 9 + k - 1])
      // 'same' pads ⌈(K − 1)/2⌉ before, 'full' K − 1: the same output starts ⌊(K − 1)/2⌋ into the full one.
      const start = Math.floor((k - 1) / 2)
      same(conv(x, w, { padding: 'same' }), slice(full, null, null, [start, start + 9]))
    }
  })
  it('refuses kernels that do not fit and mismatched channels', () => {
    expect(() => conv(draw([1, 1, 2]), draw([1, 1, 3]))).toThrow(ShapeError)
    expect(() => conv(draw([1, 3, 5]), draw([1, 2, 3]))).toThrow(ShapeError)
  })
})

describe('convTranspose is the adjoint of conv', () => {
  cases.slice(0, 4).forEach((c, j) => {
    it(`case ${j}: ⟨conv(x, w), y⟩ = ⟨x, convTranspose(y, w)⟩, and it is conv's vjp in x`, () => {
      const x = draw(c.x)
      const w = draw(c.w)
      const y = draw((conv(x, w, c.o) as Tensor).shape as number[])
      const xt = convTranspose(y, w, { ...c.o, size: c.x.slice(2) })
      expect(dot(flatten(conv(x, w, c.o)), flatten(y)) as number).toBeCloseTo(
        dot(flatten(x), flatten(xt)) as number,
        10,
      )
      same(vjp((v: Value) => conv(v, w, c.o), x).pullback(y as never) as Value, xt)
    })
  })
})

describe('pad (numpy.pad semantics)', () => {
  const x = tensor([1, 2, 3])
  it('every mode', () => {
    expect(toFlat(pad(x, 2) as Tensor)).toEqual([0, 0, 1, 2, 3, 0, 0])
    expect(toFlat(pad(x, [1, 2], 'constant', { value: 9 }) as Tensor)).toEqual([9, 1, 2, 3, 9, 9])
    expect(toFlat(pad(x, 2, 'reflect') as Tensor)).toEqual([3, 2, 1, 2, 3, 2, 1])
    expect(toFlat(pad(x, 2, 'symmetric') as Tensor)).toEqual([2, 1, 1, 2, 3, 3, 2])
    expect(toFlat(pad(x, 2, 'edge') as Tensor)).toEqual([1, 1, 1, 2, 3, 3, 3])
    expect(toFlat(pad(x, 2, 'wrap') as Tensor)).toEqual([2, 3, 1, 2, 3, 1, 2])
  })
  it('per-axis widths on a matrix; the gradient folds the border back', () => {
    const m = tensor([
      [1, 2],
      [3, 4],
    ])
    expect(
      (
        pad(m, [
          [1, 0],
          [0, 2],
        ]) as Tensor
      ).shape,
    ).toEqual([3, 4])
    // Σ reflect-pad(x): x₁ and x₂ appear twice more in [3 2 1 2 3 2 1], so the gradient is [2, 3, 2].
    expect(toFlat(grad((v: Value) => sum(pad(v, 2, 'reflect')))(x) as Tensor)).toEqual([2, 3, 2])
  })
})

describe('linearFilter', () => {
  const b = [0.2, 0.3]
  const a = [1, -0.5, 0.06]
  it('solves the difference equation, with a₀ normalised', () => {
    const x = toFlat(draw([20]))
    const y: number[] = []
    for (let t = 0; t < x.length; t++)
      y.push(
        b[0] * x[t] + (t > 0 ? b[1] * x[t - 1] : 0) - (t > 0 ? a[1] * y[t - 1] : 0) - (t > 1 ? a[2] * y[t - 2] : 0),
      )
    toFlat(linearFilter(b, a, tensor(x)) as Tensor).forEach((v, t) => expect(v).toBeCloseTo(y[t], 12))
    same(
      linearFilter(
        b.map((v) => 2 * v),
        a.map((v) => 2 * v),
        tensor(x),
      ),
      linearFilter(b, a, tensor(x)),
    )
  })
  it('a = [1] is an FIR convolution; a batch filters along the chosen axis', () => {
    const x = draw([3, 12])
    const fir = [0.5, -1, 0.25]
    const rows = [0, 1, 2].map((i) => linearFilter(fir, [1], slice(x, i)) as Tensor)
    same(
      linearFilter(fir, [1], x),
      tensor(
        rows.map((r) => toFlat(r)),
        [3, 12],
      ),
    )
    same(linearFilter(fir, [1], transpose2(x), { axis: 0 }), transpose2(linearFilter(fir, [1], x) as Tensor))
  })
  it('the initial state continues a split signal exactly', () => {
    // The state after the first half is the zi of the second: filtering [x₁, x₂] = filtering x₁, then x₂ from there.
    const x = draw([16])
    const whole = linearFilter(b, a, x) as Tensor
    const first = slice(x, [0, 8])
    const second = slice(x, [8, 16])
    const y1 = toFlat(linearFilter(b, a, first) as Tensor)
    const x1 = toFlat(first)
    // Transposed direct form II state after step 7: d₁ = b₁x₇ − a₁y₇ + d₂, d₂ = −a₂y₇ (b₂ = 0).
    const d2 = -a[2] * y1[7]
    const d1 = b[1] * x1[7] - a[1] * y1[7] - a[2] * y1[6]
    same(linearFilter(b, a, second, { zi: tensor([d1, d2]) }), slice(whole, [8, 16]))
  })
  it('vmap over signals and over coefficients equals a loop', () => {
    const X = draw([4, 10])
    same(vmap((x: Value) => linearFilter(b, a, x))(X), linearFilter(b, a, X))
    const B = draw([3, 2])
    const x = draw([10])
    same(
      vmap((bb: Value) => linearFilter(bb, a, x))(B),
      concat(
        [0, 1, 2].map((i) => linearFilter(slice(B, i), a, x) as Tensor).map((r) => tensor(toFlat(r), [1, 10])),
        0,
      ),
    )
  })
})

function transpose2(t: Tensor): Tensor {
  const [m, n] = t.shape
  const v = toFlat(t)
  return tensor(
    Array.from({ length: m * n }, (_, k) => v[(k % m) * n + Math.floor(k / m)]),
    [n, m],
  )
}

describe('the FFT method runs on the fourier primitives', () => {
  it('gradients through method fft equal those through direct, for the input and the kernel', () => {
    const x = draw([1, 2, 6, 5])
    const w = draw([3, 2, 3, 2])
    const byX = (method: 'direct' | 'fft') => (v: Value) => sum(conv(v, w, { method, padding: 'same' }) as Value)
    const byW = (method: 'direct' | 'fft') => (k: Value) => sum(conv(x, k, { method, padding: 'same' }) as Value)
    same(grad(byX('fft'))(x) as Tensor, grad(byX('direct'))(x) as Tensor)
    same(grad(byW('fft'))(w) as Tensor, grad(byW('direct'))(w) as Tensor)
  })
})

describe('complex inputs', () => {
  const cplx = (re: number[], im: number[]) => complex(tensor(re), tensor(im)) as Tensor
  /** numpy.convolve on complex lists, written out. */
  const npConvolve = (a: [number, number][], b: [number, number][]) => {
    const out = Array.from({ length: a.length + b.length - 1 }, () => [0, 0])
    a.forEach(([ar, ai], i) =>
      b.forEach(([br, bi], j) => {
        out[i + j][0] += ar * br - ai * bi
        out[i + j][1] += ar * bi + ai * br
      }),
    )
    return out
  }
  const parts = (z: Value) => [toFlat(realPart(z) as Tensor), toFlat(imagPart(z) as Tensor)]

  it('complex x and w give numpy.convolve, with every method', () => {
    const a: [number, number][] = [
      [1, 2],
      [-0.5, 0.25],
      [3, -1],
      [0, 1],
    ]
    const b: [number, number][] = [
      [0.5, -1],
      [2, 0],
    ]
    const ref = npConvolve(a, b)
    const x = cplx(
      a.map((p) => p[0]),
      a.map((p) => p[1]),
    )
    const w = cplx(
      b.map((p) => p[0]),
      b.map((p) => p[1]),
    )
    for (const method of ['direct', 'fft', 'overlapAdd'] as const) {
      const [re, im] = parts(conv(x, w, { padding: 'full', method }))
      re.forEach((v, k) => expect(v).toBeCloseTo(ref[k][0], 12))
      im.forEach((v, k) => expect(v).toBeCloseTo(ref[k][1], 12))
    }
  })
  it('one complex factor: the real kernel filters both parts', () => {
    const x = cplx([1, 2, 3], [0, -1, 1])
    const w = tensor([1, -1])
    const [re, im] = parts(conv(x, w, { padding: 'full' }))
    expect(re).toEqual([1, 1, 1, -3])
    expect(im).toEqual([0, -1, 2, -1])
    expect((conv(w, x, { padding: 'full' }) as Tensor).dtype).toBe('complex128')
  })
  it('correlate conjugates the second signal, as scipy.signal.correlate', () => {
    const [re, im] = parts(correlate(cplx([1, 0], [0, 1]), cplx([0], [1])))
    expect(re).toEqual([0, 1])
    expect(im).toEqual([-1, 0])
  })
  it('is differentiable in the real and imaginary parts (ℝ² convention)', () => {
    const w = cplx([0.5, -1], [1, 0.25])
    // f(x) = Σ Re(conv(x, w)) for real x: ∂f/∂x_i = Σ_k Re(w_k).
    const g = grad((v: Value) => sum(realPart(conv(v, w, { padding: 'full' }))))(tensor([1, 2, 3])) as Tensor
    toFlat(g).forEach((v) => expect(v).toBeCloseTo(-0.5, 12))
    const h = grad((v: Value) => sum(imagPart(conv(v, w, { padding: 'full' }))))(tensor([1, 2, 3])) as Tensor
    toFlat(h).forEach((v) => expect(v).toBeCloseTo(1.25, 12))
  })
  it('convTranspose extends bilinearly', () => {
    const y = cplx([1, -1], [2, 0.5])
    const w = cplx([0.5, 1, -2], [0, 1, 0])
    const z = convTranspose(y, w) as Tensor
    const byParts = (f: (a: Value, b: Value) => Value) =>
      complex(
        sub(f(realPart(y), realPart(w)), f(imagPart(y), imagPart(w))),
        add(f(realPart(y), imagPart(w)), f(imagPart(y), realPart(w))),
      )
    same(z, byParts((a, b) => convTranspose(a, b)) as Tensor)
  })
})
