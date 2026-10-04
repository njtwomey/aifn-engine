/**
 * Laws of the discrete Fourier transforms (values against numpy are in reference.test.ts): inverses under every norm,
 * Parseval, the DFT matrix, the half spectrum of a real signal, n-point padding, N-d transforms axis by axis, shifts,
 * the orthonormal DCT, and derivatives through all of them (ℝ² convention).
 */
import { describe, expect, it } from 'vitest'
import { grad, jvp, vjp, vmap } from 'aifn-compute/foundation/autodiff'
import { stream, normal } from 'aifn-compute/foundation/random'
import {
  dct,
  dctMatrix,
  dft,
  idct,
  dftMatrix,
  fft,
  fft2,
  fftfreq,
  fftn,
  fftshift,
  ifft,
  ifftshift,
  irfft,
  rfft,
  rfftfreq,
  type FftNorm,
} from 'aifn-compute/foundation/fourier'
import {
  abs,
  allclose,
  complex,
  conj,
  eye,
  matmul,
  slice,
  square,
  sum,
  tensor,
  toFlat,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

const s = stream('fourier laws')
const real = (shape: number[]) => normal(s, 0, 1, { shape }) as Tensor
const cplx = (shape: number[]) => complex(real(shape), real(shape)) as Tensor
const same = (a: Value, b: Value, tol = 1e-11) =>
  expect(allclose(a as Tensor, b as Tensor, { rtol: tol, atol: tol })).toBe(true)
const norms: FftNorm[] = ['backward', 'ortho', 'forward']

describe('inverses and Parseval', () => {
  for (const n of [1, 2, 7, 8, 13, 32]) {
    it(`n = ${n}`, () => {
      const z = cplx([n])
      const x = real([n])
      for (const norm of norms) {
        same(ifft(fft(z, { norm }), { norm }), z)
        same(irfft(rfft(x, { norm }), { n, norm }), x)
      }
      // Parseval: Σ|x|² = Σ|X|²/n (backward), = Σ|X|² (ortho).
      const e = sum(square(abs(z))) as number
      expect(sum(square(abs(fft(z)))) as number).toBeCloseTo(n * e, 9)
      expect(sum(square(abs(fft(z, { norm: 'ortho' })))) as number).toBeCloseTo(e, 11)
    })
  }
})

describe('the DFT matrix and the real half spectrum', () => {
  it('fft = F·x with F the DFT matrix; the ortho matrix is unitary', () => {
    const z = cplx([6])
    same(fft(z), matmul(dftMatrix(6), z))
    same(dft(z), fft(z))
    const F = dftMatrix(5, { norm: 'ortho' }) as Tensor
    // F^H F = I.
    same(matmul(transpose(conj(F)), F), complex(eye(5), 0))
  })
  it('rfft is the first ⌊n/2⌋ + 1 bins of fft; n pads with zeros or truncates', () => {
    const x = real([9])
    same(rfft(x), slice(fft(x), [0, 5]))
    same(fft(x, { n: 12 }), fft(tensor([...toFlat(x), 0, 0, 0])))
    same(fft(x, { n: 4 }), fft(slice(x, [0, 4])))
    expect(toFlat(fftfreq(4, 0.5))).toEqual([0, 0.5, -1, -0.5])
    expect(toFlat(rfftfreq(5))).toEqual([0, 0.2, 0.4])
  })
})

describe('N-d transforms and shifts', () => {
  it('fft2 = fft along each axis; fftn over chosen axes; shifts invert', () => {
    const z = cplx([3, 4, 2])
    same(fftn(z, { axes: [0, 1] }), fft(fft(z, { axis: 0 }), { axis: 1 }))
    same(fft2(slice(z, null, null, 0)), fft(fft(slice(z, null, null, 0), { axis: 0 }), { axis: 1 }))
    same(ifftshift(fftshift(z)), z)
    expect(toFlat(fftshift(tensor([0, 1, 2, 3, 4])))).toEqual([3, 4, 0, 1, 2])
  })
})

describe('the orthonormal DCT', () => {
  it('is an orthogonal matrix along the last axis', () => {
    const C = dctMatrix(6) as Tensor
    same(matmul(C, transpose(C)), eye(6))
    const x = real([2, 6])
    same(dct(x), matmul(x, transpose(C)))
    same(idct(dct(x)), x)
  })
})

describe('derivatives', () => {
  it('the ℝ² dot-product test for fft, ifft, rfft and irfft under each norm', () => {
    for (const norm of norms) {
      const cases: [(v: Value) => Value, Tensor, Tensor][] = [
        [(v) => fft(v, { norm }), cplx([8]), cplx([8])],
        [(v) => ifft(v, { norm }), cplx([5]), cplx([5])],
        [(v) => rfft(v, { norm }), real([7]), cplx([4])],
        [(v) => irfft(v, { n: 7, norm }), cplx([4]), real([7])],
      ]
      for (const [f, x, u] of cases) {
        const v = x.dtype === 'complex128' ? cplx(x.shape as number[]) : real(x.shape as number[])
        const Jv = toFlat(jvp(f, x, v).tangent as Tensor)
        const JTu = toFlat(vjp(f, x).pullback(u as never) as Tensor)
        const left = toFlat(u).reduce((a, w, k) => a + w * Jv[k], 0)
        const right = toFlat(v).reduce((a, w, k) => a + w * JTu[k], 0)
        expect(left).toBeCloseTo(right, 10)
      }
    }
  })
  it('the gradient of a power spectrum; vmap over signals', () => {
    // ∇ Σ|rfft(x)|² for the backward norm: ∂/∂x Σₖ cₖ|Xₖ|² with cₖ the half-spectrum weights; checked by differences.
    const f = (x: Value) => sum(square(abs(rfft(x))))
    const x = real([6])
    const g = toFlat(grad(f)(x) as Tensor)
    const h = 1e-6
    toFlat(x).forEach((_, k) => {
      const at = (d: number) => {
        const e = toFlat(x)
        e[k] += d
        return f(tensor(e)) as number
      }
      expect(g[k]).toBeCloseTo((at(h) - at(-h)) / (2 * h), 6)
    })
    const X = real([3, 6])
    same(vmap(grad(f))(X), tensor([0, 1, 2].map((i) => toFlat(grad(f)(slice(X, i)) as Tensor))))
  })
})

describe('one transform path', () => {
  it('the raw in-place kernel is not part of the public API', async () => {
    const mod: Record<string, unknown> = await import('aifn-compute/foundation/fourier')
    expect(mod.transformInPlace).toBeUndefined()
  })
})
