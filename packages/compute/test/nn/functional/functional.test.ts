/**
 * aifn-compute/nn/functional against torch (float64 goldens, `fixtures/nn.json`): convolutions through the
 * foundation/convolution family (values unchanged from the old nn primitives), pooling and activations, with
 * gradients; second derivatives through conv2d by finite differences.
 */
import { describe, expect, it } from 'vitest'
import { grad, gradCheck } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { mul, sum, tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import {
  activationFn,
  avgPool1d,
  avgPool2d,
  conv1d,
  conv2d,
  convOutputSize,
  elu,
  gelu,
  leakyRelu,
  maxPool1d,
  maxPool2d,
  relu,
  silu,
} from 'aifn-compute/nn/functional'
import { Conv2d } from 'aifn-compute/nn/layers'
import { fixture } from '../../fixtures'

type N = number[] | number[][] | number[][][] | number[][][][]
type Case = Record<string, N | number | number[]>
const F = fixture<Record<string, Case>>('nn')
const T = (x: unknown) => tensor(x as N)
const flat = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}
const flatN = (x: unknown): number[] => (Array.isArray(x) ? (x as unknown[]).flatMap(flatN) : [x as number])

function close(actual: Value | number[], expected: unknown, tol = 1e-9) {
  const a = Array.isArray(actual) ? actual : flat(actual)
  const e = flatN(expected)
  expect(a.length).toBe(e.length)
  let worst = 0
  a.forEach((v, i) => (worst = Math.max(worst, Math.abs(v - e[i]) / Math.max(1, Math.abs(e[i])))))
  expect(worst).toBeLessThan(tol)
}

/** Σ w·y as a function to differentiate. */
const weighted = (w: unknown) => (y: Value) => sum(mul(y, T(w)))

describe('convolution', () => {
  for (const name of ['conv2d_plain', 'conv2d_strided', 'conv2d_dilated']) {
    it(`${name} matches torch, value and gradients`, () => {
      const c = F[name]
      const opts = {
        stride: c.stride as [number, number],
        padding: c.padding as [number, number],
        dilation: c.dilation as [number, number],
      }
      close(conv2d(T(c.x), T(c.k), opts), c.y)
      const f = (x: Value, k: Value) => weighted(c.w)(conv2d(x, k, opts))
      const [gx, gk] = grad(f, { argnums: [0, 1] })(T(c.x), T(c.k))
      close(gx, c.gx)
      close(gk, c.gk)
    })
  }

  it('conv1d matches torch', () => {
    const c = F.conv1d
    const opts = { stride: 2, padding: 1, dilation: 2 }
    close(conv1d(T(c.x), T(c.k), opts), c.y)
    const [gx, gk] = grad((x: Value, k: Value) => weighted(c.w)(conv1d(x, k, opts)), { argnums: [0, 1] })(
      T(c.x),
      T(c.k),
    )
    close(gx, c.gx)
    close(gk, c.gk)
  })

  it('second derivatives through conv2d agree with finite differences', () => {
    const s = stream('conv-second')
    const x = Conv2d(1, 1, 2).init(s).weight
    const img = tensor([
      [
        [0.3, -0.2, 0.5],
        [0.1, 0.7, -0.4],
        [0.2, 0.0, 0.6],
      ],
    ])
    // g(k) = ‖∂/∂x Σ conv(x, k)²‖², a function of the kernel through the input gradient.
    const g = (k: Value) => {
      const gx = grad((u: Value) => sum(mul(conv2d(u, k), conv2d(u, k))))(img as Value)
      return sum(mul(gx, gx))
    }
    expect(gradCheck(g, x as Value, { rtol: 1e-4 }).ok).toBe(true)
  })

  it('pooling matches torch', () => {
    const m = F.maxpool2d
    close(maxPool2d(T(m.x), 2, { stride: 2, padding: 1 }), m.y)
    close(grad((x: Value) => weighted(m.w)(maxPool2d(x, 2, { stride: 2, padding: 1 })))(T(m.x)), m.gx)
    const a = F.avgpool2d
    close(avgPool2d(T(a.x), 3, { stride: 2, padding: 1 }), a.y)
    close(grad((x: Value) => weighted(a.w)(avgPool2d(x, 3, { stride: 2, padding: 1 })))(T(a.x)), a.gx)
  })

  it('conv2d supports groups: each group sees only its channels', () => {
    const x = tensor([
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
    ])
    // Two groups, one output channel each, 1×1 kernels 2 and 10.
    const w = tensor([[[[2]]], [[[10]]]])
    expect(toFlat(conv2d(x, w, { groups: 2 }) as Tensor)).toEqual([2, 4, 6, 8, 50, 60, 70, 80])
  })

  it('output sizes follow ⌊(n + 2p − d(k − 1) − 1)/s⌋ + 1', () => {
    expect(convOutputSize(7, 3, 2, 1, 1)).toBe(4)
    expect(convOutputSize(6, 2, 1, 1, 2)).toBe(6)
    const c = F.conv2d_strided
    const y = unwrap(conv2d(T(c.x), T(c.k), { stride: [2, 1], padding: [1, 2] })) as Tensor
    expect(y.shape.slice(2)).toEqual([convOutputSize(7, 3, 2, 1), convOutputSize(6, 2, 1, 2)])
  })

  it('1-D pooling is 2-D pooling of a height-1 image', () => {
    const x = tensor([[[1, 5, 2, 8, 3, 3]]])
    expect(toFlat(maxPool1d(x, 2) as Tensor)).toEqual([5, 8, 3])
    close(avgPool1d(x, 3), [8 / 3, 14 / 3], 1e-14)
  })
})

describe('activations', () => {
  it('activations match torch', () => {
    const a = F.activations
    const x = T(a.x)
    close(gelu(x), a.gelu, 1e-12)
    close(gelu(x, { approximate: 'tanh' }), a.geluTanh, 1e-12)
    close(silu(x), a.silu, 1e-12)
    close(elu(x), a.elu, 1e-12)
    close(leakyRelu(x), a.leakyRelu, 1e-12)
    expect(grad((v: Value) => relu(v))(-1)).toBe(0)
    expect(grad((v: Value) => relu(v))(2)).toBe(1)
  })

  it('elu has a finite gradient for large inputs (review: e^x overflowed in the unused branch, giving NaN)', () => {
    expect(flat(grad((v: Value) => sum(elu(v)))(tensor([1000, -1000, 0.5])) as Value)).toEqual([1, 0, 1])
    expect(flat(grad((v: Value) => sum(elu(v)))(tensor([-1])) as Value)[0]).toBeCloseTo(Math.exp(-1), 15)
  })

  it('activationFn looks activations up by name', () => {
    const x = tensor([-1, 0.5])
    expect(flat(activationFn('relu')(x))).toEqual([0, 0.5])
    expect(flat(activationFn('tanh')(x))).toEqual([Math.tanh(-1), Math.tanh(0.5)])
  })
})
