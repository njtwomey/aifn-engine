import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { normals } from 'aifn-compute/foundation/random'
import { add, sub, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { TransformerBlock } from 'aifn-compute/nn/attention'
import { activations, recordActivations } from 'aifn-compute/nn/training'

describe('recordActivations', () => {
  it('records the taps of any forward pass, in order, and agrees with activations for a layer', () => {
    const block = TransformerBlock(8, { heads: 2, causal: true })
    const params = block.init(stream('block'))
    const x = normals(stream('x'), [5, 8]) as Tensor
    const rec = recordActivations((ctx) => block.apply(params, x, ctx))
    expect(Object.keys(rec.activations).filter((k) => !k.endsWith('Dropout'))).toEqual([
      'attentionNorm',
      'attention.weights',
      'attention',
      'residual',
      'feedForwardNorm',
      'feedForward.hidden',
      'feedForward',
      'output',
    ])
    expect(toFlat(rec.output as Tensor)).toEqual(toFlat(activations(block, params, x).output as Tensor))
    // Pre-norm residual identities: residual = x + attention, output = residual + feedForward.
    const a = rec.activations as Record<string, Tensor>
    const close = (u: Tensor, v: Tensor) => toFlat(sub(u, v) as Tensor).every((d) => Math.abs(d) < 1e-12)
    expect(close(a.residual, add(x, a.attention) as Tensor)).toBe(true)
    expect(close(a.output, add(a.residual, a.feedForward) as Tensor)).toBe(true)
    expect(a['feedForward.hidden'].shape).toEqual([5, 32])
  })

  it('passes each value through the caller’s own tap', () => {
    const block = TransformerBlock(4, { heads: 1 })
    const params = block.init(stream('b'))
    const seen: string[] = []
    recordActivations((ctx) => block.apply(params, normals(stream('y'), [3, 4]) as Tensor, ctx), {
      tap: (path, v) => (seen.push(path), v),
    })
    expect(seen).toContain('feedForward.hidden')
  })
})
