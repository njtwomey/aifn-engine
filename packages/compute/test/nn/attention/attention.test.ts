/**
 * aifn-compute/nn/attention against torch (float64 goldens, `fixtures/nn/attention.json` and the multi-head case of
 * `fixtures/nn.json`): scaled dot-product attention with every mask, ALiBi bias and soft-capping; multi-head attention
 * with grouped-query and multi-query heads, QK-norm and both rotary layouts; latent attention; feed-forward layers;
 * transformer blocks (pre, post, parallel, LLaMA-like, ALiBi); the positional schemes (RoPE, its rescalings, sinusoidal,
 * ALiBi slopes, T5 buckets). Laws: the key–value cache reproduces a full causal pass token by token, the rolling
 * window cache too, RoPE scores depend only on the offset, and the tiled online softmax equals full attention.
 */
import { describe, expect, it } from 'vitest'
import { grad, gradCheck } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import {
  concat,
  dot,
  mul,
  slice,
  sum,
  tensor,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import {
  alibiSlopes,
  applyRope,
  causalMask,
  feedForward,
  flashAttention,
  flashAttentionSteps,
  kvCacheMemory,
  multiHeadAttention,
  MultiHeadAttention,
  multiHeadLatentAttention,
  MultiHeadLatentAttention,
  paddingMask,
  positionMask,
  ropeFrequencies,
  scaledDotProductAttention,
  sinusoidalPositions,
  slidingWindowMask,
  t5RelativeBias,
  t5RelativeBucket,
  TransformerBlock,
  transformerBlock,
  type FeedForwardKind,
  type KvCache,
  type LatentCache,
  type TransformerBlockOptions,
} from 'aifn-compute/nn/attention'
import { fixture } from '../../fixtures'

type Nested = number | Nested[]
type Tree = { [k: string]: Tree | Nested }
type Sdpa = Record<'q' | 'k' | 'v' | 'y' | 'weights' | 'r' | 'gq' | 'gk' | 'gv', Nested> & {
  mask?: Nested
  bias?: Nested
  softCap?: number
}
type Fx = {
  sdpa: Record<string, Sdpa>
  mha: ({ heads: number; kvHeads: number; rope: 'half' | 'interleaved' | null; qkNorm: boolean; params: Tree } & Record<
    'x' | 'y' | 'r' | 'gx',
    Nested
  >)[]
  mla: { heads: number; params: Tree } & Record<'x' | 'y' | 'r' | 'gx', Nested>
  ffn: ({ kind: FeedForwardKind; params: Tree } & Record<'x' | 'y' | 'r' | 'gx', Nested>)[]
  block: {
    x: Nested
    r: Nested
    dModel: number
    heads: number
    hidden: number
    blocks: {
      placement: 'pre' | 'post'
      norm: 'layer' | 'rms'
      causal: boolean
      feedForward: FeedForwardKind
      parallel: boolean
      kvHeads: number
      position: 'none' | 'rope' | 'alibi'
      bias: boolean
      params: Tree
      y: Nested
      gradX: Nested
    }[]
  }
  rope: Record<'x' | 'half' | 'interleaved' | 'base500', Nested>
  ropeFrequencies: Record<'default' | 'linear' | 'ntk' | 'yarn', number[]> & { yarnMagnitude: number }
  sinusoidal: Nested
  alibiSlopes: Record<string, number[]>
  t5: Record<'relative' | 'bidirectional' | 'causal' | 'small', number[]>
}
const F = fixture<Fx>('nn/attention')
const T = (x: unknown) => tensor(x as never)
const flat = (v: Value | Nested): number[] => {
  if (Array.isArray(v)) return (v as Nested[]).flatMap((e) => flat(e))
  if (typeof v === 'number') return [v]
  const r = unwrap(v as Value)
  return typeof r === 'number' ? [r] : toFlat(r)
}
function close(actual: Value | number[], expected: Nested | number[], tol = 1e-10) {
  const a = Array.isArray(actual) ? (actual as number[]) : flat(actual)
  const e = flat(expected as Nested)
  expect(a.length).toBe(e.length)
  let worst = 0
  a.forEach((v, i) => {
    if (!Number.isFinite(e[i])) expect(v).toBe(e[i])
    else worst = Math.max(worst, Math.abs(v - e[i]) / Math.max(1, Math.abs(e[i])))
  })
  expect(worst).toBeLessThan(tol)
}
const toTensors = (t: Tree): unknown =>
  Object.fromEntries(Object.entries(t).map(([k, v]) => [k, Array.isArray(v) ? T(v) : toTensors(v as Tree)]))
const weighted = (r: Nested) => (y: Value) => sum(mul(y, T(r)))

describe('scaled dot-product attention', () => {
  for (const [name, c] of Object.entries(F.sdpa))
    it(`matches torch: ${name}`, () => {
      const opts = {
        mask: c.mask === undefined ? undefined : T(c.mask),
        bias: c.bias === undefined ? undefined : T(c.bias),
        softCap: c.softCap,
      }
      const { output, weights } = scaledDotProductAttention(T(c.q), T(c.k), T(c.v), opts)
      close(output, c.y)
      close(weights, c.weights)
      const f = (q: Value, k: Value, v: Value) => weighted(c.r)(scaledDotProductAttention(q, k, v, opts).output)
      for (const [i, g] of [c.gq, c.gk, c.gv].entries())
        close(grad(f, { argnums: i as 0 | 1 | 2 })(T(c.q), T(c.k), T(c.v)) as Value, g, 1e-9)
    })

  it('the causal and window options build the same masks as the mask constructors', () => {
    const c = F.sdpa.window
    const viaOption = scaledDotProductAttention(T(c.q), T(c.k), T(c.v), { causal: true, window: 3 })
    close(viaOption.output, c.y)
    expect(toFlat(slidingWindowMask(3, 3, 7))).toEqual(flat(c.mask!))
    expect(toFlat(causalMask(3))).toEqual([1, 0, 0, 1, 1, 0, 1, 1, 1])
    expect(toFlat(paddingMask([2, 3], 3))).toEqual([1, 1, 0, 1, 1, 1])
  })

  it('the nn.json multi-head case still matches torch.nn.MultiheadAttention', () => {
    const N = fixture<Record<string, Record<string, Nested>>>('nn').mha
    const p = {
      query: { weight: T(N.wq), bias: T(N.bq) },
      key: { weight: T(N.wk), bias: T(N.bk) },
      value: { weight: T(N.wv), bias: T(N.bv) },
      output: { weight: T(N.wo), bias: T(N.bo) },
    }
    const heads = N.heads as number
    const r = multiHeadAttention(p, T(N.x), T(N.x), { heads, causal: true })
    close(r.output, N.y, 1e-9)
    close(r.weights, N.weights, 1e-9)
  })
})

describe('multi-head attention', () => {
  for (const c of F.mha)
    it(`matches torch: ${c.heads} heads, ${c.kvHeads} kv heads, rope ${c.rope}, qk-norm ${c.qkNorm}`, () => {
      const p = toTensors(c.params) as Parameters<typeof multiHeadAttention>[0]
      const opts = { heads: c.heads, kvHeads: c.kvHeads, causal: true, rope: c.rope ? { layout: c.rope } : undefined }
      close(multiHeadAttention(p, T(c.x), T(c.x), opts).output, c.y)
      const gx = grad((x: Value) => weighted(c.r)(multiHeadAttention(p, x, x, opts).output))(T(c.x))
      close(gx, c.gx, 1e-9)
    })

  it('decoding token by token through the cache equals one causal pass (and a rolling window too)', () => {
    for (const window of [undefined, 3]) {
      const layer = MultiHeadAttention(8, { heads: 4, kvHeads: 2, qkNorm: true })
      const p = layer.init(stream('cache'))
      const x = tensor(Array.from({ length: 7 }, (_, t) => Array.from({ length: 8 }, (_, j) => Math.sin(t * 1.3 + j))))
      const opts = { heads: 4, kvHeads: 2, causal: true, window, rope: true }
      const full = multiHeadAttention(p, x, x, opts).output
      let cache: KvCache | null = null
      const steps: Value[] = []
      // Prefill three tokens, then one at a time.
      const first = multiHeadAttention(p, slice(x, [0, 3]), slice(x, [0, 3]), opts, { cache })
      steps.push(first.output)
      cache = first.cache
      for (let t = 3; t < 7; t++) {
        const r = multiHeadAttention(p, slice(x, [t, t + 1]), slice(x, [t, t + 1]), opts, { cache })
        steps.push(r.output)
        cache = r.cache
      }
      close(concat(steps, 0), flat(full), 1e-12)
      expect(cache!.positions.length).toBe(window ?? 7)
    }
  })

  it('a layer has correct gradients with grouped-query heads and QK-norm', () => {
    const layer = MultiHeadAttention(8, { heads: 4, kvHeads: 2, qkNorm: true, causal: true, rope: true })
    const p = layer.init(stream('mha'))
    const x = tensor(Array.from({ length: 4 }, (_, t) => Array.from({ length: 8 }, (_, j) => Math.cos(t + 0.5 * j))))
    expect(gradCheck((q: typeof p) => sum(layer.apply(q, x)), p, { rtol: 1e-4, atol: 1e-6 }).ok).toBe(true)
  })
})

describe('multi-head latent attention', () => {
  it('matches a torch transcription of DeepSeek-V2', () => {
    const c = F.mla
    const p = toTensors(c.params) as Parameters<typeof multiHeadLatentAttention>[0]
    const opts = { heads: c.heads, causal: true }
    close(multiHeadLatentAttention(p, T(c.x), opts).output, c.y)
    close(grad((x: Value) => weighted(c.r)(multiHeadLatentAttention(p, x, opts).output))(T(c.x)), c.gx, 1e-9)
  })

  it('caches only the latent and rotary key, and decodes like a full pass', () => {
    const layer = MultiHeadLatentAttention(12, { heads: 3, latentDim: 4, ropeDim: 2, queryLatentDim: 6 })
    const p = layer.init(stream('mla'))
    const x = tensor(Array.from({ length: 5 }, (_, t) => Array.from({ length: 12 }, (_, j) => Math.sin(t + j / 3))))
    const full = multiHeadLatentAttention(p, x, { heads: 3, causal: true }).output
    let cache: LatentCache | null = null
    const outs: Value[] = []
    for (let t = 0; t < 5; t++) {
      const r = multiHeadLatentAttention(p, slice(x, [t, t + 1]), { heads: 3, causal: true }, { cache })
      outs.push(r.output)
      cache = r.cache
    }
    close(concat(outs, 0), flat(full), 1e-12)
    expect((unwrap(cache!.latent) as Tensor).shape).toEqual([5, 4])
  })
})

describe('feed-forward layers and blocks', () => {
  for (const c of F.ffn)
    it(`${c.kind} matches torch`, () => {
      const p = toTensors(c.params) as Parameters<typeof feedForward>[0]
      close(feedForward(p, T(c.x), { kind: c.kind }), c.y)
      close(grad((x: Value) => weighted(c.r)(feedForward(p, x, { kind: c.kind })))(T(c.x)), c.gx, 1e-9)
    })

  const B = F.block
  for (const c of B.blocks) {
    const label = `${c.parallel ? 'parallel' : c.placement}-norm, ${c.norm}, ${c.feedForward}, kv ${c.kvHeads}, ${c.position}`
    it(`a block matches torch (${label})`, () => {
      const opts: TransformerBlockOptions = {
        heads: B.heads,
        kvHeads: c.kvHeads,
        hidden: B.hidden,
        causal: c.causal,
        placement: c.placement,
        norm: c.norm,
        feedForward: c.feedForward,
        parallel: c.parallel,
        position: c.position,
        bias: c.bias,
      }
      const block = TransformerBlock(B.dModel, opts)
      const params = toTensors(c.params) as ReturnType<typeof block.init>
      close(block.apply(params, T(B.x)), c.y, 1e-11)
      close(grad((x: Value) => weighted(B.r)(block.apply(params, x)))(T(B.x)), c.gradX, 1e-10)
    })
  }

  for (const placement of ['pre', 'post'] as const)
    it(`a ${placement}-norm block with T5 biases keeps its shape and has correct gradients`, () => {
      const block = TransformerBlock(8, { heads: 2, causal: true, hidden: 16, placement, position: 't5' })
      const params = block.init(stream('block'))
      const x = tensor(Array.from({ length: 4 }, (_, t) => Array.from({ length: 8 }, (_, j) => Math.sin(t + j))))
      expect((unwrap(block.apply(params, x)) as Tensor).shape).toEqual([4, 8])
      expect(gradCheck((p: typeof params) => sum(block.apply(p, x)), params, { rtol: 1e-4, atol: 1e-6 }).ok).toBe(true)
    })

  it('a block decodes through its cache like a full pass', () => {
    const opts: TransformerBlockOptions = { heads: 2, kvHeads: 1, causal: true, position: 'alibi', norm: 'rms' }
    const block = TransformerBlock(8, opts)
    const p = block.init(stream('dec'))
    const x = tensor(Array.from({ length: 5 }, (_, t) => Array.from({ length: 8 }, (_, j) => Math.cos(t * j + 1))))
    const full = transformerBlock(p, x, opts).output
    let cache: KvCache | null = null
    const outs: Value[] = []
    for (let t = 0; t < 5; t++) {
      const r = transformerBlock(p, slice(x, [t, t + 1]), opts, { cache })
      outs.push(r.output)
      cache = r.cache
    }
    close(concat(outs, 0), flat(full), 1e-12)
  })
})

describe('positions', () => {
  it('RoPE matches the rotate-half and complex-number reference implementations', () => {
    const c = F.rope
    const positions = Array.from({ length: 9 }, (_, i) => i)
    close(applyRope(T(c.x), positions), c.half)
    close(applyRope(T(c.x), positions, { layout: 'interleaved' }), c.interleaved)
    close(applyRope(T(c.x), positions, { base: 500 }), c.base500)
  })

  it('rotary scores depend only on the offset', () => {
    const q = tensor([[0.3, -1.2, 0.8, 0.5]])
    const k = tensor([[1.1, 0.4, -0.7, 0.2]])
    const score = (p: number, s: number) => flat(dot(slice(applyRope(q, [p]), 0), slice(applyRope(k, [s]), 0)))[0]
    expect(score(7, 4)).toBeCloseTo(score(3, 0), 12)
    expect(score(20, 13)).toBeCloseTo(score(7, 0), 12)
  })

  it('rescaled frequencies match the references (linear, NTK-aware, YaRN)', () => {
    const c = F.ropeFrequencies
    close(ropeFrequencies(64).frequencies, c.default, 1e-13)
    close(ropeFrequencies(64, { scaling: { kind: 'linear', factor: 4 } }).frequencies, c.linear, 1e-13)
    close(ropeFrequencies(64, { scaling: { kind: 'ntk', factor: 4 } }).frequencies, c.ntk, 1e-12)
    const yarn = ropeFrequencies(64, { scaling: { kind: 'yarn', factor: 4, originalLength: 2048 } })
    close(yarn.frequencies, c.yarn, 1e-13)
    expect(yarn.magnitude).toBeCloseTo(c.yarnMagnitude, 14)
  })

  it('sinusoidal encodings, ALiBi slopes and T5 buckets match the references', () => {
    close(sinusoidalPositions(10, 6), F.sinusoidal, 1e-14)
    for (const [n, s] of Object.entries(F.alibiSlopes)) close(alibiSlopes(Number(n)), s, 1e-15)
    const r = F.t5.relative
    expect(r.map((x) => t5RelativeBucket(x))).toEqual(F.t5.bidirectional)
    expect(r.map((x) => t5RelativeBucket(x, { bidirectional: false }))).toEqual(F.t5.causal)
    expect(r.map((x) => t5RelativeBucket(x, { buckets: 8, maxDistance: 20 }))).toEqual(F.t5.small)
  })

  it('the T5 bias gathers its table by bucket and differentiates into it', () => {
    const table = tensor(Array.from({ length: 32 }, (_, b) => [b, -b]))
    const bias = unwrap(t5RelativeBias(table, [0, 1, 2], [0, 1, 2])) as Tensor
    expect(bias.shape).toEqual([2, 3, 3])
    // Query 2, key 0: relative −2, bucket 2; query 0, key 2: relative +2, bucket 16 + 2.
    expect(flat(bias)[2 * 3 + 0]).toBe(2)
    expect(flat(bias)[2]).toBe(18)
    const g = unwrap(grad((t: Value) => sum(t5RelativeBias(t, [0, 1, 2], [0, 1, 2])))(table) as Value) as Tensor
    expect(flat(g).reduce((a, b) => a + b, 0)).toBe(18)
  })
})

describe('cache accounting and tiled attention', () => {
  it('the cache memory follows 2·layers·g·d_h per token, and h/g and latent savings', () => {
    const llama = { layers: 32, heads: 32, headDim: 128 }
    const mha = kvCacheMemory(llama, 4096)
    expect(mha.bytesPerToken).toBe(2 * 32 * 32 * 128 * 2)
    expect(mha.bytes).toBe(mha.bytesPerToken * 4096)
    expect(kvCacheMemory({ ...llama, kvHeads: 8 }, 4096).relativeToMultiHead).toBe(1 / 4)
    expect(kvCacheMemory({ ...llama, kvHeads: 1 }, 10).valuesPerToken).toBe(2 * 32 * 128)
    expect(kvCacheMemory({ ...llama, latentDim: 512, ropeDim: 64 }, 1).valuesPerToken).toBe(32 * 576)
    expect(kvCacheMemory({ ...llama, window: 1024 }, 4096, 2).tokens).toBe(2048)
  })

  it('the online softmax over tiles equals full attention, causal or not', () => {
    const c = F.sdpa.plain
    const q = slice(slice(T(c.q), 0), 0)
    const k = slice(slice(T(c.k), 0), 0)
    const v = slice(slice(T(c.v), 0), 0)
    for (const causal of [false, true])
      for (const [qb, kb] of [
        [1, 1],
        [2, 3],
        [4, 4],
      ]) {
        const out = flashAttention(q, k, v, { causal, queryBlock: qb, keyBlock: kb })
        close(out, flat(scaledDotProductAttention(q, k, v, { causal }).output), 1e-13)
      }
    const s = run(flashAttentionSteps(q, k, v, { causal: true, queryBlock: 1, keyBlock: 2 }), undefined, 100)
    expect(s.skipped).toBeGreaterThan(0)
    expect(s.t).toBe(s.tiles.length)
  })
})

describe('position masks', () => {
  it('a window without causal is a symmetric band; with causal, the w keys ending at the query', () => {
    const pos = [0, 1, 2, 3, 4]
    const band = toFlat(positionMask(pos, pos, { window: 2 }))
    // Query 2 sees keys 1, 2, 3 (|p − q| < 2).
    expect(Array.from(band.slice(10, 15))).toEqual([0, 1, 1, 1, 0])
    const causal = toFlat(positionMask(pos, pos, { window: 2, causal: true }))
    expect(Array.from(causal.slice(10, 15))).toEqual([0, 1, 1, 0, 0])
  })
})
