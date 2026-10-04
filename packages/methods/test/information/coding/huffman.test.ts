/**
 * Huffman codes against a heapq reference (`fixtures/information/coding.json`): codewords, lengths, expected length,
 * length variance and the merges, for binary and ternary codes under both tie rules; the Kraft sum is 1, the expected
 * length matches a brute-force optimum for small K, and H ≤ E[ℓ] < H + 1. Then the step trace, the tree, canonical
 * codes, source extensions, and encoding and decoding.
 */
import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import { pathToRoot } from 'aifn-compute/graph'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  canonicalCode,
  huffmanCode,
  huffmanDummies,
  huffmanSteps,
  huffmanTree,
  kraftSum,
  prefixDecode,
  prefixEncode,
  sourceExtension,
  type HuffmanTies,
} from 'aifn-methods/information/coding'
import { fixture } from '../../fixtures'
import { expectProtocol } from '../../protocol'

type Ref = {
  arity: number
  ties: HuffmanTies
  dummies: number
  merges: number[][]
  codewords: string[]
  lengths: number[]
  expectedLength: number
  lengthVariance: number
  entropy: number
}
const ref =
  fixture<Record<string, { p: number[]; codes: Ref[]; optimum: Record<string, number> | null }>>('information/coding')

describe('Huffman codes against heapq', () => {
  for (const [name, { p, codes, optimum }] of Object.entries(ref))
    it(name, () => {
      for (const r of codes) {
        const options = { arity: r.arity, ties: r.ties }
        const code = huffmanCode(p, options)
        expect(huffmanDummies(p.length, r.arity)).toBe(r.dummies)
        expect(code.codewords).toEqual(r.codewords)
        expect(Array.from(toFlat(code.lengths))).toEqual(r.lengths)
        expect(code.expectedLength).toBeCloseTo(r.expectedLength, 12)
        expect(code.lengthVariance).toBeCloseTo(r.lengthVariance, 12)
        expect(code.entropy).toBeCloseTo(r.entropy, 12)
        // H ≤ E[ℓ] < H + 1 (in D-ary digits).
        expect(code.expectedLength).toBeGreaterThanOrEqual(code.entropy - 1e-12)
        expect(code.expectedLength).toBeLessThan(code.entropy + 1)
        // The tree is full: with its dummies (length = depth, weight 0) the Kraft sum is exactly 1.
        const t = code.tree
        const leafDepths = t.nodes.filter((n) => n.children.length === 0).map((n) => pathToRoot(t, n.id).length - 1)
        expect(kraftSum(leafDepths, r.arity)).toBeCloseTo(1, 12)
        if (r.dummies === 0) expect(kraftSum(code.lengths, r.arity)).toBeCloseTo(1, 12)
        if (optimum) expect(code.expectedLength).toBeCloseTo(optimum[String(r.arity)], 12)
        // The trace pops the same nodes, in the same order, as the heap.
        const tr = trace(huffmanSteps(p, options), undefined, 100)
        expect(tr.steps.slice(1).map((s) => [...s.popped])).toEqual(r.merges)
      }
    })
})

describe('Huffman steps, tree and tie rules', () => {
  const p = [0.4, 0.2, 0.2, 0.1, 0.1]
  it('records the queue, the pops, the insertion, the ties and the partial codewords', () => {
    const tr = trace(huffmanSteps(p), undefined, 10)
    const [s0, s1, s2] = tr.steps
    expect(s0.queue).toEqual([3, 4, 1, 2, 0])
    expect(s0.codewords).toEqual(['', '', '', '', ''])
    expect(s1.popped).toEqual([3, 4])
    expect(s1.created).toBe(5)
    // .1 + .1 = .2 ties with the two .2 leaves; minimum variance puts the merged node after them.
    expect(s1.queue).toEqual([1, 2, 5, 0])
    expect(s1.inserted).toBe(2)
    expect(s1.codewords).toEqual(['', '', '', '0', '1'])
    expect(s1.under).toEqual([[3], [4]])
    expect(s2.popped).toEqual([1, 2])
    expect(s2.tied).toEqual([5])
    expect(tr.steps.at(-1)!.done).toBe(true)
    expect(tr.meta.steps).toBe(4)
    const merged = trace(huffmanSteps(p, { ties: 'merged-first' }), undefined, 10).steps
    expect(merged[1].queue).toEqual([5, 1, 2, 0])
    expect(merged[2].popped).toEqual([5, 1])
  })
  it('the minimum-variance rule gives the smaller length variance at the same expected length', () => {
    const a = huffmanCode(p)
    const b = huffmanCode(p, { ties: 'merged-first' })
    expect(a.expectedLength).toBeCloseTo(b.expectedLength, 12)
    expect(a.lengthVariance).toBeCloseTo(0.16, 12)
    expect(b.lengthVariance).toBeCloseTo(1.36, 12)
  })
  it('follows the trace protocol, binary and ternary', () => {
    expectProtocol(huffmanSteps(p), undefined, { n: 4 })
    expectProtocol(huffmanSteps(p, { arity: 3, ties: 'merged-first' }), undefined, { n: 2 })
  })
  it('the tree spells the codewords; slots are digits; dummies are leaves', () => {
    const code = huffmanCode([0.3, 0.3, 0.2, 0.2], { arity: 3 })
    const t = code.tree
    expect(t.arity).toBe(3)
    expect(t.nodes.filter((n) => n.dummy)).toHaveLength(1)
    for (let k = 0; k < 4; k++) {
      const path = pathToRoot(t, k).reverse()
      expect(
        path
          .slice(1)
          .map((v) => t.edges[v]!.label)
          .join(''),
      ).toBe(code.codewords[k])
      expect(path.slice(1).map((v) => t.nodes[v].slot)).toEqual(path.slice(1).map((v) => t.edges[v]!.digit))
    }
    expect(t.nodes[t.root].weight).toBeCloseTo(1, 12)
    expect(huffmanCode([1]).codewords).toEqual(['0'])
    expect(huffmanTree(trace(huffmanSteps([1]), undefined, 1).steps[0]).nodes).toHaveLength(1)
  })
})

describe('canonical codes, extensions, encoding', () => {
  it('canonical codewords keep the lengths and are consecutive', () => {
    expect(canonicalCode([2, 1, 3, 3])).toEqual(['10', '0', '110', '111'])
    expect(canonicalCode([2, 2, 2, 3, 3])).toEqual(['00', '01', '10', '110', '111'])
    expect(canonicalCode([1, 1, 2, 2, 2], 3)).toEqual(['0', '1', '20', '21', '22'])
    expect(() => canonicalCode([1, 1, 1])).toThrow(DomainError)
    for (const [name, { p }] of Object.entries(ref)) {
      const code = huffmanCode(p)
      const canonical = canonicalCode(code.lengths)
      expect(
        canonical.map((c) => c.length),
        name,
      ).toEqual(Array.from(toFlat(code.lengths)))
      // Prefix-free.
      for (const a of canonical) for (const b of canonical) if (a !== b) expect(b.startsWith(a)).toBe(false)
    }
  })
  it('block coding: E[ℓ_n]/n stays within 1/n of H', () => {
    const p = [0.9, 0.1]
    const h = huffmanCode(p).entropy
    for (let n = 1; n <= 6; n++) {
      const ext = sourceExtension(p, n)
      expect(ext.shape).toEqual([2 ** n])
      expect(toFlat(ext).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      const perSymbol = huffmanCode(ext).expectedLength / n
      expect(perSymbol).toBeGreaterThanOrEqual(h - 1e-12)
      expect(perSymbol).toBeLessThan(h + 1 / n)
    }
    expect(Array.from(toFlat(sourceExtension([0.5, 0.25, 0.25], 2))).slice(0, 4)).toEqual([0.25, 0.125, 0.125, 0.125])
  })
  it('encodes and decodes by walking the tree', () => {
    const code = huffmanCode([0.4, 0.2, 0.2, 0.1, 0.1])
    const message = [0, 3, 1, 4, 0, 2]
    const enc = prefixEncode(code.codewords, message)
    expect(enc.digits).toBe(message.map((s) => code.codewords[s]).join(''))
    expect(enc.spans[1]).toEqual({ symbol: 3, start: 2, end: 5 })
    const dec = prefixDecode(code.tree, enc.digits)
    expect(dec.symbols).toEqual(message)
    expect(dec.rest).toBe('')
    expect(dec.walks[1]).toEqual(pathToRoot(code.tree, 3).reverse())
    expect(prefixDecode(code.tree, enc.digits + '1').rest).toBe('1')
  })
})
