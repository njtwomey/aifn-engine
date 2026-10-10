/**
 * Source codes for a discrete memoryless source, and the distances and size bounds of error-correcting codes.
 *
 * Source coding: Huffman codes (binary and $D$-ary, with a choice of tie-breaking rule) as a step-through algorithm,
 * Shannon–Fano and Shannon codes, canonical codewords from lengths, the $n$-th extension of a source for block coding,
 * encoding and decoding with a prefix code, the Kraft sum and the arithmetic-coding interval of a message. Each code
 * reports its expected length $\expect[\ell] = \sum_k p_k \ell_k$ beside the entropy $H_D(p)$, the bound
 * $H_D(p) \le \expect[\ell]$ of the source coding theorem (Cover and Thomas, 2006, ch. 5). Channel coding: Hamming
 * distance and weight, the minimum distance of a code, and the Hamming (sphere-packing), Singleton and Plotkin bounds
 * on the number of codewords.
 *
 * Symbols are indices $0, \dots, K - 1$, and probabilities may be any `Probabilities` (counts are normalised; a
 * negative entry throws `DomainError`). Codewords are strings of the digits $0, \dots, D - 1$, written 0–9 then a–z,
 * so $D \le 36$.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { treeFromChildren, type Tree } from 'aifn-compute/graph'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { flatProbabilities, type Probabilities } from 'aifn-compute/probability/information'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A prefix code for symbols $0, \dots, K - 1$ over a $D$-ary alphabet (digits 0–9 then a–z, so $D \le 36$). */
export type PrefixCode = {
  /** The codeword of each symbol (digits $0, \dots, D - 1$). */
  codewords: string[]
  /** The length $\ell_k$ of each codeword ($K$ values). */
  lengths: Tensor
  /** The code alphabet's size $D$ (2 for a binary code). */
  arity: number
  /** $\expect[\ell] = \sum_k p_k \ell_k$ in $D$-ary digits per symbol (bits when $D = 2$). */
  expectedLength: number
  /** $\sum_k p_k (\ell_k - \expect[\ell])^2$, the variance of the codeword length. */
  lengthVariance: number
  /**
   * The entropy $H_D(p) = -\sum_k p_k \log_D p_k$, the lower bound on the expected length of any $D$-ary prefix
   * code.
   */
  entropy: number
}

// ── Huffman's algorithm ──────────────────────────────────────────────────────────────────────────────────────────────

/**
 * How Huffman's algorithm orders nodes of equal weight in its queue. Every rule gives an optimal code (the same
 * expected length); they differ in the spread of the lengths.
 *
 * - `minimum-variance`: original leaves before merged nodes of the same weight, merged nodes oldest first. A merged
 *   node goes after the equal-weight originals, so it is merged as late as possible and the tree stays shallow; this
 *   gives the code of least length variance (and least maximum length) among Huffman codes (Schwartz, 1964, "An
 *   optimum encoding with minimum longest code and total number of digits", Information and Control 7(1)).
 * - `merged-first`: merged nodes before original leaves of the same weight, newest first, so a subtree just built is
 *   merged again at once and the tree grows deep.
 */
export type HuffmanTies = 'minimum-variance' | 'merged-first'

/** Options of `huffmanSteps` and `huffmanCode`. */
export type HuffmanOptions = {
  /**
   * The code alphabet's size $D$, an integer from 2 to 36 (default 2). $D > 2$ may pad the source with zero-weight
   * dummy leaves (`huffmanDummies`).
   */
  arity?: number
  /** The tie-breaking rule (default `minimum-variance`). */
  ties?: HuffmanTies
}

/** One node of Huffman's forest: a symbol's leaf, a dummy leaf (D-ary padding) or a merged node. */
export type HuffmanNode = {
  /** The node's index in the state's `nodes`. */
  id: number
  /** The probability of the symbols under the node (0 for a dummy). */
  weight: number
  /** The symbol of a leaf ($0, \dots, K - 1$); $-1$ for a merged node or a dummy. */
  symbol: number
  /** A zero-weight leaf added so that every merge takes exactly $D$ nodes. */
  dummy: boolean
  /** The node it was merged into, or null while it is a root in the queue. */
  parent: number | null
  /** Children in digit order: child $i$ hangs on the edge labelled $i$. Empty for a leaf. */
  children: readonly number[]
  /** The digit on the edge from its parent, or null for a root. */
  digit: number | null
  /** The step that created it (0 for leaves). */
  step: number
}

/** One state of Huffman's algorithm: the forest so far, its queue of roots, and what the last step did. */
export type HuffmanState = Status & {
  /** Merges done. */
  t: number
  /** The code alphabet's size $D$. */
  arity: number
  /** The tie-breaking rule in use. */
  ties: HuffmanTies
  /**
   * The number of real symbols $K$ (leaves $0, \dots, K - 1$); the $d$ dummies are ids $K, \dots, K + d - 1$, and
   * merged nodes follow in merge order.
   */
  symbols: number
  /** Every node so far. */
  nodes: readonly HuffmanNode[]
  /** The priority queue: the roots still to merge, in pop order (least weight first, ties by the rule). */
  queue: readonly number[]
  /** The $D$ nodes the last step popped, in pop order: `popped[i]` got digit $i$. Empty at step 0. */
  popped: readonly number[]
  /** The node the last step created (the popped nodes' parent), or $-1$ at step 0. */
  created: number
  /** Where the created node was inserted in the queue (its index in `queue`), or $-1$ at step 0. */
  inserted: number
  /**
   * Nodes of the same weight as the last popped one that stayed in the queue: when non-empty, the tie-breaking rule,
   * not the weights, chose which nodes the last step popped.
   */
  tied: readonly number[]
  /** The symbols under each popped node (dummies left out): each gained the popped node's digit as its first digit. */
  under: readonly (readonly number[])[]
  /**
   * Each symbol's codeword so far. A merge puts a digit in front of every symbol under the merged nodes, so codewords
   * grow from their last digit to their first as the queue shrinks; '' for a symbol still alone in the queue.
   */
  codewords: readonly string[]
  /** One root left. */
  done: boolean
}

/** Relative tolerance under which two weights count as equal (sums of probabilities round). */
const TIE = 1e-12

/**
 * The number $d$ of zero-weight dummies that make $K + d - 1$ divisible by $D - 1$, so that every merge of a $D$-ary
 * Huffman code takes $D$ nodes. Throws `DomainError` for an arity outside $2, \dots, 36$.
 *
 * @param symbols The number of real symbols $K$; 0 or 1 needs no dummies.
 * @param arity The code alphabet's size $D$.
 * @returns The number of dummies, from 0 to $D - 2$ (always 0 for a binary code).
 *
 * @example Ternary codes need a dummy for an even number of symbols
 * print('K = 2 … 7:', [2, 3, 4, 5, 6, 7].map((K) => huffmanDummies(K, 3)))
 * print('binary, K = 5:', huffmanDummies(5))
 */
export function huffmanDummies(symbols: number, arity = 2): number {
  checkArity(arity, 'huffmanDummies')
  if (symbols <= 1) return 0
  return (arity - 1 - ((symbols - 1) % (arity - 1))) % (arity - 1)
}

/**
 * Throws `DomainError` unless the arity is an integer from 2 to 36 (the digits 0–9 and a–z).
 *
 * @param arity The code alphabet's size $D$.
 * @param where The caller's name, for error messages.
 */
function checkArity(arity: number, where: string): void {
  if (!(Number.isInteger(arity) && arity >= 2 && arity <= 36))
    throw new DomainError(where, `${where}: the arity must be an integer in 2 … 36 (digits 0–9, a–z)`)
}

/**
 * Queue order under a tie rule: lighter nodes pop first, and weights equal within a relative $10^{-12}$ are ordered
 * by the rule (`HuffmanTies`).
 *
 * @param a A node in or entering the queue.
 * @param b Another node.
 * @param ties The tie-breaking rule.
 * @returns True when `a` pops before `b`.
 */
function popsBefore(a: HuffmanNode, b: HuffmanNode, ties: HuffmanTies): boolean {
  const scale = Math.max(Math.abs(a.weight), Math.abs(b.weight), 1e-300)
  if (Math.abs(a.weight - b.weight) > TIE * scale) return a.weight < b.weight
  const am = a.children.length > 0
  const bm = b.children.length > 0
  if (ties === 'minimum-variance') return am !== bm ? !am : a.id < b.id
  if (am !== bm) return am
  return am ? a.id > b.id : a.id < b.id
}

/**
 * Whether two weights are equal within a relative $10^{-12}$, the tolerance of the tie rules.
 *
 * @param a A weight.
 * @param b Another weight.
 * @returns True when they count as tied.
 */
const sameWeight = (a: number, b: number) => Math.abs(a - b) <= TIE * Math.max(Math.abs(a), Math.abs(b), 1e-300)

/**
 * Huffman's algorithm as a step-through `Algorithm` (Huffman, 1952, "A method for the construction of
 * minimum-redundancy codes", Proc. IRE 40) on the probabilities (any shape, flattened); no start. The queue starts
 * with the $K$ symbol leaves (and, for $D > 2$, `huffmanDummies(K, D)` zero-weight dummy leaves) in pop order. Each
 * step pops the $D$ least-weight roots, gives the $i$-th popped digit $i$, makes them the children of a new node whose
 * weight is their sum, and inserts that node back into the queue at its place under the tie rule (`HuffmanTies`).
 * Every symbol under a popped node gains that digit in front of its codeword. The run is done when one root is left,
 * after $(K + d - 1) / (D - 1)$ steps; a step from a finished state returns it unchanged. Each state records the
 * queue, the popped nodes, the created node and where it went, the ties the rule decided, and the partial codewords.
 * `huffmanTree(state)` turns the final state into a `Tree`. Throws `DomainError` for no symbols, a negative
 * probability, a bad arity or an unknown tie rule.
 *
 * @param probabilities The symbol probabilities $p_k$ (or counts), normalised to sum to 1; a table is read
 *   row-major.
 * @param options The code's alphabet and tie rule.
 * @param options.arity The code alphabet's size $D$, from 2 to 36.
 * @param options.ties The tie-breaking rule for nodes of equal weight.
 * @returns The algorithm: `init` takes no start, `step` is one merge.
 *
 * @example The first merge joins the two least likely symbols
 * const s = run(huffmanSteps([0.4, 0.3, 0.2, 0.1]), undefined, 1)
 * print('popped =', s.popped, 'into node', s.created)
 * print('queue =', s.queue)
 * print('codewords so far =', s.codewords)
 *
 * @example Run to the end, the codewords are complete
 * const s = run(huffmanSteps([0.4, 0.3, 0.2, 0.1]), undefined, 10)
 * print('merges =', s.t, 'done =', s.done)
 * print('codewords =', s.codewords)
 */
export function huffmanSteps(
  probabilities: Probabilities,
  { arity = 2, ties = 'minimum-variance' }: HuffmanOptions = {},
): Algorithm<void, HuffmanState> {
  const p = flatProbabilities(probabilities, 'huffmanSteps')
  if (p.length === 0) throw new DomainError('huffmanSteps', 'huffmanSteps: needs at least one symbol')
  checkArity(arity, 'huffmanSteps')
  if (ties !== 'minimum-variance' && ties !== 'merged-first')
    throw new DomainError('huffmanSteps', `huffmanSteps: unknown ties ${ties}`)
  const K = p.length
  const dummies = huffmanDummies(K, arity)
  return {
    name: 'huffman',
    init: () => {
      const nodes: HuffmanNode[] = []
      for (let i = 0; i < K + dummies; i++)
        nodes.push({
          id: i,
          weight: i < K ? p[i] : 0,
          symbol: i < K ? i : -1,
          dummy: i >= K,
          parent: null,
          children: [],
          digit: null,
          step: 0,
        })
      const queue = nodes.map((n) => n.id).sort((a, b) => (popsBefore(nodes[a], nodes[b], ties) ? -1 : 1))
      return {
        t: 0,
        arity,
        ties,
        symbols: K,
        nodes,
        queue,
        popped: [],
        created: -1,
        inserted: -1,
        tied: [],
        under: [],
        codewords: p.map(() => ''),
        done: queue.length === 1,
      }
    },
    step: (s) => {
      if (s.done) return s
      const popped = s.queue.slice(0, arity)
      const rest = s.queue.slice(arity)
      const id = s.nodes.length
      const t = s.t + 1
      const lastWeight = s.nodes[popped[popped.length - 1]].weight
      const tied = rest.filter((v) => sameWeight(s.nodes[v].weight, lastWeight))
      const created: HuffmanNode = {
        id,
        weight: popped.reduce((sum, v) => sum + s.nodes[v].weight, 0),
        symbol: -1,
        dummy: false,
        parent: null,
        children: popped,
        digit: null,
        step: t,
      }
      const nodes = s.nodes.map((n) => {
        const i = popped.indexOf(n.id)
        return i < 0 ? n : { ...n, parent: id, digit: i }
      })
      nodes.push(created)
      const codewords = [...s.codewords]
      const under = popped.map((v, digit) => {
        const symbols: number[] = []
        const stack = [v]
        while (stack.length) {
          const w = nodes[stack.pop()!]
          if (w.children.length) stack.push(...w.children)
          else if (!w.dummy) symbols.push(w.symbol)
        }
        symbols.sort((a, b) => a - b)
        for (const k of symbols) codewords[k] = digit.toString(36) + codewords[k]
        return symbols
      })
      // Binary search for the first queued node the new one pops before.
      let lo = 0
      let hi = rest.length
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (popsBefore(nodes[rest[mid]], created, ties)) lo = mid + 1
        else hi = mid
      }
      const queue = [...rest.slice(0, lo), id, ...rest.slice(lo)]
      return {
        t,
        arity,
        ties,
        symbols: K,
        nodes,
        queue,
        popped,
        created: id,
        inserted: lo,
        tied,
        under,
        codewords,
        done: queue.length === 1,
      }
    },
    done: (s) => s.done,
  }
}

/**
 * Node data of a Huffman tree: the node's `weight` (the probability under it), its `symbol` ($-1$ for a merged node or
 * a dummy) and whether it is a `dummy` leaf, as in `HuffmanNode`.
 */
export type HuffmanNodeData = { weight: number; symbol: number; dummy: boolean }
/** Edge data of a Huffman tree: the `digit` the branch appends. */
export type HuffmanEdgeData = { digit: number }
/**
 * A Huffman tree as an `aifn-compute/graph` `Tree` (`arity` $D$): node ids are those of the `HuffmanState` (symbol
 * $k$ is node $k$, dummies follow, merged nodes in merge order, the root last). Edges are labelled with their digit
 * and each child's `slot` is its digit; a symbol's codeword is the edge labels on its path from the root.
 */
export type HuffmanTree = Tree<HuffmanNodeData, HuffmanEdgeData>

/**
 * The Huffman tree of a finished `huffmanSteps` state as a `Tree`, rooted at the last node created. Throws
 * `DomainError` when the state is not finished.
 *
 * @param state A `huffmanSteps` state with one root left (`done`).
 * @returns The tree, with node ids as in the state and each edge carrying its digit.
 *
 * @example The tree of a three-symbol code
 * const tree = huffmanTree(run(huffmanSteps([0.5, 0.25, 0.25]), undefined, 10))
 * print('root =', tree.root, 'with children', tree.nodes[tree.root].children)
 * print('leaves:', tree.nodes.filter((n) => n.children.length === 0).map((n) => `${n.symbol} (${n.weight})`))
 * print('edge digits:', tree.edges.map((e) => (e ? e.digit : null)))
 */
export function huffmanTree(state: HuffmanState): HuffmanTree {
  if (!state.done)
    throw new DomainError('huffmanTree', 'huffmanTree: the state is not finished (the queue holds more than one root)')
  const tree = treeFromChildren<HuffmanNodeData, HuffmanEdgeData>(
    state.nodes.map((n) => [...n.children]),
    state.queue[0],
    {
      data: (i) => ({ weight: state.nodes[i].weight, symbol: state.nodes[i].symbol, dummy: state.nodes[i].dummy }),
      edge: (c) => ({ digit: state.nodes[c].digit!, label: state.nodes[c].digit!.toString(36) }),
    },
  )
  for (const n of tree.nodes) if (n.parent !== null) n.slot = state.nodes[n.id].digit!
  return { ...tree, arity: state.arity }
}

/**
 * The entropy $-\sum_k p_k \log_b p_k$ of a distribution, zero-probability symbols contributing nothing.
 *
 * @param p The probabilities, summing to 1.
 * @param base The base $b$ of the logarithm (the code's arity).
 * @returns The entropy in base-$b$ digits.
 */
function entropyBase(p: number[], base: number): number {
  let h = 0
  for (const v of p) if (v > 0) h -= v * Math.log(v)
  return h / Math.log(base)
}

/**
 * A `PrefixCode` from its codewords: their lengths, expected length, length variance and the source's entropy.
 *
 * @param p The symbol probabilities, summing to 1, aligned with `codewords`.
 * @param codewords The codeword of each symbol.
 * @param arity The code alphabet's size $D$, the base of the entropy.
 * @returns The code with its statistics.
 */
function finish(p: number[], codewords: string[], arity = 2): PrefixCode {
  const lengths = Int32Array.from(codewords, (c) => c.length)
  let expected = 0
  p.forEach((v, k) => (expected += v * lengths[k]))
  let variance = 0
  p.forEach((v, k) => (variance += v * (lengths[k] - expected) ** 2))
  return {
    codewords,
    lengths: fromData(lengths, [lengths.length]),
    arity,
    expectedLength: expected,
    lengthVariance: variance,
    entropy: entropyBase(p, arity),
  }
}

/**
 * A Huffman code: `huffmanSteps` run to the end. The result is an optimal $D$-ary prefix code, with
 * $H_D(p) \le \expect[\ell] < H_D(p) + 1$; the tie rule changes the lengths' variance, never $\expect[\ell]$. A single
 * symbol gets the codeword `'0'`. Throws as `huffmanSteps` does.
 *
 * @param probabilities The symbol probabilities $p_k$ (or counts), normalised to sum to 1; a table is read
 *   row-major.
 * @param options The code's alphabet size and tie rule; see `HuffmanOptions`.
 * @returns The code with its expected length, length variance and entropy, and its `tree` (a `HuffmanTree`).
 *
 * @example Four symbols: the expected length against the entropy
 * const code = huffmanCode([0.4, 0.3, 0.2, 0.1])
 * print('codewords =', code.codewords)
 * print('E[ℓ] =', code.expectedLength, 'bits; H(p) =', code.entropy, 'bits')
 *
 * @example The tie rule changes the spread of the lengths, not their mean
 * const p = [0.4, 0.2, 0.2, 0.1, 0.1]
 * for (const ties of ['minimum-variance', 'merged-first']) {
 *   const code = huffmanCode(p, { ties })
 *   print(ties, code.codewords, 'E[ℓ] =', code.expectedLength, 'var =', code.lengthVariance)
 * }
 *
 * @example A ternary code
 * const code = huffmanCode([0.3, 0.2, 0.2, 0.1, 0.1, 0.1], { arity: 3 })
 * print('codewords =', code.codewords)
 * print('E[ℓ] =', code.expectedLength, 'trits; H₃(p) =', code.entropy)
 */
export function huffmanCode(
  probabilities: Probabilities,
  options: HuffmanOptions = {},
): PrefixCode & { tree: HuffmanTree } {
  const p = flatProbabilities(probabilities, 'huffmanCode')
  if (p.length === 0) throw new DomainError('huffmanCode', 'huffmanCode: needs at least one symbol')
  const steps = huffmanSteps(p, options)
  const s = run(steps, undefined, p.length + huffmanDummies(p.length, options.arity ?? 2))
  const tree = huffmanTree(s)
  return { ...finish(p, p.length === 1 ? ['0'] : [...s.codewords], s.arity), tree }
}

/**
 * Canonical codewords for given lengths (Schwartz and Kallick, 1964): symbols sorted by (length, index) get
 * consecutive $D$-ary numbers, each written with its length's number of digits; going to a longer length appends
 * zeros ($c \leftarrow (c + 1) \cdot D^{\Delta\ell}$). Any prefix code's lengths give a canonical code with the same
 * lengths, so a decoder needs only the lengths. A length of 0 gives `''`. Throws `DomainError` when the nonzero
 * lengths break Kraft's inequality, or for a bad arity.
 *
 * @param lengths The codeword length $\ell_k$ of each symbol: non-negative integers.
 * @param arity The code alphabet's size $D$, from 2 to 36.
 * @returns The codeword of each symbol, in symbol order.
 *
 * @example Codewords from the lengths of a Huffman code
 * print('binary:', canonicalCode([2, 1, 3, 3]))
 * print('ternary:', canonicalCode([1, 1, 2, 2, 2], 3))
 *
 * @example Lengths that no prefix code has are refused
 * try {
 *   canonicalCode([1, 1, 2])
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function canonicalCode(lengths: ArrayLike<number> | Tensor, arity = 2): string[] {
  checkArity(arity, 'canonicalCode')
  const ls = isTensor(lengths) ? toFlat(lengths) : Array.from(lengths)
  if (
    kraftSum(
      ls.filter((l) => l > 0),
      arity,
    ) >
    1 + 1e-12
  )
    throw new DomainError('canonicalCode', 'canonicalCode: the lengths break Kraft’s inequality (Σ D^−ℓ > 1)')
  const order = ls
    .map((_, k) => k)
    .filter((k) => ls[k] > 0)
    .sort((a, b) => ls[a] - ls[b] || a - b)
  const out = ls.map(() => '')
  const D = BigInt(arity)
  let code = -1n
  let length = 0
  for (const k of order) {
    code = (code + 1n) * D ** BigInt(ls[k] - length)
    length = ls[k]
    out[k] = code.toString(arity).padStart(length, '0')
  }
  return out
}

/**
 * The $n$-th extension of a memoryless source: the $K^n$ probabilities of blocks of $n$ symbols, $\prod_i p_{s_i}$ over
 * the block, in lexicographic order (the first symbol most significant). Coding blocks with a Huffman code gives
 * $\expect[\ell_n] / n \to H(p)$ digits per symbol, since $H(p) \le \expect[\ell_n] / n < H(p) + 1/n$ (Cover and
 * Thomas, 2006, §5.4). Throws `DomainError` unless $n$ is a positive integer.
 *
 * @param probabilities The symbol probabilities (or counts), normalised to sum to 1.
 * @param n The block length $n$.
 * @returns The $K^n$ block probabilities; block $(s_1, \dots, s_n)$ is entry $\sum_i s_i K^{n - i}$.
 *
 * @example Block coding approaches the entropy
 * print('blocks of 2:', sourceExtension([0.9, 0.1], 2))
 * for (const n of [1, 2, 3]) {
 *   const code = huffmanCode(sourceExtension([0.9, 0.1], n))
 *   print(`n = ${n}: E[ℓ]/n =`, code.expectedLength / n, 'H(p) =', code.entropy / n)
 * }
 */
export function sourceExtension(probabilities: Probabilities, n: number): Tensor {
  const p = flatProbabilities(probabilities, 'sourceExtension')
  if (!(Number.isInteger(n) && n >= 1))
    throw new DomainError('sourceExtension', 'sourceExtension: n must be a positive integer')
  let out = Float64Array.of(1)
  for (let i = 0; i < n; i++) {
    const next = new Float64Array(out.length * p.length)
    for (let a = 0; a < out.length; a++) for (let b = 0; b < p.length; b++) next[a * p.length + b] = out[a] * p[b]
    out = next
  }
  return fromData(out, [out.length])
}

/**
 * A message encoded with a prefix code: the `digits` string, and in `spans` the `symbol` of each message position
 * with the `start` and (exclusive) `end` of its codeword in the string.
 */
export type Encoded = { digits: string; spans: { symbol: number; start: number; end: number }[] }

/**
 * Encode a message (symbol indices) by concatenating codewords. Throws `DomainError` for a symbol with no codeword
 * (out of range, or an empty codeword).
 *
 * @param codewords The codeword of each symbol, as a `PrefixCode` holds them.
 * @param message The symbols to encode, in order.
 * @returns The digit string and the span of each symbol's codeword in it.
 *
 * @example Encode with a Huffman code
 * const code = huffmanCode([0.4, 0.3, 0.2, 0.1])
 * const { digits, spans } = prefixEncode(code.codewords, [0, 1, 2, 3, 0])
 * print('digits =', digits)
 * print('symbol 2 at', spans[2].start, 'to', spans[2].end)
 */
export function prefixEncode(codewords: readonly string[], message: ArrayLike<number>): Encoded {
  let digits = ''
  const spans: Encoded['spans'] = []
  for (let i = 0; i < message.length; i++) {
    const s = message[i]
    const c = codewords[s]
    if (c === undefined || c === '') throw new DomainError('prefixEncode', `prefixEncode: symbol ${s} has no codeword`)
    spans.push({ symbol: s, start: digits.length, end: digits.length + c.length })
    digits += c
  }
  return { digits, spans }
}

/** A digit string decoded by walking a code tree. */
export type Decoded = {
  /** The symbols decoded. */
  symbols: number[]
  /** For each decoded symbol, the nodes walked from the root to its leaf. */
  walks: number[][]
  /** Digits at the end that stop inside the tree (a codeword cut short); '' when the string decodes exactly. */
  rest: string
}

/**
 * Decode a digit string by walking a `HuffmanTree`: from the root, follow the child whose edge digit is the next
 * digit; at a leaf, emit its symbol and go back to the root. A prefix code needs no separators, since no codeword is
 * the start of another. A one-leaf tree decodes every digit, whatever it is, as its one symbol. Throws `DomainError`
 * on a digit with no edge or a walk that reaches a dummy leaf.
 *
 * @param tree The code tree, as `huffmanCode` or `huffmanTree` returns it.
 * @param digits The digit string to decode.
 * @returns The symbols decoded, the walk of node ids behind each, and any digits left over at the end.
 *
 * @example Decode what `prefixEncode` wrote
 * const code = huffmanCode([0.4, 0.3, 0.2, 0.1])
 * const { digits } = prefixEncode(code.codewords, [0, 1, 2, 3, 0])
 * print('decoded =', prefixDecode(code.tree, digits).symbols)
 * print('cut short:', prefixDecode(code.tree, digits.slice(0, 5)))
 */
export function prefixDecode(tree: HuffmanTree, digits: string): Decoded {
  const symbols: number[] = []
  const walks: number[][] = []
  let walk = [tree.root]
  let start = 0
  // A one-leaf tree: every digit is the one symbol.
  if (tree.nodes[tree.root].children.length === 0) {
    for (let i = 0; i < digits.length; i++) {
      symbols.push(tree.nodes[tree.root].symbol)
      walks.push([tree.root])
    }
    return { symbols, walks, rest: '' }
  }
  for (let i = 0; i < digits.length; i++) {
    const v = walk[walk.length - 1]
    const next = tree.nodes[v].children.find((c) => tree.edges[c]!.digit === parseInt(digits[i], 36))
    if (next === undefined)
      throw new DomainError('prefixDecode', `prefixDecode: no edge for digit ${digits[i]} at position ${i}`)
    walk.push(next)
    const node = tree.nodes[next]
    if (node.children.length === 0) {
      if (node.dummy)
        throw new DomainError('prefixDecode', `prefixDecode: digits ${digits.slice(start, i + 1)} reach a dummy leaf`)
      symbols.push(node.symbol)
      walks.push(walk)
      walk = [tree.root]
      start = i + 1
    }
  }
  return { symbols, walks, rest: digits.slice(start) }
}

/**
 * A binary Shannon–Fano code by Fano's method (Fano, 1949, "The transmission of information", MIT RLE TR 65): sort
 * the symbols by decreasing probability (stable), split the list where the two parts' totals are closest, give the
 * first part 0 and the second 1, and recurse. Not always optimal; its expected length is below $H(p) + 2$. A single
 * symbol gets the codeword `'0'`; zero-probability symbols get codewords too.
 *
 * @param probabilities The symbol probabilities (or counts), normalised to sum to 1.
 * @returns The binary code with its expected length and entropy.
 *
 * @example Fano's code beside the entropy
 * const code = shannonFanoCode([0.4, 0.3, 0.2, 0.1])
 * print('codewords =', code.codewords)
 * print('E[ℓ] =', code.expectedLength, 'H(p) =', code.entropy)
 */
export function shannonFanoCode(probabilities: Probabilities): PrefixCode {
  const p = flatProbabilities(probabilities, 'shannonFanoCode')
  const order = p.map((_, k) => k).sort((a, b) => p[b] - p[a] || a - b)
  const codewords = new Array<string>(p.length).fill('')
  const split = (items: number[], prefix: string) => {
    if (items.length === 1) {
      codewords[items[0]] = prefix || '0'
      return
    }
    const total = items.reduce((s, k) => s + p[k], 0)
    let left = 0
    let best = 1
    let bestGap = Infinity
    for (let i = 1; i < items.length; i++) {
      left += p[items[i - 1]]
      const gap = Math.abs(total - 2 * left)
      if (gap < bestGap) {
        bestGap = gap
        best = i
      }
    }
    split(items.slice(0, best), prefix + '0')
    split(items.slice(best), prefix + '1')
  }
  split(order, '')
  return finish(p, codewords)
}

/**
 * The binary Shannon code (Shannon, 1948, §9): symbols sorted by decreasing probability get lengths
 * $\ell_k = \lceil -\log_2 p_k \rceil$ (at least 1) and, as codewords, the first $\ell_k$ bits of the binary
 * expansion of the cumulative probability of the symbols before them. Its expected length is below $H(p) + 1$ (except
 * for a single symbol, whose one bit is $H(p) + 1$). Zero-probability symbols get no codeword (`''`).
 *
 * @param probabilities The symbol probabilities (or counts), normalised to sum to 1.
 * @returns The binary code with its expected length and entropy.
 *
 * @example Shannon's code is longer than Huffman's, but within a bit of the entropy
 * const p = [0.4, 0.3, 0.2, 0.1]
 * const code = shannonCode(p)
 * print('codewords =', code.codewords)
 * print('E[ℓ] =', code.expectedLength, 'Huffman:', huffmanCode(p).expectedLength, 'H(p) =', code.entropy)
 */
export function shannonCode(probabilities: Probabilities): PrefixCode {
  const p = flatProbabilities(probabilities, 'shannonCode')
  const order = p.map((_, k) => k).sort((a, b) => p[b] - p[a] || a - b)
  const codewords = new Array<string>(p.length).fill('')
  let cumulative = 0
  for (const k of order) {
    if (p[k] === 0) continue
    const length = Math.max(1, Math.ceil(-Math.log2(p[k]) - 1e-12))
    let bits = ''
    let f = cumulative
    for (let i = 0; i < length; i++) {
      f *= 2
      const bit = f >= 1 ? 1 : 0
      bits += bit
      f -= bit
    }
    codewords[k] = bits
    cumulative += p[k]
  }
  return finish(p, codewords)
}

/**
 * The Kraft sum $\sum_k D^{-\ell_k}$ of codeword lengths for a $D$-ary alphabet. A prefix code with these lengths
 * exists if and only if the sum is at most 1 (Kraft, 1949; McMillan, 1956, for uniquely decodable codes).
 *
 * @param lengths The codeword lengths $\ell_k$; a length of 0 adds 1 to the sum.
 * @param arity The code alphabet's size $D$.
 * @returns The sum.
 *
 * @example A complete code sums to 1; impossible lengths exceed it
 * print('[1, 2, 3, 3]:', kraftSum([1, 2, 3, 3]))
 * print('[1, 1, 2]:', kraftSum([1, 1, 2]))
 * print('ternary [1, 1, 2, 2, 2]:', kraftSum([1, 1, 2, 2, 2], 3))
 */
export function kraftSum(lengths: ArrayLike<number> | Tensor, arity = 2): number {
  const values = isTensor(lengths) ? toFlat(lengths) : Array.from(lengths)
  return values.reduce((s, l) => s + arity ** -l, 0)
}

/** The interval of a message under arithmetic coding, and how it narrows symbol by symbol. */
export type ArithmeticInterval = {
  /** The lower end of the final interval (included). */
  low: number
  /** The upper end of the final interval (excluded). */
  high: number
  /** $\text{high} - \text{low} = \prod_i p(s_i)$, the probability of the message. */
  width: number
  /**
   * $\lceil -\log_2 \text{width} \rceil + 1$: enough bits to name a binary fraction inside the interval (Cover and
   * Thomas, 2006, §13.3).
   */
  bits: number
  /** The interval after each symbol (entry 0 is $[0, 1)$). */
  steps: { low: number; high: number }[]
}

/**
 * The arithmetic-coding interval of a message (symbol indices) for i.i.d. symbols with probabilities $p$: start from
 * $[0, 1)$ and, for each symbol $s$, keep the sub-interval $[F(s - 1), F(s))$ of the current one, with $F$ the
 * cumulative distribution in index order (Rissanen, 1976; Witten, Neal and Cleary, 1987). In double precision, so
 * messages whose probability underflows (about $10^{-300}$) cannot be represented: the width then reaches 0. Throws
 * `DomainError` for a symbol that is not an index of `probabilities`.
 *
 * @param message The symbols, in order.
 * @param probabilities The symbol probabilities (or counts), normalised to sum to 1.
 * @returns The final interval, its width and the bits to code it, and the interval after each symbol.
 *
 * @example The interval narrows by each symbol's probability
 * const { low, high, width, bits } = arithmeticInterval([0, 1, 0], [0.8, 0.2])
 * print('[low, high) =', [low, high])
 * print('width =', width, '= 0.8 · 0.2 · 0.8; bits =', bits)
 */
export function arithmeticInterval(message: ArrayLike<number>, probabilities: Probabilities): ArithmeticInterval {
  const p = flatProbabilities(probabilities, 'arithmeticInterval')
  const cumulative = [0]
  for (const v of p) cumulative.push(cumulative[cumulative.length - 1] + v)
  let low = 0
  let high = 1
  const steps = [{ low, high }]
  for (let i = 0; i < message.length; i++) {
    const s = message[i]
    if (!(Number.isInteger(s) && s >= 0 && s < p.length))
      throw new DomainError('arithmeticInterval', `arithmeticInterval: bad symbol ${s}`)
    const width = high - low
    high = low + width * cumulative[s + 1]
    low = low + width * cumulative[s]
    steps.push({ low, high })
  }
  const width = high - low
  return { low, high, width, bits: Math.ceil(-Math.log2(width)) + 1, steps }
}

// ── Error-correcting codes ───────────────────────────────────────────────────────────────────────────────────────────

/** A word: a string of symbols or an array of numbers. Two words are compared position by position with `!==`. */
export type Word = string | ArrayLike<number>

/**
 * The number of positions where two words of equal length differ (Hamming, 1950). Throws `ShapeError` for words of
 * different lengths.
 *
 * @param a A word.
 * @param b A word of the same length and kind (a character `'1'` differs from a number 1).
 * @returns The Hamming distance.
 *
 * @example Two binary words
 * print(hammingDistance('10110', '11100'))
 */
export function hammingDistance(a: Word, b: Word): number {
  if (a.length !== b.length) throw new ShapeError('hammingDistance', 'hammingDistance: words of different lengths')
  let d = 0
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++
  return d
}

/**
 * The number of non-zero positions of a word (characters other than `'0'` in a string).
 *
 * @param a The word.
 * @returns Its Hamming weight.
 *
 * @example A string and an array
 * print(hammingWeight('10110'), hammingWeight([0, 2, 0, 1]))
 */
export function hammingWeight(a: Word): number {
  let w = 0
  for (let i = 0; i < a.length; i++) if (typeof a === 'string' ? a[i] !== '0' : a[i] !== 0) w++
  return w
}

/**
 * The minimum Hamming distance between distinct codewords of a code ($\infty$ for fewer than two codewords), by
 * comparing every pair. Throws `ShapeError` when two codewords differ in length.
 *
 * @param code The codewords, all of one length.
 * @returns The least distance between two of them.
 *
 * @example The even-weight code detects one error; the repetition code corrects two
 * print('even weight:', minimumDistance(['000', '011', '101', '110']))
 * print('repetition:', minimumDistance(['00000', '11111']))
 */
export function minimumDistance(code: readonly Word[]): number {
  let best = Infinity
  for (let i = 0; i < code.length; i++)
    for (let j = i + 1; j < code.length; j++) best = Math.min(best, hammingDistance(code[i], code[j]))
  return best
}

/**
 * $\binom{n}{k}$ as a float, by the running product (exact while below $2^{53}$).
 *
 * @param n The number of items.
 * @param k The number chosen, $0 \le k \le n$.
 * @returns The binomial coefficient, rounded to an integer.
 */
function choose(n: number, k: number): number {
  let c = 1
  for (let i = 1; i <= k; i++) c = (c * (n - k + i)) / i
  return Math.round(c)
}

/**
 * Throws `DomainError` unless $n$ is a positive integer, $d$ an integer with $1 \le d \le n$ and $q$ an integer
 * $\ge 2$.
 *
 * @param n The code length.
 * @param d The minimum distance.
 * @param q The alphabet size.
 * @param where The caller's name, for error messages.
 */
function checkCode(n: number, d: number, q: number, where: string): void {
  if (!(Number.isInteger(n) && n >= 1)) throw new DomainError(where, `${where}: n must be a positive integer`)
  if (!(Number.isInteger(d) && d >= 1 && d <= n)) throw new DomainError(where, `${where}: need 1 ≤ d ≤ n`)
  if (!(Number.isInteger(q) && q >= 2)) throw new DomainError(where, `${where}: q must be an integer ≥ 2`)
}

/**
 * The Hamming (sphere-packing) bound on the number of codewords $A_q(n, d)$ of a $q$-ary code of length $n$ and
 * minimum distance $d$: $\lfloor q^n / \sum_{i \le t} \binom{n}{i}(q - 1)^i \rfloor$ with
 * $t = \lfloor (d - 1)/2 \rfloor$ (Hamming, 1950). Perfect codes attain it. Throws `DomainError` for parameters
 * outside $n \ge 1$, $1 \le d \le n$, $q \ge 2$ (integers).
 *
 * @param n The code length.
 * @param d The minimum distance.
 * @param q The alphabet size.
 * @returns The bound.
 *
 * @example The Hamming $(7, 4)$ code is perfect: 16 codewords
 * print('A(7, 3) ≤', hammingBound(7, 3))
 * print('A(7, 5) ≤', hammingBound(7, 5))
 */
export function hammingBound(n: number, d: number, q = 2): number {
  checkCode(n, d, q, 'hammingBound')
  const t = Math.floor((d - 1) / 2)
  let volume = 0
  for (let i = 0; i <= t; i++) volume += choose(n, i) * (q - 1) ** i
  return Math.floor(q ** n / volume)
}

/**
 * The Singleton bound $A_q(n, d) \le q^{n - d + 1}$ (Singleton, 1964). MDS codes (e.g. Reed–Solomon) attain it.
 * Throws `DomainError` for parameters outside $n \ge 1$, $1 \le d \le n$, $q \ge 2$ (integers).
 *
 * @param n The code length.
 * @param d The minimum distance.
 * @param q The alphabet size.
 * @returns The bound.
 *
 * @example Binary codes, and a Reed–Solomon length over $q = 7$
 * print('A(7, 3) ≤', singletonBound(7, 3))
 * print('A_7(6, 3) ≤', singletonBound(6, 3, 7))
 */
export function singletonBound(n: number, d: number, q = 2): number {
  checkCode(n, d, q, 'singletonBound')
  return q ** (n - d + 1)
}

/**
 * The Plotkin bound (Plotkin, 1960). For binary codes, with $d$ even: $A \le 2\lfloor d/(2d - n) \rfloor$ when
 * $2d > n$ and $A \le 4d$ when $n = 2d$; with $d$ odd: $A \le 2\lfloor (d + 1)/(2d + 1 - n) \rfloor$ when
 * $2d + 1 > n$ and $A \le 4d + 4$ when $n = 2d + 1$ (MacWilliams and Sloane, 1977, ch. 2 Theorem 8). For $q$-ary
 * codes with $\theta = 1 - 1/q$ and $d > \theta n$: $A \le \lfloor d/(d - \theta n) \rfloor$. Returns $\infty$ when
 * the bound does not apply (the minimum distance is too small relative to $n$). Throws `DomainError` for parameters
 * outside $n \ge 1$, $1 \le d \le n$, $q \ge 2$ (integers).
 *
 * @param n The code length.
 * @param d The minimum distance.
 * @param q The alphabet size.
 * @returns The bound, or $\infty$.
 *
 * @example Strong for large distances, silent for small ones
 * print('A(8, 5) ≤', plotkinBound(8, 5))
 * print('A(7, 3) ≤', plotkinBound(7, 3))
 * print('A(10, 3) ≤', plotkinBound(10, 3))
 */
export function plotkinBound(n: number, d: number, q = 2): number {
  checkCode(n, d, q, 'plotkinBound')
  if (q === 2) {
    if (d % 2 === 0) {
      if (2 * d > n) return 2 * Math.floor(d / (2 * d - n))
      if (n === 2 * d) return 4 * d
    } else {
      if (2 * d + 1 > n) return 2 * Math.floor((d + 1) / (2 * d + 1 - n))
      if (n === 2 * d + 1) return 4 * d + 4
    }
    return Infinity
  }
  const theta = 1 - 1 / q
  return d > theta * n ? Math.floor(d / (d - theta * n)) : Infinity
}
