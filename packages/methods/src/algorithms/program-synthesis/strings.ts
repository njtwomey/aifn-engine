/**
 * Programming by example for string transformations: the version space of every program consistent with
 * input-output examples, as a graph, with counting and ranking by dynamic programming over it.
 *
 * After FlashFill (Gulwani, 2011, "Automating string processing in spreadsheets using input-output examples", POPL,
 * §§4 and 5) and the version-space algebra of Lau, Wolfman, Domingos and Weld (2003). A program is a concatenation of
 * atoms. An atom is a constant string, a token of the input (the $i$-th maximal run of letters and digits, counted from
 * the start for $i > 0$ or from the end for $i < 0$) with a case transformation, or the first character of a token.
 * For one example, every program that maps the input to the output is a path through a directed acyclic graph whose
 * nodes are the positions $0, \dots, n$ of the output: the edge $(i, j)$ carries every atom whose value on the input is
 * the piece of the output from $i$ to $j$. The programs consistent with several examples are the paths of the product
 * graph, whose nodes are tuples of positions and whose edges carry the atoms common to every example. The graph
 * represents exponentially many programs in polynomial size; counting them and ranking them are dynamic programmes
 * over it.
 *
 * Differences from FlashFill: positions are whole tokens and first characters instead of regular-expression positions,
 * there are no loops or conditionals, and the ranking is an additive cost per atom instead of FlashFill's learned
 * preferences. Counts are doubles (exact up to $2^{53}$).
 */
import { DomainError } from 'aifn-compute/foundation/errors'

/** A case transformation of a token: unchanged, upper case, lower case, or title case (first letter upper). */
export type CaseTransform = 'id' | 'upper' | 'lower' | 'title'

/** An atom of a program, told apart by `kind`. */
export type StringAtom =
  | {
      /** A constant string. */
      readonly kind: 'const'
      /** The string. */
      readonly value: string
    }
  | {
      /** A token of the input. */
      readonly kind: 'token'
      /** Which token: $1, 2, \dots$ from the start, $-1, -2, \dots$ from the end. */
      readonly index: number
      /** The case transformation applied to it. */
      readonly transform: CaseTransform
    }
  | {
      /** The first character of a token of the input. */
      readonly kind: 'initial'
      /** Which token, numbered as for `token`. */
      readonly index: number
      /** The case transformation applied to the character. */
      readonly transform: Exclude<CaseTransform, 'title'>
    }

/** An input-output example. */
export interface StringExample {
  /** The input string. */
  readonly input: string
  /** The output the program must produce from it. */
  readonly output: string
}

/** The version space of the examples so far: a directed acyclic graph over tuples of output positions. */
export interface StringVersionSpace {
  /** Node keys (tuples of positions joined by commas), in topological order; node 0 is the start. */
  readonly nodes: readonly string[]
  /** Index of the end node, or $-1$ when no program is consistent. */
  readonly end: number
  /** Edges: from and to (node indices) and the atoms on the edge (by key into `atoms`). */
  readonly edges: readonly { readonly from: number; readonly to: number; readonly atoms: readonly string[] }[]
  /** The atoms by key. */
  readonly atoms: ReadonlyMap<string, StringAtom>
}

/** A program with its ranking cost. */
export interface RankedProgram {
  /** The atoms, concatenated in order. */
  readonly atoms: readonly StringAtom[]
  /** The sum of the atoms' costs (`atomCost`). */
  readonly cost: number
}

/**
 * The tokens of a string: its maximal runs of letters and digits.
 *
 * @param s The string.
 * @returns The tokens, in order.
 *
 * @example Tokens of a name
 * print(stringTokens('grace brewster-hopper, 1906'))
 */
export function stringTokens(s: string): string[] {
  return s.match(/[\p{L}\p{N}]+/gu) ?? []
}

/**
 * A string with a case transformation applied.
 *
 * @param s The string.
 * @param t The transformation.
 * @returns The transformed string.
 */
const transform = (s: string, t: CaseTransform) =>
  t === 'upper'
    ? s.toUpperCase()
    : t === 'lower'
      ? s.toLowerCase()
      : t === 'title'
        ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase()
        : s

/**
 * The atom's value on an input, or null when it is undefined there (a token index out of range).
 *
 * @param atom The atom.
 * @param input The input string.
 * @returns The string the atom produces, or null.
 *
 * @example The last token, in title case
 * print(evalAtom({ kind: 'token', index: -1, transform: 'title' }, 'grace brewster hopper'))
 * print(evalAtom({ kind: 'token', index: 4, transform: 'id' }, 'grace brewster hopper'))
 */
export function evalAtom(atom: StringAtom, input: string): string | null {
  if (atom.kind === 'const') return atom.value
  const toks = stringTokens(input)
  const i = atom.index > 0 ? atom.index - 1 : toks.length + atom.index
  if (i < 0 || i >= toks.length) return null
  return atom.kind === 'token' ? transform(toks[i], atom.transform) : transform(toks[i].charAt(0), atom.transform)
}

/**
 * The program's output on an input: its atoms' values concatenated, or null when an atom is undefined there.
 *
 * @param program The atoms, in order.
 * @param input The input string.
 * @returns The output, or null.
 *
 * @example Surname, initial
 * const program = [
 *   { kind: 'token', index: -1, transform: 'title' },
 *   { kind: 'const', value: ', ' },
 *   { kind: 'initial', index: 1, transform: 'upper' },
 *   { kind: 'const', value: '.' },
 * ]
 * print(runStringProgram(program, 'ada lovelace'))
 */
export function runStringProgram(program: readonly StringAtom[], input: string): string | null {
  let out = ''
  for (const a of program) {
    const v = evalAtom(a, input)
    if (v === null) return null
    out += v
  }
  return out
}

/**
 * An atom as text, such as `title(token 2)`, `initial(token 1)` or `", "`.
 *
 * @param a The atom.
 * @returns The text.
 *
 * @example Three atoms
 * print(atomText({ kind: 'const', value: ', ' }), atomText({ kind: 'token', index: -1, transform: 'title' }))
 * print(atomText({ kind: 'initial', index: 1, transform: 'upper' }))
 */
export function atomText(a: StringAtom): string {
  if (a.kind === 'const') return JSON.stringify(a.value)
  const base = a.kind === 'token' ? `token ${a.index}` : `initial(token ${a.index})`
  return a.transform === 'id' ? base : `${a.transform}(${base})`
}

/**
 * An atom's key in a version space's table of atoms.
 *
 * @param a The atom.
 * @returns The key.
 */
const atomKey = (a: StringAtom) => (a.kind === 'const' ? `c:${a.value}` : `${a.kind}:${a.index}:${a.transform}`)

/**
 * Every non-constant atom with a value on this input: each token counted from both ends, with each transformation.
 *
 * @param input The input string.
 * @returns The atoms.
 */
function inputAtoms(input: string): StringAtom[] {
  const n = stringTokens(input).length
  const out: StringAtom[] = []
  for (let k = 1; k <= n; k++)
    for (const index of [k, k - n - 1]) {
      for (const t of ['id', 'upper', 'lower', 'title'] as const) out.push({ kind: 'token', index, transform: t })
      for (const t of ['id', 'upper', 'lower'] as const) out.push({ kind: 'initial', index, transform: t })
    }
  return out
}

/**
 * The version space of one example: the graph over the positions $0, \dots, n$ of the output.
 *
 * @param example The input and the output.
 * @returns The graph; every path from node 0 to the end is a program mapping the input to the output.
 *
 * @example How many programs map one name
 * const space = exampleVersionSpace({ input: 'jane doe', output: 'Doe, J.' })
 * print(space.nodes.length, 'nodes,', space.edges.length, 'edges')
 * print('programs by length:', countByLength(space))
 */
export function exampleVersionSpace(example: StringExample): StringVersionSpace {
  const { input, output } = example
  const n = output.length
  const atoms = new Map<string, StringAtom>()
  const values = inputAtoms(input).map((a) => ({ a, v: evalAtom(a, input)! }))
  const edges: { from: number; to: number; atoms: string[] }[] = []
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j <= n; j++) {
      const piece = output.slice(i, j)
      const here: string[] = []
      const c: StringAtom = { kind: 'const', value: piece }
      atoms.set(atomKey(c), c)
      here.push(atomKey(c))
      for (const { a, v } of values)
        if (v === piece) {
          atoms.set(atomKey(a), a)
          here.push(atomKey(a))
        }
      edges.push({ from: i, to: j, atoms: here })
    }
  return { nodes: Array.from({ length: n + 1 }, (_, i) => `${i}`), end: n, edges, atoms }
}

/**
 * Intersect a version space with one more example: the product graph, keeping the atoms that also produce the new
 * output's piece on the new input, and only the nodes on a path from the start to the end.
 *
 * @param space The version space of the earlier examples.
 * @param example The new example.
 * @returns The version space of all the examples (`end` is $-1$ when no program is consistent).
 *
 * @example A second example narrows the space
 * const one = exampleVersionSpace({ input: 'jane doe', output: 'Doe, J.' })
 * const two = intersectVersionSpace(one, { input: 'alan turing', output: 'Turing, A.' })
 * print(countByLength(one).reduce((s, x) => s + x, 0), '->', countByLength(two).reduce((s, x) => s + x, 0))
 */
export function intersectVersionSpace(space: StringVersionSpace, example: StringExample): StringVersionSpace {
  const { input, output } = example
  const out = new Map<string, number>()
  const nodes: string[] = []
  const nodeOf = (u: number, p: number) => {
    const key = `${space.nodes[u]},${p}`
    let id = out.get(key)
    if (id === undefined) {
      id = nodes.length
      out.set(key, id)
      nodes.push(key)
    }
    return { id, key }
  }
  const pos = new Map<number, number>() // new node id -> position in the new output
  const base = new Map<number, number>() // new node id -> old node
  const startNode = nodeOf(0, 0)
  pos.set(startNode.id, 0)
  base.set(startNode.id, 0)
  const byFrom = new Map<number, typeof space.edges>()
  for (const e of space.edges) byFrom.set(e.from, [...(byFrom.get(e.from) ?? []), e])
  const edgeAtoms = new Map<string, Set<string>>()
  const queue = [startNode.id]
  const seen = new Set(queue)
  while (queue.length) {
    const id = queue.shift()!
    const u = base.get(id)!
    const p = pos.get(id)!
    for (const e of byFrom.get(u) ?? [])
      for (const key of e.atoms) {
        const v = evalAtom(space.atoms.get(key)!, input)
        if (!v || !output.startsWith(v, p)) continue
        const to = nodeOf(e.to, p + v.length)
        pos.set(to.id, p + v.length)
        base.set(to.id, e.to)
        const ek = `${id}>${to.id}`
        if (!edgeAtoms.has(ek)) edgeAtoms.set(ek, new Set())
        edgeAtoms.get(ek)!.add(key)
        if (!seen.has(to.id)) {
          seen.add(to.id)
          queue.push(to.id)
        }
      }
  }
  const endKey = `${space.nodes[space.end]},${output.length}`
  const end = space.end >= 0 ? (out.get(endKey) ?? -1) : -1
  const edges = [...edgeAtoms].map(([k, s]) => {
    const [from, to] = k.split('>').map(Number)
    return { from, to, atoms: [...s] }
  })
  // Keep the nodes that reach the end.
  const alive = new Set<number>()
  if (end >= 0) {
    alive.add(end)
    let changed = true
    while (changed) {
      changed = false
      for (const e of edges)
        if (alive.has(e.to) && !alive.has(e.from)) {
          alive.add(e.from)
          changed = true
        }
    }
  }
  // Renumber the live nodes in topological order (by the sum of positions, which grows along every edge).
  const sum = (key: string) => key.split(',').reduce((s, x) => s + Number(x), 0)
  const live = [...alive].sort((a, b) => sum(nodes[a]) - sum(nodes[b]))
  if (!live.length) return { nodes: [nodes[0]], end: -1, edges: [], atoms: space.atoms }
  const renum = new Map(live.map((id, i) => [id, i]))
  return {
    nodes: live.map((id) => nodes[id]),
    end: renum.get(end)!,
    edges: edges
      .filter((e) => alive.has(e.from) && alive.has(e.to))
      .map((e) => ({ from: renum.get(e.from)!, to: renum.get(e.to)!, atoms: e.atoms })),
    atoms: space.atoms,
  }
}

/**
 * The version space of a list of examples: the first example's graph intersected with each of the others. An empty
 * list throws `DomainError`.
 *
 * @param examples At least one example.
 * @returns The graph of the programs consistent with all of them.
 *
 * @example Two examples leave eight programs
 * const space = versionSpace([
 *   { input: 'jane doe', output: 'Doe, J.' },
 *   { input: 'alan turing', output: 'Turing, A.' },
 * ])
 * print(countByLength(space))
 */
export function versionSpace(examples: readonly StringExample[]): StringVersionSpace {
  if (examples.length === 0) throw new DomainError('versionSpace', 'versionSpace: no example')
  let space = exampleVersionSpace(examples[0])
  for (const e of examples.slice(1)) space = intersectVersionSpace(space, e)
  return space
}

/**
 * How many consistent programs there are of each length (number of atoms), by dynamic programming over the graph.
 *
 * @param space A version space.
 * @returns `counts[k]`, the number of programs of $k$ atoms (index 0 is 0); empty when no program is consistent.
 *
 * @example Programs of one example by length
 * print(countByLength(exampleVersionSpace({ input: 'jane doe', output: 'Doe, J.' })))
 */
export function countByLength(space: StringVersionSpace): number[] {
  if (space.end < 0) return []
  const ways: number[][] = space.nodes.map(() => [])
  ways[0] = [1]
  const sorted = [...space.edges].sort((a, b) => a.from - b.from)
  for (const e of sorted) {
    const w = ways[e.from]
    for (let k = 0; k < w.length; k++) {
      if (!w[k]) continue
      ways[e.to][k + 1] = (ways[e.to][k + 1] ?? 0) + w[k] * e.atoms.length
    }
  }
  return Array.from(ways[space.end], (x) => x ?? 0)
}

/**
 * The ranking cost of an atom. `fewest-atoms`: every atom costs 1, a constant $10^{-3}$ more per character (to break
 * ties); `fewest-constants`: a constant costs one plus its length, any other atom 1.
 *
 * @param a The atom.
 * @param ranking The ranking.
 * @returns The cost.
 *
 * @example Constants cost more
 * const c = { kind: 'const', value: ', ' }
 * print(atomCost(c, 'fewest-atoms'), atomCost(c, 'fewest-constants'))
 */
export function atomCost(a: StringAtom, ranking: 'fewest-atoms' | 'fewest-constants'): number {
  if (ranking === 'fewest-atoms') return a.kind === 'const' ? 1 + 1e-3 * a.value.length : 1
  return a.kind === 'const' ? 1 + a.value.length : 1
}

/**
 * The $k$ cheapest consistent programs, by dynamic programming over the graph (a sorted list of the $k$ best partial
 * programs per node).
 *
 * @param space A version space.
 * @param k How many programs.
 * @param ranking The cost of an atom (see `atomCost`).
 * @returns Up to $k$ programs, cheapest first.
 *
 * @example The best program generalises to a new name
 * const space = versionSpace([
 *   { input: 'jane doe', output: 'Doe, J.' },
 *   { input: 'alan turing', output: 'Turing, A.' },
 * ])
 * const [best] = topPrograms(space, 1, 'fewest-atoms')
 * print(best.atoms.map(atomText).join(' + '))
 * print(runStringProgram(best.atoms, 'ada lovelace'))
 */
export function topPrograms(
  space: StringVersionSpace,
  k: number,
  ranking: 'fewest-atoms' | 'fewest-constants',
): RankedProgram[] {
  if (space.end < 0) return []
  const best: RankedProgram[][] = space.nodes.map(() => [])
  best[0] = [{ atoms: [], cost: 0 }]
  const sorted = [...space.edges].sort((a, b) => a.from - b.from)
  for (const e of sorted) {
    const into = best[e.to]
    for (const p of best[e.from])
      for (const key of e.atoms) {
        const a = space.atoms.get(key)!
        into.push({ atoms: [...p.atoms, a], cost: p.cost + atomCost(a, ranking) })
      }
    into.sort((x, y) => x.cost - y.cost)
    into.length = Math.min(into.length, k)
  }
  return best[space.end]
}
