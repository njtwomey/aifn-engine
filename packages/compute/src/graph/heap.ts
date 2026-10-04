/**
 * Two small data structures used by the graph algorithms and exported for other modules: a binary min-heap (priority
 * queue) and a disjoint-set forest (union–find). Both are plain data that the functions below mutate; an algorithm
 * whose states must stay immutable copies them first (`heapCopy`, `unionFindCopy`).
 */

// ---------------------------------------------------------------------------------------------------------------------
// Binary heap (Williams, 1964, "Algorithm 232: Heapsort", CACM 7(6)).

/** A heap entry: a value, its priority, and its insertion number (ties in priority pop in insertion order). */
export interface HeapEntry<T> {
  value: T
  priority: number
  seq: number
}

/** A binary min-heap: `entries` in heap order (entry k's children are 2k + 1 and 2k + 2), `pushed` a counter. */
export interface Heap<T> {
  entries: HeapEntry<T>[]
  pushed: number
}

/** An empty min-heap. */
export function createHeap<T>(): Heap<T> {
  return { entries: [], pushed: 0 }
}

const before = <T>(a: HeapEntry<T>, b: HeapEntry<T>) =>
  a.priority < b.priority || (a.priority === b.priority && a.seq < b.seq)

/** Add `value` with `priority` (O(log n)); mutates the heap. */
export function heapPush<T>(h: Heap<T>, value: T, priority: number): void {
  if (Number.isNaN(priority)) throw new DomainError('heapPush', 'heapPush: priority is NaN')
  const e = h.entries
  e.push({ value, priority, seq: h.pushed++ })
  let k = e.length - 1
  while (k > 0) {
    const parent = (k - 1) >> 1
    if (!before(e[k], e[parent])) break
    ;[e[k], e[parent]] = [e[parent], e[k]]
    k = parent
  }
}

/** Remove and return the entry of least priority (earliest pushed among ties), or undefined when empty; mutates. */
export function heapPop<T>(h: Heap<T>): HeapEntry<T> | undefined {
  const e = h.entries
  if (e.length === 0) return undefined
  const top = e[0]
  const last = e.pop()!
  if (e.length > 0) {
    e[0] = last
    let k = 0
    for (;;) {
      const l = 2 * k + 1
      const r = l + 1
      let m = k
      if (l < e.length && before(e[l], e[m])) m = l
      if (r < e.length && before(e[r], e[m])) m = r
      if (m === k) break
      ;[e[k], e[m]] = [e[m], e[k]]
      k = m
    }
  }
  return top
}

/** The entry of least priority without removing it, or undefined when empty. */
export function heapPeek<T>(h: Heap<T>): HeapEntry<T> | undefined {
  return h.entries[0]
}

/** A copy of the heap that can be mutated without touching the original (entries are shared, being immutable). */
export function heapCopy<T>(h: Heap<T>): Heap<T> {
  return { entries: [...h.entries], pushed: h.pushed }
}

/** The entries in pop order (a sorted copy), e.g. to show a priority queue. */
export function heapSorted<T>(h: Heap<T>): HeapEntry<T>[] {
  return [...h.entries].sort((a, b) => (before(a, b) ? -1 : 1))
}

// ---------------------------------------------------------------------------------------------------------------------
// Union–find with union by rank and path halving (Tarjan, 1975, "Efficiency of a good but not linear set union
// algorithm", JACM 22(2); Tarjan and van Leeuwen, 1984).

/** A disjoint-set forest over 0 … n − 1: `parent[v]` (a root is its own parent), `rank[v]`, and the number of sets. */
export interface UnionFind {
  parent: number[]
  rank: number[]
  count: number
}

/** n singleton sets. */
export function unionFind(n: number): UnionFind {
  return { parent: Array.from({ length: n }, (_, v) => v), rank: new Array<number>(n).fill(0), count: n }
}

/** The representative (root) of v's set; halves the path as it goes, so it mutates `parent`. */
export function unionFindRoot(uf: UnionFind, v: number): number {
  const p = uf.parent
  while (p[v] !== v) {
    p[v] = p[p[v]]
    v = p[v]
  }
  return v
}

/** Merge the sets of a and b (union by rank); false when they were already one set. Mutates. */
export function unite(uf: UnionFind, a: number, b: number): boolean {
  let ra = unionFindRoot(uf, a)
  let rb = unionFindRoot(uf, b)
  if (ra === rb) return false
  if (uf.rank[ra] < uf.rank[rb]) [ra, rb] = [rb, ra]
  uf.parent[rb] = ra
  if (uf.rank[ra] === uf.rank[rb]) uf.rank[ra]++
  uf.count--
  return true
}

/** A copy that can be mutated without touching the original. */
export function unionFindCopy(uf: UnionFind): UnionFind {
  return { parent: [...uf.parent], rank: [...uf.rank], count: uf.count }
}
import { DomainError } from 'aifn-compute/foundation/errors'
