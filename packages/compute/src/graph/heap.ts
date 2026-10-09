/**
 * Two small data structures used by the graph algorithms and exported for other modules: a binary min-heap (priority
 * queue) and a disjoint-set forest (union–find). Both are plain data that the functions below mutate; an algorithm
 * whose states must stay immutable copies them first (`heapCopy`, `unionFindCopy`).
 */

// ---------------------------------------------------------------------------------------------------------------------
// Binary heap (Williams, 1964, "Algorithm 232: Heapsort", CACM 7(6)).

/** A heap entry: a value, its priority, and its insertion number (ties in priority pop in insertion order). */
export interface HeapEntry<T> {
  /** What was pushed. */
  value: T
  /** Its priority: the least pops first. */
  priority: number
  /** Its insertion number, from 0, which breaks ties. */
  seq: number
}

/** A binary min-heap: `entries` in heap order (entry $k$'s children are $2k + 1$ and $2k + 2$), `pushed` a counter. */
export interface Heap<T> {
  /** The entries in heap order: none comes before its parent. */
  entries: HeapEntry<T>[]
  /** The number of pushes so far, the next entry's `seq`. */
  pushed: number
}

/**
 * An empty min-heap.
 *
 * @returns A heap with no entries.
 *
 * @example Push, peek and pop
 * const h = createHeap()
 * heapPush(h, 'c', 3)
 * heapPush(h, 'a', 1)
 * heapPush(h, 'b', 2)
 * print('peek:', heapPeek(h).value)
 * print('pops:', heapPop(h).value, heapPop(h).value, heapPop(h).value, heapPop(h))
 */
export function createHeap<T>(): Heap<T> {
  return { entries: [], pushed: 0 }
}

/**
 * Whether entry `a` pops before entry `b`: a smaller priority, or an equal one pushed earlier.
 *
 * @param a An entry.
 * @param b Another entry.
 * @returns True when `a` comes first.
 */
const before = <T>(a: HeapEntry<T>, b: HeapEntry<T>) =>
  a.priority < b.priority || (a.priority === b.priority && a.seq < b.seq)

/**
 * Add `value` with `priority` ($O(\log n)$ for $n$ entries); mutates the heap. Throws `DomainError` for a NaN
 * priority.
 *
 * @param h The heap, modified in place.
 * @param value What to store.
 * @param priority The key it pops by: the least first, ties in push order.
 *
 * @example Ties pop in push order
 * const h = createHeap()
 * heapPush(h, 'first', 1)
 * heapPush(h, 'second', 1)
 * heapPush(h, 'urgent', 0)
 * print(heapSorted(h).map((e) => e.value))
 */
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

/**
 * Remove and return the entry of least priority (earliest pushed among ties), or undefined when empty; mutates.
 * $O(\log n)$ for $n$ entries.
 *
 * @param h The heap, modified in place.
 * @returns The entry removed, with its value, priority and insertion number.
 *
 * @example Pop in priority order
 * const h = createHeap()
 * for (const [v, p] of [['x', 5], ['y', 2], ['z', 9]]) heapPush(h, v, p)
 * print(heapPop(h))
 * print('left:', h.entries.length)
 */
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

/**
 * The entry of least priority without removing it, or undefined when empty.
 *
 * @param h The heap (not modified).
 * @returns The entry that `heapPop` would remove.
 *
 * @example Peeking leaves the heap as it is
 * const h = createHeap()
 * heapPush(h, 'b', 2)
 * heapPush(h, 'a', 1)
 * print('peek:', heapPeek(h).value, 'size:', h.entries.length)
 * print('empty:', heapPeek(createHeap()))
 */
export function heapPeek<T>(h: Heap<T>): HeapEntry<T> | undefined {
  return h.entries[0]
}

/**
 * A copy of the heap that can be mutated without touching the original (entries are shared, being immutable).
 *
 * @param h The heap to copy.
 * @returns A new heap with the same entries and counter.
 *
 * @example Popping the copy leaves the original whole
 * const h = createHeap()
 * heapPush(h, 'a', 1)
 * const c = heapCopy(h)
 * heapPop(c)
 * print('original:', h.entries.length, 'copy:', c.entries.length)
 */
export function heapCopy<T>(h: Heap<T>): Heap<T> {
  return { entries: [...h.entries], pushed: h.pushed }
}

/**
 * The entries in pop order (a sorted copy), e.g. to show a priority queue.
 *
 * @param h The heap (not modified).
 * @returns Its entries, least priority first and ties in push order.
 *
 * @example A queue in pop order
 * const h = createHeap()
 * for (const [v, p] of [['c', 3], ['a', 1], ['b', 2]]) heapPush(h, v, p)
 * print(heapSorted(h).map((e) => `${e.value}:${e.priority}`))
 */
export function heapSorted<T>(h: Heap<T>): HeapEntry<T>[] {
  return [...h.entries].sort((a, b) => (before(a, b) ? -1 : 1))
}

// ---------------------------------------------------------------------------------------------------------------------
// Union–find with union by rank and path halving (Tarjan, 1975, "Efficiency of a good but not linear set union
// algorithm", JACM 22(2); Tarjan and van Leeuwen, 1984).

/**
 * A disjoint-set forest over $0, \dots, n - 1$: `parent[v]` (a root is its own parent), `rank[v]`, and the number of
 * sets.
 */
export interface UnionFind {
  /** The parent of each element; a set's representative is its own parent. */
  parent: number[]
  /** An upper bound on the height of each root's tree, used to keep trees shallow. */
  rank: number[]
  /** The number of disjoint sets. */
  count: number
}

/**
 * $n$ singleton sets.
 *
 * @param n The number of elements, $0, \dots, n - 1$.
 * @returns The forest, each element its own set.
 *
 * @example Merging sets
 * const uf = unionFind(4)
 * unite(uf, 0, 1)
 * unite(uf, 2, 3)
 * print('sets:', uf.count)
 * print('0 and 1 together:', unionFindRoot(uf, 0) === unionFindRoot(uf, 1))
 * print('1 and 2 together:', unionFindRoot(uf, 1) === unionFindRoot(uf, 2))
 */
export function unionFind(n: number): UnionFind {
  return { parent: Array.from({ length: n }, (_, v) => v), rank: new Array<number>(n).fill(0), count: n }
}

/**
 * The representative (root) of $v$'s set; halves the path as it goes, so it mutates `parent`.
 *
 * @param uf The forest, whose `parent` is shortened in place.
 * @param v The element.
 * @returns The root of its set: two elements are in one set exactly when their roots are equal.
 *
 * @example Elements of one set share a root
 * const uf = unionFind(3)
 * unite(uf, 0, 1)
 * print('roots:', [0, 1, 2].map((v) => unionFindRoot(uf, v)))
 */
export function unionFindRoot(uf: UnionFind, v: number): number {
  const p = uf.parent
  while (p[v] !== v) {
    p[v] = p[p[v]]
    v = p[v]
  }
  return v
}

/**
 * Merge the sets of $a$ and $b$ (union by rank); false when they were already one set. Mutates.
 *
 * @param uf The forest, modified in place.
 * @param a An element.
 * @param b Another element.
 * @returns True when two sets were merged, false when `a` and `b` were already in one (a cycle, for Kruskal).
 *
 * @example A second union of the same pair does nothing
 * const uf = unionFind(3)
 * print('first:', unite(uf, 0, 1), 'again:', unite(uf, 1, 0), 'sets:', uf.count)
 */
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

/**
 * A copy that can be mutated without touching the original.
 *
 * @param uf The forest to copy.
 * @returns A new forest with the same sets.
 *
 * @example Uniting in the copy leaves the original as it was
 * const uf = unionFind(2)
 * const c = unionFindCopy(uf)
 * unite(c, 0, 1)
 * print('original:', uf.count, 'copy:', c.count)
 */
export function unionFindCopy(uf: UnionFind): UnionFind {
  return { parent: [...uf.parent], rank: [...uf.rank], count: uf.count }
}
import { DomainError } from 'aifn-compute/foundation/errors'
