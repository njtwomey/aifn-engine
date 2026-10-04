/** Test graphs shared by the graph tests: lettered graphs (CLRS figures) and seeded random graphs. */
import { fromEdges, type Graph } from 'aifn-compute/graph'
import { stream, uniform } from 'aifn-compute/foundation/random'

export const names = (s: string) => s.split('')
export const byName = (labels: string[]) => (x: string) => labels.indexOf(x)

/** A graph from edges written as two-letter strings, 'uv' for u → v, over the given node letters. */
export function lettered(nodes: string, edges: string[], directed = true): Graph {
  const at = byName(names(nodes))
  return fromEdges(
    nodes.length,
    edges.map((e) => [at(e[0]), at(e[1])] as const),
    { directed, labels: names(nodes) },
  )
}

/** A random graph on n nodes with each ordered pair present with probability p and weights in [0, 10). */
export function randomGraph(seed: number, n: number, p: number, directed = true): Graph {
  const s = stream(seed)
  const edges: [number, number, number][] = []
  for (let i = 0; i < n; i++)
    for (let j = directed ? 0 : i + 1; j < n; j++)
      if (i !== j && uniform(s) < p) edges.push([i, j, Math.floor(10 * uniform(s))])
  return fromEdges(n, edges, { directed })
}
