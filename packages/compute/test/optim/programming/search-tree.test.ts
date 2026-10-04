/** The branch-and-bound search tree is a binary `Tree` (aifn-compute/graph) with bounds and pruning reasons. */
import { describe, expect, it } from 'vitest'
import { branchAndBound, branchAndBoundTree, milp } from 'aifn-compute/optim/programming'
import { run } from 'aifn-compute/foundation/trace'

describe('branch-and-bound search tree', () => {
  it('is a binary Tree with bounds and pruning reasons', () => {
    const problem = {
      c: [-5, -8],
      A_ub: [
        [1, 1],
        [5, 9],
      ],
      b_ub: [6, 45],
    }
    const r = milp(problem, { strategy: 'depth-first' })
    const t = r.searchTree
    expect(t.nodes).toHaveLength(r.tree.length)
    expect(t.arity).toBe(2)
    for (const n of t.nodes) {
      expect(n.status).toBe(r.tree[n.id].status)
      if (n.parent !== null) expect(n.slot).toBe(r.tree[n.id].branch!.direction === 'down' ? 0 : 1)
    }
    expect(t.nodes.some((n) => n.prunedBy !== null)).toBe(true)
    expect(t.edges[1]!.label).toMatch(/^\$x_\{\d\} \\(le|ge) \d+\$$/)
    // A partial state's tree.
    const s = run(branchAndBound(problem), {}, 2)
    expect(branchAndBoundTree(s.nodes).nodes).toHaveLength(s.nodes.length)
  })
})
