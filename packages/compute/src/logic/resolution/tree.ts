/**
 * The SLD tree of a traced run of `sldSteps`: one node per resolvent, with how it was reached (a clause, a built-in, an
 * alternative resumed on backtracking), the unifier, the goals left, the query's bindings there, when it was created
 * and closed, and how it ended. Pruned nodes are the branches a cut removed before they were tried.
 */
import type { Clause, Substitution, Term } from 'aifn-compute/logic/terms'
import type { SldState } from './sld'

/** How a node ended: still `open` at the last step, a `success` (empty resolvent), a `failure`, `pruned` by a cut,
 * cut off at the depth `limit`, or an `error`. Inner nodes that were expanded are `expanded`. */
export type SldNodeStatus = 'open' | 'expanded' | 'success' | 'failure' | 'pruned' | 'limit' | 'error'

/** A node of the SLD tree. */
export interface SldNode {
  readonly id: number
  readonly parent: number | null
  readonly depth: number
  /** The goal resolved at the parent to reach this node (null for the root and resumed alternatives). */
  readonly goal: Term | null
  /** The clause used, by index, or null for built-ins, alternatives and the root. */
  readonly clause: number | null
  /** The clause head renamed apart, for clause steps. */
  readonly head: Term | null
  readonly unifier: Substitution
  /** A short description: `clause 2`, a built-in's note, an alternative's note, or why it was pruned. */
  readonly note: string
  /** The resolvent at this node (empty for a success; empty for a pruned node, which was never computed). */
  readonly goals: readonly Term[]
  /** The query's variables (by id) at this node. */
  readonly answer: readonly Term[]
  readonly status: SldNodeStatus
  /** The step that created the node, and the step that closed it (success, failure, error), if any. */
  readonly created: number
  readonly closed: number | null
  /** For a success, its index among the solutions; for a failure or error, the reason. */
  readonly solution: number | null
  readonly reason: string | null
}

/**
 * The SLD tree of a run from its states, in step order (e.g. a trace's `steps` with every state kept). Statuses are as
 * of the last state given; read a node's `created` and `closed` against a step t to draw the tree at step t.
 */
export function sldTree(states: readonly SldState[], clauses?: readonly Clause[]): SldNode[] {
  const nodes: (SldNode & { status: SldNodeStatus; closed: number | null })[] = []
  type Mutable = { -readonly [K in keyof SldNode]: SldNode[K] }
  const add = (n: Mutable) => (nodes[n.id] = n)
  const terms = (s: SldState) => (s.goals ?? []).map((g) => g.term)
  const clauseNote = (i: number) =>
    clauses && clauses[i]?.line ? `clause ${i + 1} (line ${clauses[i].line})` : `clause ${i + 1}`
  for (const s of states) {
    const e = s.event
    const parentDepth = (p: number) => (nodes[p]?.depth ?? -1) + 1
    const markExpanded = (p: number) => {
      if (nodes[p] && nodes[p].status === 'open') (nodes[p] as Mutable).status = 'expanded'
    }
    const pruned = 'pruned' in e ? e.pruned : []
    switch (e.kind) {
      case 'start':
        add({
          id: 0,
          parent: null,
          depth: 0,
          goal: null,
          clause: null,
          head: null,
          unifier: [],
          note: 'the query',
          goals: terms(s),
          answer: s.answer,
          status: 'open',
          created: s.t,
          closed: null,
          solution: null,
          reason: null,
        })
        break
      case 'resolve':
        markExpanded(e.parent)
        add({
          id: e.node,
          parent: e.parent,
          depth: parentDepth(e.parent),
          goal: e.goal,
          clause: e.clause,
          head: e.head,
          unifier: e.unifier,
          note: clauseNote(e.clause) + (e.retry ? ' (after backtracking)' : ''),
          goals: terms(s),
          answer: s.answer,
          status: 'open',
          created: s.t,
          closed: null,
          solution: null,
          reason: null,
        })
        break
      case 'builtin':
        markExpanded(e.parent)
        add({
          id: e.node,
          parent: e.parent,
          depth: parentDepth(e.parent),
          goal: e.goal,
          clause: null,
          head: null,
          unifier: e.unifier,
          note: e.note,
          goals: terms(s),
          answer: s.answer,
          status: 'open',
          created: s.t,
          closed: null,
          solution: null,
          reason: null,
        })
        break
      case 'alternative':
        markExpanded(e.parent)
        add({
          id: e.node,
          parent: e.parent,
          depth: parentDepth(e.parent),
          goal: null,
          clause: null,
          head: null,
          unifier: [],
          note: e.note,
          goals: terms(s),
          answer: s.answer,
          status: s.goals === null ? 'failure' : 'open',
          created: s.t,
          closed: s.goals === null ? s.t : null,
          solution: null,
          reason: s.goals === null ? e.note : null,
        })
        break
      case 'success': {
        const n = nodes[e.node] as Mutable
        n.status = 'success'
        n.closed = s.t
        n.solution = e.solution
        break
      }
      case 'fail': {
        const n = nodes[e.node] as Mutable
        if (n.status === 'open') {
          n.status = s.depthLimited && /depth limit/.test(e.reason) ? 'limit' : 'failure'
          n.closed = s.t
          n.reason = e.reason
        }
        break
      }
      case 'error': {
        const n = nodes[e.node] as Mutable
        n.status = 'error'
        n.closed = s.t
        n.reason = e.message
        break
      }
      case 'exhausted':
        break
    }
    for (const p of pruned)
      add({
        id: p.node,
        parent: p.parent,
        depth: parentDepth(p.parent),
        goal: null,
        clause: p.clause,
        head: null,
        unifier: [],
        note: p.clause === null ? `pruned: ${p.note}` : `${clauseNote(p.clause)}, pruned by cut`,
        goals: [],
        answer: [],
        status: 'pruned',
        created: s.t,
        closed: s.t,
        solution: null,
        reason: 'removed by a cut before it was tried',
      })
  }
  return nodes
}
