/**
 * FOIL (Quinlan 1990): learning a definition of a target relation from ground background facts and positive and
 * negative examples, by sequential covering.
 *
 * The outer loop learns one clause at a time: each clause starts as `target(A, B, …) :- true`, which covers every
 * example still uncovered; literals are added until it covers no negative example; the positives it covers are
 * removed, and the loop continues until none are left. The inner loop is greedy: of all literals that can be added,
 * the one with the highest FOIL gain is chosen. Clauses are evaluated on tuples (bindings of the clause's variables):
 * a literal that introduces a new variable extends each tuple by every value that makes the literal true. With p₀, n₀
 * the positive and negative tuples before the literal, p₁, n₁ after, and t the positive tuples before that have at
 * least one extension,
 *
 *     gain = t · (log₂ p₁/(p₁ + n₁) − log₂ p₀/(p₀ + n₀)).
 *
 * Background relations are extensional (sets of ground facts). The target may appear in a body (recursion); it is then
 * true of the positive examples, as in FOIL, and its arguments must be variables already bound and not the head's
 * own tuple. A finished clause is simplified by dropping literals it does not need. The inner loop is `refinementSearchSteps` of `aifn-compute/optim/search` with beam width 1: the refinements of a
 * clause are the clause with one more literal (`foilRefinements`) and the quality of each is its gain.
 */
import { DomainError } from 'aifn-compute/foundation/errors'
import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import { run } from 'aifn-compute/foundation/trace'
import { refinementSearchSteps, type SearchSpace, type SearchState, type SearchVisit } from 'aifn-compute/optim/search'
import {
  compound,
  indicator,
  isGround,
  parseProgram,
  termToString,
  variable,
  type Term,
} from 'aifn-compute/logic/terms'

// ── Problems ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A relation: its name, arity and tuples of constant ids. */
export interface FoilRelation {
  readonly name: string
  readonly arity: number
  readonly tuples: readonly (readonly number[])[]
}

/** A FOIL problem with constants interned: relation tuples and examples are arrays of constant ids. */
export interface FoilProblem {
  readonly target: { readonly name: string; readonly arity: number }
  readonly constants: readonly string[]
  readonly relations: readonly FoilRelation[]
  readonly positives: readonly (readonly number[])[]
  readonly negatives: readonly (readonly number[])[]
}

const constantName = (t: Term): string => {
  if (t.kind === 'atom') return t.name
  if (t.kind === 'number') return String(t.value)
  throw new DomainError('foilProblem', `foilProblem: ${termToString(t)} is not a constant`)
}

/** Ground atoms from text (facts separated by full stops) or terms. */
const groundAtoms = (x: string | readonly Term[], what: string): Term[] => {
  const terms = typeof x === 'string' ? parseProgram(x).clauses.map((c) => c.head) : [...x]
  for (const t of terms) if (!isGround(t)) throw new DomainError('foilProblem', `foilProblem: ${what} must be ground`)
  return terms
}

/**
 * A FOIL problem from ground background facts and examples (Prolog text or terms). Without `negatives`, the closed
 * world assumption gives every tuple of the examples' constants that is not positive. The target is the predicate of
 * the positives.
 */
export function foilProblem(
  background: string | readonly Term[],
  positives: string | readonly Term[],
  negatives?: string | readonly Term[],
): FoilProblem {
  const facts = groundAtoms(background, 'background facts')
  const pos = groundAtoms(positives, 'examples')
  if (pos.length === 0) throw new DomainError('foilProblem', 'foilProblem: needs at least one positive example')
  const key = indicator(pos[0])
  const [name, arityText] = [key.slice(0, key.lastIndexOf('/')), key.slice(key.lastIndexOf('/') + 1)]
  const arity = Number(arityText)
  const neg = negatives === undefined ? null : groundAtoms(negatives, 'examples')
  for (const e of [...pos, ...(neg ?? [])])
    if (indicator(e) !== key) throw new DomainError('foilProblem', `foilProblem: every example must be ${key}`)
  const ids = new Map<string, number>()
  const constants: string[] = []
  const intern = (t: Term) => {
    const c = constantName(t)
    if (!ids.has(c)) {
      ids.set(c, constants.length)
      constants.push(c)
    }
    return ids.get(c)!
  }
  const argsOf = (t: Term) => (t.kind === 'compound' ? t.args.map(intern) : [])
  const relations = new Map<string, { name: string; arity: number; tuples: number[][]; seen: Set<string> }>()
  for (const f of facts) {
    const k = indicator(f)
    if (k === key)
      throw new DomainError('foilProblem', `foilProblem: background facts may not define the target ${key}`)
    const r = relations.get(k) ?? {
      name: k.slice(0, k.lastIndexOf('/')),
      arity: Number(k.slice(k.lastIndexOf('/') + 1)),
      tuples: [],
      seen: new Set(),
    }
    relations.set(k, r)
    const tuple = argsOf(f)
    const s = tuple.join(',')
    if (!r.seen.has(s)) {
      r.seen.add(s)
      r.tuples.push(tuple)
    }
  }
  const posTuples = pos.map(argsOf)
  let negTuples: number[][]
  if (neg) negTuples = neg.map(argsOf)
  else {
    const isPos = new Set(posTuples.map((t) => t.join(',')))
    negTuples = []
    const n = constants.length
    const total = n ** arity
    if (total > 200_000) throw new DomainError('foilProblem', 'foilProblem: closed world too large; give negatives')
    for (let code = 0; code < total; code++) {
      const tuple: number[] = []
      for (let i = 0, c = code; i < arity; i++, c = Math.floor(c / n)) tuple.unshift(c % n)
      if (!isPos.has(tuple.join(','))) negTuples.push(tuple)
    }
  }
  return {
    target: { name, arity },
    constants,
    relations: [...relations.values()].map(({ name: rn, arity: ra, tuples }) => ({ name: rn, arity: ra, tuples })),
    positives: posTuples,
    negatives: negTuples,
  }
}

// ── Clauses being grown ──────────────────────────────────────────────────────────────────────────────────────────────

/** A body literal: a relation (index into `relations`, or −1 for the target itself) applied to variables by index. */
export interface FoilLiteral {
  readonly relation: number
  readonly args: readonly number[]
}

/**
 * A clause being grown with its tuples: each tuple is `[example, value of variable 0, 1, …]`. `p0`, `n0` and `t` are
 * the counts the gain of its last literal was computed from.
 */
export interface FoilNode {
  readonly body: readonly FoilLiteral[]
  /** Variables 0…arity−1 are the head's. */
  readonly variables: number
  readonly pos: readonly (readonly number[])[]
  readonly neg: readonly (readonly number[])[]
  readonly p0: number
  readonly n0: number
  readonly t: number
  readonly gain: number
}

/** Options of `foilSteps`. */
export interface FoilOptions {
  /** Literals per clause at most (default 4). */
  maxBodyLength?: number
  /** Clauses at most (default 6). */
  maxClauses?: number
  /** New variables a literal may introduce (default: the relation's arity − 1). */
  maxNewVariables?: number
  /** Allow the target in bodies (default true). */
  recursion?: boolean
}

const log2 = Math.log2

/** FOIL gain t · (log₂ p₁/(p₁+n₁) − log₂ p₀/(p₀+n₀)); −∞ when no positive tuple survives. */
export function foilGain(p0: number, n0: number, p1: number, n1: number, t: number): number {
  if (p1 === 0 || p0 === 0) return -Infinity
  return t * (log2(p1 / (p1 + n1)) - log2(p0 / (p0 + n0)))
}

/** The relation a literal names, its name and arity. */
const relationOf = (problem: FoilProblem, r: number) =>
  r < 0 ? { name: problem.target.name, arity: problem.target.arity } : problem.relations[r]

/** Variable names: the head's A, B, …; then C, D, … for new ones. */
export const foilVariableName = (i: number): string => (i < 26 ? String.fromCharCode(65 + i) : `V${i}`)

/** A literal (or the head with `relation` −1 and args 0…arity−1) as a term. */
export function foilLiteralTerm(problem: FoilProblem, literal: FoilLiteral): Term {
  const r = relationOf(problem, literal.relation)
  return compound(
    r.name,
    literal.args.map((v) => variable(foilVariableName(v), v)),
  )
}

/** The clause as `{ head, body }` terms. */
export function foilClause(problem: FoilProblem, body: readonly FoilLiteral[]): { head: Term; body: Term[] } {
  const head = foilLiteralTerm(problem, {
    relation: -1,
    args: Array.from({ length: problem.target.arity }, (_, i) => i),
  })
  return { head, body: body.map((l) => foilLiteralTerm(problem, l)) }
}

/** The clause as Prolog text. */
export function foilClauseText(problem: FoilProblem, body: readonly FoilLiteral[]): string {
  const { head, body: goals } = foilClause(problem, body)
  return goals.length === 0
    ? `${termToString(head)}.`
    : `${termToString(head)} :- ${goals.map((g) => termToString(g)).join(', ')}.`
}

/** The root node of a clause search: no body, one tuple per example. */
function rootNode(positives: readonly number[], problem: FoilProblem): FoilNode {
  const pos = positives.map((e) => [e, ...problem.positives[e]])
  const neg = problem.negatives.map((x, e) => [e, ...x])
  return { body: [], variables: problem.target.arity, pos, neg, p0: pos.length, n0: neg.length, t: pos.length, gain: 0 }
}

/** Index of each relation's tuples: a set of keys, for membership tests on bound arguments. */
type Indexed = { arity: number; tuples: readonly (readonly number[])[]; keys: Set<string> }

/**
 * The candidate literals for a clause, each with the clause it makes (tuples extended, counts and gain): the
 * refinement operator of FOIL's search. Argument tuples use at least one variable of the clause; new variables are
 * numbered in order of first use, so each literal is generated once.
 */
export function foilRefinements(problem: FoilProblem, node: FoilNode, options: FoilOptions = {}): FoilNode[] {
  const recursion = options.recursion ?? true
  const indexed: Indexed[] = problem.relations.map((r) => ({
    arity: r.arity,
    tuples: r.tuples,
    keys: new Set(r.tuples.map((t) => t.join(','))),
  }))
  const targetIndex: Indexed = {
    arity: problem.target.arity,
    tuples: problem.positives,
    keys: new Set(problem.positives.map((t) => t.join(','))),
  }
  const out: FoilNode[] = []
  const used = new Set(node.body.map((l) => `${l.relation}:${l.args.join(',')}`))
  const relations = [...problem.relations.map((_, i) => i), ...(recursion ? [-1] : [])]
  for (const r of relations) {
    const rel = r < 0 ? targetIndex : indexed[r]
    const arity = rel.arity
    const maxNew = r < 0 ? 0 : Math.min(options.maxNewVariables ?? arity - 1, arity - 1)
    // Argument tuples over old variables 0…v−1 and new ones v, v+1, … (first use in order).
    const args: number[] = []
    const emit = (newCount: number) => {
      if (args.length === arity) {
        if (newCount === arity) return
        if (r < 0 && args.every((a, i) => a === i)) return
        const literal: FoilLiteral = { relation: r, args: [...args] }
        if (used.has(`${r}:${args.join(',')}`)) return
        out.push(extend(node, literal, rel))
        return
      }
      for (let v = 0; v < node.variables; v++) {
        args.push(v)
        emit(newCount)
        args.pop()
      }
      if (newCount < maxNew) {
        args.push(node.variables + newCount)
        emit(newCount + 1)
        args.pop()
      }
    }
    emit(0)
  }
  return out
}

/** The clause with `literal` added: tuples extended by every binding of its new variables that makes it true. */
function extend(node: FoilNode, literal: FoilLiteral, rel: Indexed): FoilNode {
  const v = node.variables
  const fresh = literal.args.filter((a) => a >= v)
  const variables = v + (fresh.length ? Math.max(...fresh) - v + 1 : 0)
  const grow = (tuples: readonly (readonly number[])[]) => {
    const out: number[][] = []
    let extended = 0
    for (const tuple of tuples) {
      let any = false
      if (fresh.length === 0) {
        if (rel.keys.has(literal.args.map((a) => tuple[a + 1]).join(','))) {
          out.push([...tuple])
          any = true
        }
      } else
        for (const fact of rel.tuples) {
          const binding = new Map<number, number>()
          let ok = true
          for (let i = 0; i < literal.args.length && ok; i++) {
            const a = literal.args[i]
            if (a < v) ok = tuple[a + 1] === fact[i]
            else if (binding.has(a)) ok = binding.get(a) === fact[i]
            else binding.set(a, fact[i])
          }
          if (!ok) continue
          const next = [...tuple]
          for (let a = v; a < variables; a++) next.push(binding.get(a)!)
          out.push(next)
          any = true
        }
      if (any) extended++
    }
    return { out, extended }
  }
  const pos = grow(node.pos)
  const neg = grow(node.neg)
  const p0 = node.pos.length
  const n0 = node.neg.length
  return {
    body: [...node.body, literal],
    variables,
    pos: pos.out,
    neg: neg.out,
    p0,
    n0,
    t: pos.extended,
    gain: foilGain(p0, n0, pos.out.length, neg.out.length, pos.extended),
  }
}

/** The distinct examples (by index) a node's tuples come from. */
export const coveredExamples = (tuples: readonly (readonly number[])[]): number[] =>
  [...new Set(tuples.map((t) => t[0]))].sort((a, b) => a - b)

/**
 * The examples a clause body covers, by backtracking over bindings (not tuples): an example is covered when some
 * binding of the body's other variables makes every literal true. The target in a body is true of the positives.
 */
export function foilCoverage(
  problem: FoilProblem,
  body: readonly FoilLiteral[],
): { positives: number[]; negatives: number[] } {
  const keys = problem.relations.map((r) => new Set(r.tuples.map((t) => t.join(','))))
  const targetKeys = new Set(problem.positives.map((t) => t.join(',')))
  const covers = (example: readonly number[]) => {
    const binding = new Map<number, number>(example.map((c, i) => [i, c]))
    const go = (i: number): boolean => {
      if (i === body.length) return true
      const l = body[i]
      const tuples = l.relation < 0 ? problem.positives : problem.relations[l.relation].tuples
      if (l.args.every((a) => binding.has(a)))
        return (
          (l.relation < 0 ? targetKeys : keys[l.relation]).has(l.args.map((a) => binding.get(a)).join(',')) && go(i + 1)
        )
      for (const fact of tuples) {
        const added: number[] = []
        let ok = true
        for (let k = 0; k < l.args.length && ok; k++) {
          const a = l.args[k]
          const v = binding.get(a)
          if (v === undefined) {
            binding.set(a, fact[k])
            added.push(a)
          } else ok = v === fact[k]
        }
        if (ok && go(i + 1)) {
          added.forEach((a) => binding.delete(a))
          return true
        }
        added.forEach((a) => binding.delete(a))
      }
      return false
    }
    return go(0)
  }
  const which = (examples: readonly (readonly number[])[]) => examples.flatMap((e, i) => (covers(e) ? [i] : []))
  return { positives: which(problem.positives), negatives: which(problem.negatives) }
}

/** The body with variables after the head's renumbered in order of first use (after literals were removed). */
function renumber(arity: number, body: readonly FoilLiteral[]): FoilLiteral[] {
  const map = new Map<number, number>()
  return body.map((l) => ({
    relation: l.relation,
    args: l.args.map((a) => {
      if (a < arity) return a
      if (!map.has(a)) map.set(a, arity + map.size)
      return map.get(a)!
    }),
  }))
}

/**
 * FOIL's clause simplification: drop each literal (first to last) whose removal keeps the clause covering no negatives
 * and at least the same positives. Returns the simplified body and the literals removed.
 */
export function foilSimplify(
  problem: FoilProblem,
  body: readonly FoilLiteral[],
): { body: FoilLiteral[]; removed: FoilLiteral[] } {
  let kept = [...body]
  const removed: FoilLiteral[] = []
  const base = foilCoverage(problem, kept).positives.length
  for (let i = 0; i < kept.length && kept.length > 1;) {
    const without = kept.filter((_, j) => j !== i)
    const c = foilCoverage(problem, without)
    if (c.negatives.length === 0 && c.positives.length >= base) {
      removed.push(kept[i])
      kept = without
    } else i++
  }
  return { body: renumber(problem.target.arity, kept), removed }
}

// ── The algorithm ────────────────────────────────────────────────────────────────────────────────────────────────────

/** A clause FOIL has learned, with the examples it covers. */
export interface FoilLearned {
  readonly body: readonly FoilLiteral[]
  readonly text: string
  /** Positive examples it covers that were still uncovered when it was learned. */
  readonly newlyCovered: readonly number[]
  /** The clause as grown, before simplification, and the literals simplification removed (as text). */
  readonly grown: string
  readonly removed: readonly string[]
}

/** A candidate literal of one step, with its counts and gain. */
export interface FoilCandidate {
  readonly literal: FoilLiteral
  readonly text: string
  readonly p: number
  readonly n: number
  readonly t: number
  readonly gain: number
  readonly positives: readonly number[]
  readonly negatives: readonly number[]
}

/** What a step of `foilSteps` did. */
export type FoilEvent =
  | { readonly kind: 'start' }
  | { readonly kind: 'literal'; readonly chosen: FoilCandidate }
  | { readonly kind: 'clause'; readonly clause: FoilLearned }
  | { readonly kind: 'stuck'; readonly reason: string }
  | { readonly kind: 'done' }

/** The state of `foilSteps`. */
export interface FoilState extends Status {
  /** Clauses learned so far. */
  readonly clauses: readonly FoilLearned[]
  /** Positive examples (indices) not yet covered. */
  readonly uncovered: readonly number[]
  /** The clause being grown, and its search's state. */
  readonly current: FoilNode
  readonly currentText: string
  readonly search: SearchState<FoilNode>
  /** The candidates scored at this step, best first, and the positives and negatives the clause covers now. */
  readonly candidates: readonly FoilCandidate[]
  readonly coveredPositives: readonly number[]
  readonly coveredNegatives: readonly number[]
  readonly event: FoilEvent
  readonly terminated: boolean
}

/**
 * FOIL as a step-through algorithm: each step adds the best literal to the clause being grown (showing every
 * candidate's gain), commits a clause that covers no negatives (removing the positives it covers), or stops when the
 * positives are covered, no literal has positive gain, or a limit is reached.
 */
export function foilSteps(problem: FoilProblem, options: FoilOptions = {}): Algorithm<void, FoilState> {
  const maxBody = options.maxBodyLength ?? 4
  const maxClauses = options.maxClauses ?? 6
  if (!(maxBody >= 1)) throw new DomainError('foilSteps', 'foilSteps: maxBodyLength must be at least 1')
  const space = (positives: readonly number[]): SearchSpace<FoilNode> => ({
    root: rootNode(positives, problem),
    refine: (n) => foilRefinements(problem, n, options),
    quality: (n) => n.gain,
    key: (n) => n.body.map((l) => `${l.relation}:${l.args.join(',')}`).join(';'),
  })
  const searchOf = (positives: readonly number[]) =>
    // One level deeper than the body limit, so a refinement at the limit is still queued (not a leaf) and can be chosen.
    refinementSearchSteps(space(positives), { strategy: 'beam', beamWidth: 1, maxDepth: maxBody + 1, k: 1 })
  const start = (positives: readonly number[]) => {
    const search = searchOf(positives).init(undefined, undefined as never)
    return { search, current: search.frontier[0]?.node ?? space(positives).root }
  }
  const text = (n: FoilNode) => foilClauseText(problem, n.body)
  const candidate = (v: SearchVisit<FoilNode>): FoilCandidate => ({
    literal: v.node.body[v.node.body.length - 1],
    text: termToString(foilLiteralTerm(problem, v.node.body[v.node.body.length - 1])),
    p: v.node.pos.length,
    n: v.node.neg.length,
    t: v.node.t,
    gain: v.node.gain,
    positives: coveredExamples(v.node.pos),
    negatives: coveredExamples(v.node.neg),
  })
  const view = (n: FoilNode) => ({
    currentText: text(n),
    coveredPositives: coveredExamples(n.pos),
    coveredNegatives: coveredExamples(n.neg),
  })

  return {
    name: 'foil',
    init: () => {
      const uncovered = problem.positives.map((_, i) => i)
      const { search, current } = start(uncovered)
      return {
        t: 0,
        clauses: [],
        uncovered,
        current,
        search,
        candidates: [],
        ...view(current),
        event: { kind: 'start' },
        terminated: false,
      }
    },
    step: (s, ctx) => {
      if (s.terminated) return { ...s, t: s.t + 1 }
      const stop = (event: FoilEvent): FoilState => ({ ...s, t: s.t + 1, candidates: [], event, terminated: true })
      // A clause that covers no negatives is learned; its positives are removed.
      if (s.current.neg.length === 0 && s.current.body.length > 0) {
        const simple = foilSimplify(problem, s.current.body)
        const covered = new Set(foilCoverage(problem, simple.body).positives)
        const clause: FoilLearned = {
          body: simple.body,
          text: foilClauseText(problem, simple.body),
          newlyCovered: s.uncovered.filter((e) => covered.has(e)),
          grown: text(s.current),
          removed: simple.removed.map((l) => termToString(foilLiteralTerm(problem, l))),
        }
        const clauses = [...s.clauses, clause]
        const uncovered = s.uncovered.filter((e) => !covered.has(e))
        if (uncovered.length === 0 || clauses.length >= maxClauses)
          return {
            ...s,
            t: s.t + 1,
            clauses,
            uncovered,
            candidates: [],
            event: { kind: 'clause', clause },
            terminated: true,
          }
        const next = start(uncovered)
        return {
          ...s,
          t: s.t + 1,
          clauses,
          uncovered,
          current: next.current,
          search: next.search,
          candidates: [],
          ...view(next.current),
          event: { kind: 'clause', clause },
        }
      }
      if (s.current.body.length >= maxBody)
        return stop({ kind: 'stuck', reason: `the clause reached ${maxBody} literals and still covers negatives` })
      const search = searchOf(s.uncovered).step(s.search, ctx)
      const candidates = search.generated
        .filter((v) => v.fate !== 'duplicate')
        .map(candidate)
        .sort((a, b) => b.gain - a.gain)
      const best = search.frontier[0]
      if (!best || !(best.node.gain > 0))
        return {
          ...stop({ kind: 'stuck', reason: 'no literal has positive gain; the remaining positives stay uncovered' }),
          candidates,
        }
      return {
        ...s,
        t: s.t + 1,
        current: best.node,
        search,
        candidates,
        ...view(best.node),
        event: { kind: 'literal', chosen: candidate(best) },
      }
    },
  }
}

/** The outcome of `foil`: the learned clauses as Prolog text, and the positives left uncovered. */
export interface FoilResult {
  readonly clauses: readonly string[]
  readonly program: string
  readonly uncovered: readonly number[]
  readonly steps: number
  readonly stopped: string
}

/** Run FOIL to the end (see `foilSteps`). */
export function foil(problem: FoilProblem, options: FoilOptions = {}): FoilResult {
  const limit = ((options.maxBodyLength ?? 4) + 1) * (options.maxClauses ?? 6) + 2
  const s = run(foilSteps(problem, options), undefined, limit)
  const clauses = s.clauses.map((c) => c.text)
  return {
    clauses,
    program: clauses.join('\n'),
    uncovered: s.uncovered,
    steps: s.t,
    stopped:
      s.event.kind === 'stuck' ? s.event.reason : s.uncovered.length ? 'clause limit reached' : 'all positives covered',
  }
}
