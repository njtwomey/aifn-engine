/**
 * FOIL (Quinlan 1990): learning a definition of a target relation from ground background facts and positive and
 * negative examples, by sequential covering.
 *
 * The outer loop learns one clause at a time: each clause starts as `target(A, B, …) :- true`, which covers every
 * example still uncovered; literals are added until it covers no negative example; the positives it covers are
 * removed, and the loop continues until none are left. The inner loop is greedy: of all literals that can be added,
 * the one with the highest FOIL gain is chosen. Clauses are evaluated on tuples (bindings of the clause's variables):
 * a literal that introduces a new variable extends each tuple by every value that makes the literal true. With $p_0$,
 * $n_0$ the positive and negative tuples before the literal, $p_1$, $n_1$ after, and $t$ the positive tuples before
 * that have at least one extension, the gain is
 * $t \left(\log_2 \frac{p_1}{p_1 + n_1} - \log_2 \frac{p_0}{p_0 + n_0}\right)$.
 *
 * Background relations are extensional (sets of ground facts). The target may appear in a body (recursion); it is then
 * true of the positive examples, as in FOIL, and its arguments must be variables already bound and not the head's
 * own tuple. A finished clause is simplified by dropping literals it does not need. The inner loop is
 * `refinementSearchSteps` of `aifn-compute/optim/search` with beam width 1: the refinements of a clause are the clause
 * with one more literal (`foilRefinements`) and the quality of each is its gain.
 *
 * Constants are interned: a problem's relations and examples are tuples of constant ids, indices into its
 * `constants`, and a clause's variables are numbered, the head's $0, \dots, a - 1$ for a target of arity $a$ first.
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
  /** The predicate's name. */
  readonly name: string
  /** Its number of arguments. */
  readonly arity: number
  /** Its ground facts, each a tuple of `arity` constant ids, without repeats. */
  readonly tuples: readonly (readonly number[])[]
}

/** A FOIL problem with constants interned: relation tuples and examples are arrays of constant ids. */
export interface FoilProblem {
  /** The relation to learn: its name and arity. */
  readonly target: { readonly name: string; readonly arity: number }
  /** The constants' names, indexed by id. */
  readonly constants: readonly string[]
  /** The background relations, in order of first appearance. */
  readonly relations: readonly FoilRelation[]
  /** The positive examples, each a tuple of constant ids. */
  readonly positives: readonly (readonly number[])[]
  /** The negative examples, each a tuple of constant ids. */
  readonly negatives: readonly (readonly number[])[]
}

/**
 * The name a constant is interned under: an atom's name, or a number's text. Throws `DomainError` for a variable or
 * compound term.
 *
 * @param t An argument of a fact or example.
 * @returns Its name.
 */
const constantName = (t: Term): string => {
  if (t.kind === 'atom') return t.name
  if (t.kind === 'number') return String(t.value)
  throw new DomainError('foilProblem', `foilProblem: ${termToString(t)} is not a constant`)
}

/**
 * Ground atoms from text (facts separated by full stops) or terms. Throws `DomainError` when one has a variable, or
 * when the text holds a rule (a clause with a body).
 *
 * @param x Prolog text of facts, or the terms themselves.
 * @param what What they are, for the error message.
 * @returns The atoms.
 */
const groundAtoms = (x: string | readonly Term[], what: string): Term[] => {
  if (typeof x === 'string') {
    const { clauses } = parseProgram(x)
    for (const c of clauses)
      if (c.body.length > 0)
        throw new DomainError('foilProblem', `foilProblem: ${what} must be facts, not rules (line ${c.line})`)
    const terms = clauses.map((c) => c.head)
    for (const t of terms) if (!isGround(t)) throw new DomainError('foilProblem', `foilProblem: ${what} must be ground`)
    return terms
  }
  for (const t of x) if (!isGround(t)) throw new DomainError('foilProblem', `foilProblem: ${what} must be ground`)
  return [...x]
}

/**
 * A FOIL problem from ground background facts and examples (Prolog text or terms). Without `negatives`, the closed
 * world assumption gives every tuple of the problem's constants (those of the background facts and of the examples)
 * that is not positive. The target is the predicate of the positives. Throws `DomainError` when there is no positive,
 * an example is of another predicate, a background fact is of the target, an argument is not a constant, the text holds
 * a rule rather than a fact, or the closed world has more than 200 000 tuples.
 *
 * @param background The background facts: ground atoms, as Prolog text (`parent(ann, bob). ...`) or terms.
 * @param positives The positive examples of the target, ground atoms of one predicate.
 * @param negatives The negative examples, of the same predicate; when left out, the closed world gives them.
 * @returns The problem, its constants numbered in order of first appearance (background first).
 *
 * @example Grandparents, with closed-world negatives
 * const problem = foilProblem(
 *   'parent(ann, bob). parent(bob, cid). parent(cid, dan).',
 *   'grandparent(ann, cid). grandparent(bob, dan).',
 * )
 * print('constants:', problem.constants.join(', '))
 * print('positives:', problem.positives)
 * print('closed-world negatives:', problem.negatives.length)
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

/**
 * A body literal: a relation (index into `relations`, or $-1$ for the target itself) applied to variables by index.
 */
export interface FoilLiteral {
  /** The relation's index in `relations`, or $-1$ for the target. */
  readonly relation: number
  /** The variable of each argument, by index. */
  readonly args: readonly number[]
}

/**
 * A clause being grown with its tuples: each tuple is `[example, value of variable 0, 1, …]`. `p0`, `n0` and `t` are
 * the counts the gain of its last literal was computed from.
 */
export interface FoilNode {
  /** The body literals, in order. */
  readonly body: readonly FoilLiteral[]
  /** The number of variables; variables $0, \dots, a - 1$ ($a$ the target's arity) are the head's. */
  readonly variables: number
  /** The positive tuples the clause covers. */
  readonly pos: readonly (readonly number[])[]
  /** The negative tuples the clause covers. */
  readonly neg: readonly (readonly number[])[]
  /** The positive and negative tuples before the last literal. */
  readonly p0: number
  readonly n0: number
  /** The positive tuples before the last literal that it extends. */
  readonly t: number
  /** The FOIL gain of the last literal (0 for the empty body). */
  readonly gain: number
}

/** Options of `foilSteps`. */
export interface FoilOptions {
  /** Literals per clause at most (default 4). */
  maxBodyLength?: number
  /** Clauses at most (default 6). */
  maxClauses?: number
  /**
   * New variables a literal may introduce (default and largest: the relation's arity $-1$, so a literal uses at least
   * one variable of the clause).
   */
  maxNewVariables?: number
  /** Allow the target in bodies (default true). */
  recursion?: boolean
}

const log2 = Math.log2

/**
 * FOIL gain $t \left(\log_2 \frac{p_1}{p_1 + n_1} - \log_2 \frac{p_0}{p_0 + n_0}\right)$: the information gained
 * about the positive tuples, weighted by the positives kept. $-\infty$ when no positive tuple survives (or there was
 * none before).
 *
 * @param p0 The positive tuples $p_0$ before the literal.
 * @param n0 The negative tuples $n_0$ before the literal.
 * @param p1 The positive tuples $p_1$ after it.
 * @param n1 The negative tuples $n_1$ after it.
 * @param t The positive tuples $t$ before the literal that have at least one extension after it.
 * @returns The gain in bits.
 *
 * @example A literal that keeps the positives and drops every negative
 * print(foilGain(2, 14, 2, 0, 2))
 * print(foilGain(2, 14, 2, 4, 2))
 * print(foilGain(2, 14, 0, 3, 0))
 */
export function foilGain(p0: number, n0: number, p1: number, n1: number, t: number): number {
  if (p1 === 0 || p0 === 0) return -Infinity
  return t * (log2(p1 / (p1 + n1)) - log2(p0 / (p0 + n0)))
}

/**
 * The relation a literal names, its name and arity.
 *
 * @param problem The problem.
 * @param r The literal's `relation`: an index into `problem.relations`, or $-1$ for the target.
 * @returns The relation (the target's name and arity for $-1$).
 */
const relationOf = (problem: FoilProblem, r: number) =>
  r < 0 ? { name: problem.target.name, arity: problem.target.arity } : problem.relations[r]

/**
 * The name of variable `i`: A to Z, then `V26`, `V27`, …. The head's variables come first, so a target of arity 2 has
 * the head `A, B` and new variables `C, D, …`.
 *
 * @param i The variable's index, from 0.
 * @returns Its name.
 *
 * @example Variable names
 * print([0, 1, 2, 25, 26].map(foilVariableName).join(' '))
 */
export const foilVariableName = (i: number): string => (i < 26 ? String.fromCharCode(65 + i) : `V${i}`)

/**
 * A literal (or the head, with `relation` $-1$ and args $0, \dots, a - 1$) as a term, its variables named by
 * `foilVariableName`.
 *
 * @param problem The problem the literal belongs to.
 * @param literal The literal: a relation index ($-1$ for the target) and its argument variables.
 * @returns The term, such as `parent(A, C)`; variable $i$ has id $i$.
 *
 * @example A literal as a term
 * const problem = foilProblem('parent(ann, bob).', 'grandparent(ann, bob).')
 * const t = foilLiteralTerm(problem, { relation: 0, args: [0, 2] })
 * print(t.functor, t.args.map((v) => v.name).join(', '))
 */
export function foilLiteralTerm(problem: FoilProblem, literal: FoilLiteral): Term {
  const r = relationOf(problem, literal.relation)
  return compound(
    r.name,
    literal.args.map((v) => variable(foilVariableName(v), v)),
  )
}

/**
 * The clause as `{ head, body }` terms: the head is the target applied to the head's variables.
 *
 * @param problem The problem the clause is for.
 * @param body The body literals, in order (empty for the clause that covers everything).
 * @returns The head and body goals as terms.
 *
 * @example The grandparent clause as terms
 * const problem = foilProblem('parent(ann, bob). parent(bob, cid).', 'grandparent(ann, cid).')
 * const { head, body } = foilClause(problem, [{ relation: 0, args: [0, 2] }, { relation: 0, args: [2, 1] }])
 * print('head:', head.functor, head.args.map((v) => v.name).join(', '))
 * print('body:', body.map((g) => `${g.functor}(${g.args.map((v) => v.name).join(', ')})`).join(', '))
 */
export function foilClause(problem: FoilProblem, body: readonly FoilLiteral[]): { head: Term; body: Term[] } {
  const head = foilLiteralTerm(problem, {
    relation: -1,
    args: Array.from({ length: problem.target.arity }, (_, i) => i),
  })
  return { head, body: body.map((l) => foilLiteralTerm(problem, l)) }
}

/**
 * The clause as Prolog text: `head.` for an empty body, `head :- literal, literal.` otherwise.
 *
 * @param problem The problem the clause is for.
 * @param body The body literals, in order.
 * @returns The text.
 *
 * @example A clause and the empty clause
 * const problem = foilProblem('parent(ann, bob). parent(bob, cid).', 'grandparent(ann, cid).')
 * print(foilClauseText(problem, [{ relation: 0, args: [0, 2] }, { relation: 0, args: [2, 1] }]))
 * print(foilClauseText(problem, []))
 */
export function foilClauseText(problem: FoilProblem, body: readonly FoilLiteral[]): string {
  const { head, body: goals } = foilClause(problem, body)
  return goals.length === 0
    ? `${termToString(head)}.`
    : `${termToString(head)} :- ${goals.map((g) => termToString(g)).join(', ')}.`
}

/**
 * The root node of a clause search: no body, one tuple per example.
 *
 * @param positives The positive examples still uncovered, by index into `problem.positives`.
 * @param problem The problem; all its negatives are included.
 * @returns The node, each tuple the example's index followed by its constants.
 */
function rootNode(positives: readonly number[], problem: FoilProblem): FoilNode {
  const pos = positives.map((e) => [e, ...problem.positives[e]])
  const neg = problem.negatives.map((x, e) => [e, ...x])
  return { body: [], variables: problem.target.arity, pos, neg, p0: pos.length, n0: neg.length, t: pos.length, gain: 0 }
}

/**
 * Index of each relation's tuples: a set of keys, for membership tests on bound arguments. `arity` and `tuples` are
 * the relation's (the positives for the target), `keys` each tuple joined with commas.
 */
type Indexed = { arity: number; tuples: readonly (readonly number[])[]; keys: Set<string> }

/**
 * The candidate literals for a clause, each with the clause it makes (tuples extended, counts and gain): the
 * refinement operator of FOIL's search. Argument tuples use at least one variable of the clause; new variables are
 * numbered in order of first use, so each literal is generated once. A literal already in the body is skipped, and so
 * is the target applied to the head's own variables; the target takes no new variables.
 *
 * @param problem The problem.
 * @param node The clause to refine, with its tuples: a root, or an earlier refinement.
 * @param options Only `maxNewVariables` and `recursion` are read (see `FoilOptions`).
 * @returns The refined clauses, by relation (the target last) and then argument tuple.
 *
 * @example The first literals FOIL considers for grandparent
 * const problem = foilProblem(
 *   'parent(ann, bob). parent(bob, cid). parent(cid, dan).',
 *   'grandparent(ann, cid). grandparent(bob, dan).',
 * )
 * const root = foilSteps(problem).init().current
 * const refined = foilRefinements(problem, root)
 * print(refined.length, 'refinements')
 * for (const n of refined.slice(0, 4)) print(foilClauseText(problem, n.body), 'gain', n.gain)
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

/**
 * The clause with `literal` added: tuples extended by every binding of its new variables that makes it true.
 *
 * @param node The clause before the literal.
 * @param literal The literal to add; its new variables are numbered from `node.variables` on.
 * @param rel The literal's relation, with its tuples and their keys.
 * @returns The extended clause, with $p_0$, $n_0$, $t$ and the gain of the literal.
 */
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

/**
 * The distinct examples (by index) a node's tuples come from, in ascending order.
 *
 * @param tuples Tuples of a `FoilNode`, each starting with the index of its example.
 * @returns The example indices.
 *
 * @example Two tuples of example 2 and one of example 0
 * print(coveredExamples([[2, 5, 1], [0, 3, 3], [2, 6, 1]]))
 */
export const coveredExamples = (tuples: readonly (readonly number[])[]): number[] =>
  [...new Set(tuples.map((t) => t[0]))].sort((a, b) => a - b)

/**
 * The examples a clause body covers, by backtracking over bindings (not tuples): an example is covered when some
 * binding of the body's other variables makes every literal true. The target in a body is true of the positives.
 *
 * @param problem The problem.
 * @param body The body literals; the head's variables are bound to each example in turn.
 * @returns The indices of the positives and of the negatives covered.
 *
 * @example The grandparent clause covers no negative; its first literal alone does
 * const problem = foilProblem(
 *   'parent(ann, bob). parent(bob, cid). parent(cid, dan).',
 *   'grandparent(ann, cid). grandparent(bob, dan).',
 * )
 * print(foilCoverage(problem, [{ relation: 0, args: [0, 2] }, { relation: 0, args: [2, 1] }]))
 * print(foilCoverage(problem, [{ relation: 0, args: [0, 2] }]))
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

/**
 * The body with variables after the head's renumbered in order of first use (after literals were removed).
 *
 * @param arity The target's arity: variables below it are the head's and keep their numbers.
 * @param body The body literals.
 * @returns The body with the other variables numbered from `arity` on.
 */
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
 * and at least as many positives. The last literal is never dropped. Returns the simplified body and the literals
 * removed.
 *
 * @param problem The problem.
 * @param body The body of a learned clause.
 * @returns `body`, the literals kept with their new variables renumbered, and `removed`, the literals dropped.
 *
 * @example A literal the clause does not need
 * const problem = foilProblem(
 *   'parent(ann, bob). parent(bob, cid). parent(cid, dan).',
 *   'grandparent(ann, cid). grandparent(bob, dan).',
 * )
 * const body = [{ relation: 0, args: [0, 2] }, { relation: 0, args: [2, 1] }, { relation: 0, args: [2, 3] }]
 * const simple = foilSimplify(problem, body)
 * print(foilClauseText(problem, body))
 * print(foilClauseText(problem, simple.body))
 * print('removed:', simple.removed)
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
  /** The simplified body. */
  readonly body: readonly FoilLiteral[]
  /** The clause as Prolog text. */
  readonly text: string
  /** Positive examples it covers that were still uncovered when it was learned. */
  readonly newlyCovered: readonly number[]
  /** The clause as grown, before simplification, and the literals simplification removed (as text). */
  readonly grown: string
  readonly removed: readonly string[]
}

/** A candidate literal of one step, with its counts and gain. */
export interface FoilCandidate {
  /** The literal. */
  readonly literal: FoilLiteral
  /** The literal as text. */
  readonly text: string
  /** The positive and negative tuples after it. */
  readonly p: number
  readonly n: number
  /** The positive tuples before it that it extends. */
  readonly t: number
  /** Its FOIL gain. */
  readonly gain: number
  /** The positive and negative examples the clause with it covers. */
  readonly positives: readonly number[]
  readonly negatives: readonly number[]
}

/**
 * What a step of `foilSteps` did: `start`, a `literal` added, a `clause` learned, `stuck` (with why), or `done`.
 */
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
  /** The clause being grown, as text. */
  readonly currentText: string
  /** The state of the refinement search growing it. */
  readonly search: SearchState<FoilNode>
  /** The candidates scored at this step, best first, and the positives and negatives the clause covers now. */
  readonly candidates: readonly FoilCandidate[]
  readonly coveredPositives: readonly number[]
  readonly coveredNegatives: readonly number[]
  /** What this step did. */
  readonly event: FoilEvent
  /** True once FOIL has stopped. */
  readonly terminated: boolean
}

/**
 * FOIL as a step-through algorithm: each step adds the best literal to the clause being grown (showing every
 * candidate's gain), commits a clause that covers no negatives (removing the positives it covers), or stops when the
 * positives are covered, no literal has positive gain, or a limit is reached. Throws `DomainError` when
 * `maxBodyLength` is less than 1.
 *
 * @param problem The problem, as `foilProblem` makes it.
 * @param options The limits on clauses, literals and new variables, and whether recursion is allowed (see
 *   `FoilOptions`).
 * @returns The algorithm: `init` takes no argument; run it with `run` or `trace`.
 *
 * @example Step through learning grandparent
 * const problem = foilProblem(
 *   'parent(ann, bob). parent(bob, cid). parent(cid, dan).',
 *   'grandparent(ann, cid). grandparent(bob, dan).',
 * )
 * const tr = trace(foilSteps(problem), undefined, 10)
 * for (const s of tr.steps) print(s.t, s.event.kind, s.currentText)
 *
 * @example The candidates scored at the first step
 * const problem = foilProblem(
 *   'parent(ann, bob). parent(bob, cid). parent(cid, dan).',
 *   'grandparent(ann, cid). grandparent(bob, dan).',
 * )
 * const s = run(foilSteps(problem), undefined, 1)
 * for (const c of s.candidates.slice(0, 4)) print(c.text, 'gain', c.gain, 'p', c.p, 'n', c.n)
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
  /** Each learned clause as Prolog text. */
  readonly clauses: readonly string[]
  /** The clauses, one per line. */
  readonly program: string
  /** The positive examples (indices) left uncovered. */
  readonly uncovered: readonly number[]
  /** The steps taken. */
  readonly steps: number
  /** Why FOIL stopped, in words. */
  readonly stopped: string
}

/**
 * Run FOIL to the end (see `foilSteps`), with a step limit large enough for every clause to reach the body limit.
 *
 * @param problem The problem, as `foilProblem` makes it.
 * @param options The limits and recursion switch (see `FoilOptions`).
 * @returns The learned clauses as text, the positives left uncovered, and why it stopped.
 *
 * @example Learn grandparent from parent
 * const problem = foilProblem(
 *   'parent(ann, bob). parent(bob, cid). parent(cid, dan).',
 *   'grandparent(ann, cid). grandparent(bob, dan).',
 * )
 * const result = foil(problem)
 * print(result.program)
 * print(result.stopped, 'in', result.steps, 'steps')
 *
 * @example A recursive definition
 * const problem = foilProblem(
 *   'parent(ann, bob). parent(bob, cid). parent(cid, dan).',
 *   'ancestor(ann, bob). ancestor(bob, cid). ancestor(cid, dan). ' +
 *     'ancestor(ann, cid). ancestor(bob, dan). ancestor(ann, dan).',
 * )
 * print(foil(problem).program)
 */
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
