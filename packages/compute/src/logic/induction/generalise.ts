/**
 * Generality of terms and clauses (Plotkin 1970): $\theta$-subsumption and the least general generalisation, each
 * with a traced version that records the search one comparison or one attempt at a time.
 *
 * A clause $C$ $\theta$-subsumes $D$ when some substitution $\theta$ maps every literal of $C$ onto a literal of $D$
 * ($C\theta \subseteq D$): $C$ is then at least as general as $D$, and $C \models D$. The search for $\theta$ is
 * backtracking over the literals of $C$ (Gottlob and Leitsch, 1985, "On the efficiency of subsumption algorithms",
 * JACM 32(2)), each matched by one-way matching onto a literal of $D$ with the same predicate; `thetaSubsumption`
 * takes the literals with fewest candidates first, `thetaSubsumptionTrace` by default in the written order, so a
 * reader can follow it. The least general generalisation (lgg) of two terms replaces each pair of differing subterms
 * by a variable, the same pair always by the same variable, and keeps identical subterms (Plotkin's
 * $\operatorname{lgg}(t, t) = t$): $\operatorname{lgg}(f(a, g(a)), f(b, g(b))) = f(X, g(X))$. The lgg of two clauses
 * pairs every two literals of the same predicate and sign with one table of pairs; it $\theta$-subsumes both clauses,
 * and every clause that $\theta$-subsumes both $\theta$-subsumes it. Golem's relative lgg (Muggleton and Feng, 1990)
 * is the lgg of $e_1 \leftarrow B$ and $e_2 \leftarrow B$ for ground background facts $B$, without the literals $B$
 * makes true whatever the head's values, reduced.
 *
 * Clauses are Horn clauses, a head and a body; terms are those of `aifn-compute/logic/terms`. Literals of a clause are
 * numbered from 0, the head, so body literal $k$ is number $k + 1$. The examples build terms with `parseTerm` and
 * clauses with `parseProgram` of `aifn-compute/logic/terms`.
 */
import {
  compound,
  matchTerm,
  renameVariables,
  termsEqual,
  termToString,
  termVariables,
  variable,
  type Substitution,
  type Term,
} from 'aifn-compute/logic/terms'

/** A clause as a head and body literals (a fact has an empty body). */
export interface HornClause {
  /** The head literal. */
  readonly head: Term
  /** The body literals (empty for a fact). */
  readonly body: readonly Term[]
}

/**
 * The predicate symbol and arity of a literal, such as `parent/2` (`name/0` for an atom); a variable or a number,
 * which is no literal, gives `?var` or `?number`.
 *
 * @param t The literal.
 * @returns Its predicate indicator; two literals can be matched or generalised together only when these are equal.
 *
 * @example Predicate indicators
 * import { parseTerm } from 'aifn-compute/logic/terms'
 * print(predicateOf(parseTerm('parent(ann, bob)')), predicateOf(parseTerm('cyclic')))
 */
export function predicateOf(t: Term): string {
  if (t.kind === 'compound') return `${t.functor}/${t.args.length}`
  if (t.kind === 'atom') return `${t.name}/0`
  return `?${t.kind}`
}

/**
 * A clause's literals in one list: the head, then the body.
 *
 * @param c The clause.
 * @returns The literals, numbered from 0 (the head).
 */
const literalsOf = (c: HornClause): Term[] => [c.head, ...c.body]

/**
 * The largest variable id in the terms, or $-1$.
 *
 * @param terms The terms.
 * @returns The largest id.
 */
const maxId = (terms: readonly Term[]) => Math.max(-1, ...termVariables(...terms).map((v) => v.id))

// ── Least general generalisation ─────────────────────────────────────────────────────────────────────────────────────

/** What one comparison of `termLggTraced` did: kept, descended into, replaced by a new or an earlier variable. */
export type LggAction = 'same' | 'descend' | 'new' | 'reuse'

/** One comparison of two subterms. */
export interface LggStep {
  /** The position of the pair in the term, as argument indices from the root (empty for the root). */
  readonly path: readonly number[]
  /** The subterm of the first term. */
  readonly left: Term
  /** The subterm of the second term. */
  readonly right: Term
  /** Kept because identical, compared argument by argument, replaced by a new variable, or by an earlier one. */
  readonly action: LggAction
  /** The variable that replaces the pair (for `new` and `reuse`), else null. */
  readonly variable: Term | null
}

/** A pair of differing subterms and the variable that stands for it. */
export interface LggPair {
  /** The subterm of the first term. */
  readonly left: Term
  /** The subterm of the second term. */
  readonly right: Term
  /** The variable of the generalisation that stands for the pair. */
  readonly variable: Term
}

/** A table of pairs shared across the terms of one generalisation. */
export interface LggTable {
  /** The pairs met so far, in order of creation; the variable of pair $i$ is named by $i$ (A, B, …). */
  readonly pairs: LggPair[]
  /**
   * The id of the first new variable (default 0); later ones count up from it. Set it above every variable id of the
   * terms generalised, so that a variable kept from them is never confused with a new one.
   */
  readonly firstId?: number
}

/**
 * The name of the lgg variable numbered `i`: A to Z, then `V26`, `V27`, ….
 *
 * @param i The variable's number, from 0.
 * @returns Its name.
 */
const varName = (i: number) => (i < 26 ? String.fromCharCode(65 + i) : `V${i}`)

/**
 * A new, empty table of pairs.
 *
 * @param firstId The id of the first new variable: 0 for ground terms, else one above their largest variable id.
 * @returns The table.
 *
 * @example One table shared by two generalisations
 * import { parseTerm, termToString } from 'aifn-compute/logic/terms'
 * const table = lggTable()
 * termLggTraced(parseTerm('p(a, b)'), parseTerm('p(c, d)'), table)
 * print(termToString(termLggTraced(parseTerm('q(b)'), parseTerm('q(d)'), table)), table.pairs.length, 'pairs')
 */
export const lggTable = (firstId = 0): LggTable => ({ pairs: [], firstId })

/**
 * The least general generalisation of two terms with every comparison recorded, using and extending a table of pairs:
 * identical subterms are kept, compounds with the same functor and arity are compared argument by argument, and any
 * other pair becomes the table's variable for it (a new one, named A, B, … in order of creation, the first time).
 *
 * @param a The first term. Its variables are generalised like constants: a variable paired with itself is kept, a
 *   pair of different variables becomes a variable of the result.
 * @param b The second term.
 * @param table The table of pairs met so far, extended with the new ones; share one across the literals of a clause.
 *   Left out, a new table whose variable ids start above those of `a` and `b`.
 * @param steps Where the comparisons are appended, root first, arguments left to right.
 * @param path The position of `a` and `b` in an enclosing term, prefixed to every step's `path`.
 * @returns The generalisation.
 *
 * @example Each comparison of an anti-unification
 * import { parseTerm, termToString } from 'aifn-compute/logic/terms'
 * const steps = []
 * const g = termLggTraced(parseTerm('p(f(a, b), g(a), b)'), parseTerm('p(f(c, d), g(c), e)'), undefined, steps)
 * print(termToString(g))
 * for (const s of steps) print(s.path, termToString(s.left), 'vs', termToString(s.right), s.action)
 */
export function termLggTraced(
  a: Term,
  b: Term,
  table: LggTable = lggTable(maxId([a, b]) + 1),
  steps: LggStep[] = [],
  path: readonly number[] = [],
): Term {
  if (termsEqual(a, b)) {
    steps.push({ path, left: a, right: b, action: 'same', variable: null })
    return a
  }
  if (a.kind === 'compound' && b.kind === 'compound' && a.functor === b.functor && a.args.length === b.args.length) {
    steps.push({ path, left: a, right: b, action: 'descend', variable: null })
    const args = a.args.map((x, i) => termLggTraced(x, b.args[i], table, steps, [...path, i]))
    return compound(a.functor, args)
  }
  const seen = table.pairs.find((p) => termsEqual(p.left, a) && termsEqual(p.right, b))
  if (seen) {
    steps.push({ path, left: a, right: b, action: 'reuse', variable: seen.variable })
    return seen.variable
  }
  const n = table.pairs.length
  const v = variable(varName(n), (table.firstId ?? 0) + n)
  table.pairs.push({ left: a, right: b, variable: v })
  steps.push({ path, left: a, right: b, action: 'new', variable: v })
  return v
}

/**
 * The least general generalisation of two terms (Plotkin; Reynolds 1970). Its new variables are named A, B, … in
 * order of creation, with ids from one above the largest id in `a` and `b` (from 0 for ground terms).
 *
 * @param a The first term. Its variables are treated like constants: a variable paired with itself is kept, and a
 *   pair of different variables becomes a variable of the result.
 * @param b The second term.
 * @returns The most specific term that both `a` and `b` are instances of.
 *
 * @example The same pair of constants becomes the same variable
 * import { parseTerm, termToString } from 'aifn-compute/logic/terms'
 * print(termToString(termLgg(parseTerm('f(a, g(a))'), parseTerm('f(b, g(b))'))))
 * print(termToString(termLgg(parseTerm('f(a, b)'), parseTerm('f(b, a)'))))
 */
export function termLgg(a: Term, b: Term): Term {
  return termLggTraced(a, b)
}

/** One pair of literals generalised in a clause lgg. */
export interface LiteralPairStep {
  /** The literal's number in the first clause (0 is the head, $k + 1$ body literal $k$). */
  readonly left: number
  /** The literal's number in the second clause. */
  readonly right: number
  /** The generalised literal. */
  readonly literal: Term
  /** Whether the literal repeats one already in the body (and is dropped). */
  readonly duplicate: boolean
  /** The comparisons made for this pair. */
  readonly comparisons: readonly LggStep[]
  /** The table of pairs after this literal. */
  readonly pairs: readonly LggPair[]
}

/**
 * The least general generalisation of two clauses, one literal pair per step: the lgg of the heads first, then of
 * every pair of body literals of the same predicate, the first clause's literals in the outer loop, with one table of
 * pairs throughout, so a pair of constants is one variable everywhere. Not reduced (`reduceClause` removes redundant
 * literals). Returns null when the heads have different predicates.
 *
 * @param c The first clause. Its variables are generalised like constants.
 * @param d The second clause.
 * @returns The steps and the generalised clause (repeated literals dropped), or null.
 *
 * @example Generalise two daughter clauses pair by pair
 * import { clauseToString, parseProgram, termToString } from 'aifn-compute/logic/terms'
 * const c = parseProgram('daughter(mary, ann) :- female(mary), female(ann), parent(ann, mary).').clauses[0]
 * const d = parseProgram('daughter(eve, tom) :- female(eve), parent(tom, eve).').clauses[0]
 * const { steps, clause } = clauseLggTrace(c, d)
 * for (const s of steps) print(s.left, s.right, termToString(s.literal))
 * print(clauseToString(clause))
 */
export function clauseLggTrace(c: HornClause, d: HornClause): { steps: LiteralPairStep[]; clause: HornClause } | null {
  if (predicateOf(c.head) !== predicateOf(d.head)) return null
  const table = lggTable(maxId([...literalsOf(c), ...literalsOf(d)]) + 1)
  const steps: LiteralPairStep[] = []
  const comparisons: LggStep[] = []
  const head = termLggTraced(c.head, d.head, table, comparisons)
  steps.push({ left: 0, right: 0, literal: head, duplicate: false, comparisons, pairs: [...table.pairs] })
  const body: Term[] = []
  c.body.forEach((x, i) =>
    d.body.forEach((y, j) => {
      if (predicateOf(x) !== predicateOf(y)) return
      const cmp: LggStep[] = []
      const lit = termLggTraced(x, y, table, cmp)
      const duplicate = body.some((l) => termsEqual(l, lit))
      if (!duplicate) body.push(lit)
      steps.push({ left: i + 1, right: j + 1, literal: lit, duplicate, comparisons: cmp, pairs: [...table.pairs] })
    }),
  )
  return { steps, clause: { head, body } }
}

/**
 * The least general generalisation of two clauses: the lgg of the heads, and the lggs of every pair of body literals
 * of the same predicate, with one table of variables throughout. Not reduced (it may hold literals that others
 * $\theta$-subsume; `reduceClause` removes them). Returns null when the heads have different predicates.
 *
 * @param c The first clause. Its variables are generalised like constants.
 * @param d The second clause.
 * @returns The lgg, with repeated body literals dropped, or null.
 *
 * @example Two instances of the same pattern
 * import { clauseToString, parseProgram } from 'aifn-compute/logic/terms'
 * const [c, d] = parseProgram('p(a) :- q(a, b), r(b). p(c) :- q(c, d), r(d).').clauses
 * print(clauseToString(clauseLgg(c, d)))
 * print('different heads:', clauseLgg(c, parseProgram('s(a).').clauses[0]))
 */
export function clauseLgg(c: HornClause, d: HornClause): HornClause | null {
  return clauseLggTrace(c, d)?.clause ?? null
}

/** The stages of a relative least general generalisation (`relativeLgg`). */
export interface RelativeLgg {
  /** The lgg of $e_1 \leftarrow B$ and $e_2 \leftarrow B$. */
  readonly raw: HornClause
  /** The steps of that lgg, one per literal pair, as `clauseLggTrace` records them. */
  readonly steps: readonly LiteralPairStep[]
  /** The body literals dropped because they are ground (facts of $B$). */
  readonly ground: readonly Term[]
  /** The body literals dropped because no chain of shared variables links them to the head. */
  readonly unlinked: readonly Term[]
  /** The head-linked literals left. */
  readonly linked: HornClause
  /** The linked clause after Plotkin's reduction (redundant literals removed): the result. */
  readonly clause: HornClause
}

/**
 * The body literals linked to the head: those sharing a variable with the head, or with a literal already linked.
 *
 * @param c The clause.
 * @returns The linked literals and the others, each in body order.
 *
 * @example A literal linked through another
 * import { parseProgram, termToString } from 'aifn-compute/logic/terms'
 * const c = parseProgram('p(X) :- q(X, Y), r(Y), s(Z).').clauses[0]
 * const { linked, unlinked } = headLinked(c)
 * print('linked:', linked.map((t) => termToString(t)), 'unlinked:', unlinked.map((t) => termToString(t)))
 */
export function headLinked(c: HornClause): { linked: Term[]; unlinked: Term[] } {
  const vars = new Set(termVariables(c.head).map((v) => v.id))
  const linked = new Set<number>()
  let grew = true
  while (grew) {
    grew = false
    c.body.forEach((l, i) => {
      if (linked.has(i)) return
      const ids = termVariables(l).map((v) => v.id)
      if (ids.some((id) => vars.has(id))) {
        linked.add(i)
        ids.forEach((id) => vars.add(id))
        grew = true
      }
    })
  }
  return {
    linked: c.body.filter((_, i) => linked.has(i)),
    unlinked: c.body.filter((_, i) => !linked.has(i)),
  }
}

/**
 * Golem's relative least general generalisation of two ground examples with respect to ground background facts $B$
 * (Muggleton and Feng, 1990): the lgg of $e_1 \leftarrow B$ and $e_2 \leftarrow B$, without the literals $B$ makes
 * true whatever the head's values (ground facts of $B$, and groups of literals linked to the head by no variable,
 * each of which maps into $B$), then reduced. Dropping those literals first keeps the reduction, which is exponential
 * in the worst case, small: reducing the raw lgg gives the same clause far more slowly.
 *
 * @param e1 The first example, a ground atom.
 * @param e2 The second example, of the same predicate.
 * @param background The ground facts $B$.
 * @returns The raw lgg and its steps, the literals dropped, the linked clause and the reduced result; null when the
 *   examples have different predicates.
 *
 * @example Golem's daughter clause
 * import { clauseToString, parseProgram, parseTerm } from 'aifn-compute/logic/terms'
 * const B = parseProgram(`parent(ann, mary). parent(ann, tom). parent(tom, eve). parent(tom, ian).
 *   female(ann). female(mary). female(eve).`).clauses.map((c) => c.head)
 * const r = relativeLgg(parseTerm('daughter(mary, ann)'), parseTerm('daughter(eve, tom)'), B)
 * print('raw:', r.raw.body.length, 'ground:', r.ground.length, 'unlinked:', r.unlinked.length)
 * print(clauseToString(r.clause))
 */
export function relativeLgg(e1: Term, e2: Term, background: readonly Term[]): RelativeLgg | null {
  const lgg = clauseLggTrace({ head: e1, body: background }, { head: e2, body: background })
  if (!lgg) return null
  const ground = lgg.clause.body.filter((l) => termVariables(l).length === 0)
  const open = { head: lgg.clause.head, body: lgg.clause.body.filter((l) => termVariables(l).length > 0) }
  const { linked, unlinked } = headLinked(open)
  const linkedClause = { head: open.head, body: linked }
  return {
    raw: lgg.clause,
    steps: lgg.steps,
    ground,
    unlinked,
    linked: linkedClause,
    clause: reduceClause(linkedClause),
  }
}

// ── θ-subsumption ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * $D$ with its variables renamed to ids above every id of $C$, renamed as one term so that shared variables stay
 * shared; names are kept.
 *
 * @param c The clause $C$.
 * @param d The clause $D$ to rename.
 * @returns $D$ renamed apart from $C$.
 *
 * @example Rename apart before matching
 * import { parseProgram, termVariables } from 'aifn-compute/logic/terms'
 * const [c, d] = parseProgram('p(X) :- q(X, Y). p(X) :- q(X, X).').clauses
 * const r = renameApart(c, d)
 * print(termVariables(r.head, ...r.body).map((v) => `${v.name}#${v.id}`))
 */
export function renameApart(c: HornClause, d: HornClause): HornClause {
  const whole = renameVariables(compound('$clause', literalsOf(d)), maxId(literalsOf(c)) + 1)
  const parts = whole.kind === 'compound' ? whole.args : [whole]
  return { head: parts[0], body: parts.slice(1) }
}

/**
 * $D$'s literals renamed apart from $C$, numbered as the steps of `thetaSubsumptionTrace` number them (0 is the head).
 *
 * @param c The clause $C$.
 * @param d The clause $D$.
 * @returns The head and then the body literals of the renamed $D$.
 *
 * @example The targets of a subsumption search
 * import { parseProgram, termToString } from 'aifn-compute/logic/terms'
 * const [c, d] = parseProgram('p(X) :- q(X, Y). p(a) :- q(a, Y), r(Y).').clauses
 * print(targetLiterals(c, d).map((t) => termToString(t)))
 */
export function targetLiterals(c: HornClause, d: HornClause): Term[] {
  return literalsOf(renameApart(c, d))
}

/** What a step of the subsumption search did. */
export type SubsumptionEvent = 'start' | 'match' | 'clash' | 'backtrack' | 'subsumes' | 'fails'

/** One state of the subsumption search. */
export interface SubsumptionStep {
  /** What this step did. */
  readonly event: SubsumptionEvent
  /** The literal of $C$ treated (0 is the head, $k + 1$ body literal $k$), or null at the start and end. */
  readonly literal: number | null
  /** The literal of $D$ it was matched against (same numbering, of `targetLiterals`), or null. */
  readonly target: number | null
  /** The target chosen for each literal of $C$ mapped so far, in the order the literals are taken. */
  readonly assignment: readonly number[]
  /** The substitution built so far, binding variables of $C$ to terms of the renamed $D$. */
  readonly theta: Substitution
  /** What happened, in words. */
  readonly note: string
}

/** Options of `thetaSubsumptionTrace`. */
export interface SubsumptionTraceOptions {
  /**
   * The order in which the body literals of $C$ are taken: `'written'` (default, the clause's order) or
   * `'constrained'` (fewest candidate literals in $D$ first, as `thetaSubsumption`). The head is always first.
   */
  order?: 'written' | 'constrained'
  /** Stop after this many states (default 2000), ending with `fails` and a note saying so. */
  maxSteps?: number
  /** How terms print in the notes (default `termToString`). */
  text?: (t: Term) => string
}

/**
 * The backtracking search for $\theta$ with $C\theta \subseteq D$, one definition for the traced and the plain
 * versions. The note of a step is built only when `emit` is given.
 *
 * @param c The clause $C$.
 * @param d The clause $D$.
 * @param order The order of $C$'s body literals (see `SubsumptionTraceOptions`).
 * @param maxSteps The most states before giving up (`Infinity` for no limit).
 * @param emit Called with every state, or null for no trace.
 * @param show How terms print in the notes.
 * @returns $\theta$, null when $C$ does not $\theta$-subsume $D$, or undefined when the step limit was reached.
 */
function subsumptionSearch(
  c: HornClause,
  d: HornClause,
  order: 'written' | 'constrained',
  maxSteps: number,
  emit: ((s: SubsumptionStep) => void) | null,
  show: (t: Term) => string,
): Substitution | null | undefined {
  const cl = literalsOf(c)
  const dl = targetLiterals(c, d)
  const name = (k: number) => (k === 0 ? 'the head' : `body literal ${k}`)
  // The literals of D each literal of C may map onto: the head onto the head, a body literal onto a body literal of
  // the same predicate and arity.
  const targetsOf = (k: number) =>
    k === 0
      ? predicateOf(dl[0]) === predicateOf(cl[0])
        ? [0]
        : []
      : dl.flatMap((m, j) => (j > 0 && predicateOf(m) === predicateOf(cl[k]) ? [j] : []))
  const seq = cl.map((_, k) => k)
  // Most constrained literals first (a stable sort, so ties keep the written order).
  if (order === 'constrained') {
    const body = seq.slice(1).sort((x, y) => targetsOf(x).length - targetsOf(y).length)
    seq.splice(1, body.length, ...body)
  }
  const targets = seq.map(targetsOf)
  let count = 0
  const record = (
    event: SubsumptionEvent,
    literal: number | null,
    target: number | null,
    assignment: readonly number[],
    theta: Substitution,
    note: () => string,
  ) => {
    count++
    if (emit) emit({ event, literal, target, assignment: [...assignment], theta, note: note() })
  }
  record(
    'start',
    null,
    null,
    [],
    [],
    () => `Map the ${cl.length} literals of C onto literals of D in order, the head first.`,
  )
  const choice: number[] = [0]
  const thetas: Substitution[] = [[]]
  const assignment: number[] = []
  let i = 0
  while (count < maxSteps) {
    if (i === seq.length) {
      record(
        'subsumes',
        null,
        null,
        assignment,
        thetas[i],
        () => 'Every literal of C is mapped onto a literal of D, so C θ-subsumes D.',
      )
      return thetas[i]
    }
    const candidates = targets[i]
    if (choice[i] >= candidates.length) {
      if (i === 0) {
        record('fails', 0, null, [], [], () =>
          candidates.length === 0
            ? 'The heads have different predicates, so no substitution maps C into D.'
            : 'Every choice has been tried, so C does not θ-subsume D.',
        )
        return null
      }
      const k = seq[i]
      const reason =
        candidates.length === 0
          ? `no literal of D has the predicate ${predicateOf(cl[k])}`
          : `${name(k)} of C, ${show(cl[k])}, has no target left`
      i -= 1
      assignment.pop()
      choice[i] += 1
      record(
        'backtrack',
        seq[i],
        null,
        assignment,
        thetas[i],
        () => `Backtrack: ${reason}. Undo the choice for ${name(seq[i])} and try its next target.`,
      )
      continue
    }
    const k = seq[i]
    const j = candidates[choice[i]]
    const theta = matchTerm(cl[k], dl[j], thetas[i])
    if (theta) {
      assignment.push(j)
      thetas[i + 1] = theta
      const before = thetas[i].length
      record(
        'match',
        k,
        j,
        assignment,
        theta,
        () =>
          `${show(cl[k])} matches ${show(dl[j])}${
            theta.length > before
              ? `, binding ${theta
                  .slice(before)
                  .map((b) => `${b.variable.name} ↦ ${show(b.value)}`)
                  .join(', ')}`
              : ' with the bindings already made'
          }.`,
      )
      i += 1
      choice[i] = 0
    } else {
      record(
        'clash',
        k,
        j,
        assignment,
        thetas[i],
        () => `${show(cl[k])} does not match ${show(dl[j])} under the bindings so far.`,
      )
      choice[i] += 1
    }
  }
  record('fails', null, null, [], [], () => `Stopped after ${maxSteps} steps.`)
  return undefined
}

/**
 * Every state of the backtracking search for $\theta$ with $C\theta \subseteq D$, from the start to `subsumes` or
 * `fails`, one match attempt per state: the literals of $C$ in turn (the head first), each tried against the next
 * literal of $D$ with its predicate by one-way matching, extending $\theta$; when no literal of $D$ is left for one,
 * the search backtracks to the previous literal and tries its next target. $D$'s variables are renamed apart first
 * (`targetLiterals`) and act as constants. The final event agrees with `thetaSubsumes` in either order.
 *
 * @param c The clause $C$, the candidate for the more general one; its variables may be bound.
 * @param d The clause $D$; renamed apart from $C$, its variables act as constants.
 * @param options The order of $C$'s literals, the step limit and the printing of the notes (see
 *   `SubsumptionTraceOptions`).
 * @returns The states, the first with nothing mapped; the last is `subsumes` ($\theta$ found) or `fails` (none
 *   exists, or the step limit was reached).
 *
 * @example A search that needs a backtrack
 * import { parseProgram } from 'aifn-compute/logic/terms'
 * const [c, d] = parseProgram('p(X) :- q(X, Y), r(Y). p(a) :- q(a, b), q(a, c), r(c).').clauses
 * for (const s of thetaSubsumptionTrace(c, d)) print(s.event, '|', s.note)
 *
 * @example A triangle does not map into a 4-cycle
 * import { parseProgram } from 'aifn-compute/logic/terms'
 * const [c, d] = parseProgram(`cyclic :- e(X, Y), e(Y, Z), e(Z, X).
 *   cyclic :- e(a, b), e(b, a), e(b, c), e(c, b), e(c, d), e(d, c), e(d, a), e(a, d).`).clauses
 * const steps = thetaSubsumptionTrace(c, d)
 * print(steps.length, 'states, ending', steps[steps.length - 1].event)
 */
export function thetaSubsumptionTrace(
  c: HornClause,
  d: HornClause,
  options: SubsumptionTraceOptions = {},
): SubsumptionStep[] {
  const steps: SubsumptionStep[] = []
  const show = options.text ?? ((t: Term) => termToString(t))
  subsumptionSearch(c, d, options.order ?? 'written', options.maxSteps ?? 2000, (s) => steps.push(s), show)
  return steps
}

/**
 * A substitution $\theta$ with $C\theta \subseteq D$ (head onto head, each body literal of $C$ onto some body literal
 * of $D$), or null. $D$'s variables are held fixed (renamed apart first). Backtracking search over the choices for
 * each literal, the literals with fewest candidates first; `thetaSubsumptionTrace` records the same search.
 *
 * @param c The clause $C$, the more general one, whose variables are bound.
 * @param d The clause $D$. Its variables are renamed to ids above those of $C$ (names kept), and act as constants.
 * @returns $\theta$, binding the variables of $C$ to terms of the renamed $D$, or null when $C$ does not
 *   $\theta$-subsume $D$.
 *
 * @example A general clause onto a specific one
 * import { parseProgram, substitutionToString } from 'aifn-compute/logic/terms'
 * const [general, specific] = parseProgram('p(X) :- q(X, Y). p(a) :- q(a, b), r(b).').clauses
 * print(substitutionToString(thetaSubsumption(general, specific)))
 * print('converse:', thetaSubsumption(specific, general))
 */
export function thetaSubsumption(c: HornClause, d: HornClause): Substitution | null {
  return subsumptionSearch(c, d, 'constrained', Infinity, null, termToString) ?? null
}

/**
 * True when $C$ $\theta$-subsumes $D$ ($C$ is at least as general as $D$).
 *
 * @param c The clause $C$.
 * @param d The clause $D$.
 * @returns Whether `thetaSubsumption` finds a substitution.
 *
 * @example Subsumption goes one way
 * import { parseProgram } from 'aifn-compute/logic/terms'
 * const [general, specific] = parseProgram('p(X) :- q(X, Y). p(a) :- q(a, b), r(b).').clauses
 * print(thetaSubsumes(general, specific), thetaSubsumes(specific, general))
 */
export function thetaSubsumes(c: HornClause, d: HornClause): boolean {
  return thetaSubsumption(c, d) !== null
}

/**
 * The clause with redundant body literals removed (Plotkin's reduction): a literal is dropped when the clause with it
 * $\theta$-subsumes the clause without it (the converse always holds), so both are equivalent. Greedy, last literal
 * first.
 *
 * @param c The clause to reduce; not modified.
 * @returns The reduced clause, with the same head and the body literals kept in order.
 *
 * @example A literal that adds nothing
 * import { clauseToString, parseProgram } from 'aifn-compute/logic/terms'
 * const c = parseProgram('p(X) :- q(X, Y), q(X, Z).').clauses[0]
 * print(clauseToString(reduceClause(c)))
 */
export function reduceClause(c: HornClause): HornClause {
  let body = [...c.body]
  for (let i = body.length - 1; i >= 0; i--) {
    const without = { head: c.head, body: body.filter((_, j) => j !== i) }
    if (thetaSubsumes({ head: c.head, body }, without)) body = without.body
  }
  return { head: c.head, body }
}
