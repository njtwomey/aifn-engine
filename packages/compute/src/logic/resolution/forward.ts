/**
 * Forward chaining: the least Herbrand model of a definite, function-free program (Datalog) by iterating the
 * immediate-consequence operator, the bottom-up counterpart of SLD resolution.
 *
 * Naive bottom-up evaluation (van Emden and Kowalski, 1976; Lloyd, 1987, "Foundations of Logic Programming", §6):
 * iterate $T_P$ from the empty set,
 *
 * $$
 * T_P(I) = \{ h\theta : (h \leftarrow b_1, \dots, b_n) \in P,\ b_i\theta \in I \text{ for all } i \},
 * $$
 *
 * with $\theta$ ground. Round $k$ adds the facts of $T_P \uparrow k$ that $T_P \uparrow (k - 1)$ lacks; the run
 * stops at the fixpoint, the least model. Each derived fact keeps the clause and the substitution that first derived it and the facts it was
 * derived from, so a derivation can be shown. Body goals are matched left to right against the facts known before the
 * round (`matchTerm`), with the tests `=`, `\=`, `==` and `\==` allowed once their arguments are bound. Rules must be
 * range-restricted (every head variable occurs in a body atom) and facts ground; anything else, or a goal of another
 * built-in, is reported in the result's `error` rather than thrown. Clauses are those `parseProgram` returns.
 */
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  applySubstitution,
  composeSubstitutions,
  indicator,
  isGround,
  matchTerm,
  termsEqual,
  termToString,
  unify,
  type Clause,
  type Compound,
  type Substitution,
  type Term,
} from 'aifn-compute/logic/terms'

/** A fact of the least model, with how it was first derived. */
export interface DerivedFact {
  /** The ground atom. */
  readonly term: Term
  /** It as text, e.g. `grandparent(ann, cid)`. */
  readonly text: string
  /** The round that added it: 1 for the program's facts. */
  readonly round: number
  /** The clause (index into the program) that derived it. */
  readonly clause: number
  /** The substitution of the clause's variables that derived it (empty for a fact). */
  readonly substitution: Substitution
  /** The facts (indices into `facts`) its body goals matched, in body order; built-in tests are left out. */
  readonly premises: readonly number[]
}

/** The outcome of `leastHerbrandModel`. */
export interface LeastModelResult {
  /** Every fact derived, in the order added (by round, then clause, then match). */
  readonly facts: readonly DerivedFact[]
  /** The number of rounds that added facts; $T_P \uparrow$ `rounds` is the least model when `fixpoint` holds. */
  readonly rounds: number
  /** True when a further round added nothing. */
  readonly fixpoint: boolean
  /** Why the run stopped early (a limit, an unsafe clause, a goal that cannot be evaluated), or null. */
  readonly error: string | null
}

/** Options of `leastHerbrandModel`. */
export interface LeastModelOptions {
  /** The most rounds after the first before the run stops with an error (default 50). */
  maxRounds?: number
  /** The most facts before the run stops with an error (default 5000). */
  maxFacts?: number
}

/** The built-in tests a body may hold: unification, its negation, identity and its negation. */
const TESTS = new Set(['=/2', '\\=/2', '==/2', '\\==/2'])

/**
 * One built-in test or binding under $\theta$. Throws `DomainError` when a test other than `=` meets an unbound
 * argument (caught by `leastHerbrandModel` and reported).
 *
 * @param goal The goal, a compound of one of the `TESTS` predicates.
 * @param theta The bindings made so far by the clause's earlier goals.
 * @returns $\theta$ extended (by the unifier, for `=`), or null when the test fails.
 */
function builtin(goal: Term, theta: Substitution): Substitution | null {
  const g = applySubstitution(goal, theta) as Compound
  const [a, b] = g.args
  const key = indicator(g)
  if (key === '=/2') {
    const u = unify(a, b)
    return u === null ? null : composeSubstitutions(theta, u)
  }
  if (!isGround(a) || !isGround(b))
    throw new DomainError('leastHerbrandModel', `the test ${termToString(g)} needs its arguments bound`)
  if (key === '\\=/2') return unify(a, b) === null ? theta : null
  if (key === '==/2') return termsEqual(a, b) ? theta : null
  return termsEqual(a, b) ? null : theta
}

/**
 * The least Herbrand model of a definite, function-free program by naive iteration of the immediate-consequence
 * operator $T_P$ from the empty set (van Emden and Kowalski, 1976). Round 1 holds the program's facts; each later
 * round applies every rule to the facts known before it. Problems (a non-ground fact, an unsafe rule, negation, a
 * test on unbound arguments, a limit) end the run with `error` set; nothing is thrown. A plain function of plain
 * data, so it can run in a worker.
 *
 * @param clauses The program's clauses, as `parseProgram(source).clauses`.
 * @param options Limits on the rounds and the facts (see `LeastModelOptions`).
 * @returns The facts with their derivations, the number of rounds, whether the fixpoint was reached, and any error.
 *
 * @example The least model of a family program, round by round
 * import { parseProgram } from 'aifn-compute/logic/terms'
 * const { clauses } = parseProgram(`
 *   parent(ann, bob). parent(bob, cid). parent(bob, dee). parent(cid, eve).
 *   grandparent(X, Z) :- parent(X, Y), parent(Y, Z).`)
 * const m = leastHerbrandModel(clauses)
 * for (const f of m.facts) print('round', f.round, f.text)
 * print('rounds:', m.rounds, 'fixpoint:', m.fixpoint)
 *
 * @example An unsafe rule is reported
 * import { parseProgram } from 'aifn-compute/logic/terms'
 * const { clauses } = parseProgram('p(a). q(X, Y) :- p(X).')
 * print(leastHerbrandModel(clauses).error)
 */
export function leastHerbrandModel(clauses: readonly Clause[], options: LeastModelOptions = {}): LeastModelResult {
  const maxRounds = options.maxRounds ?? 50
  const maxFacts = options.maxFacts ?? 5000
  const facts: DerivedFact[] = []
  const known = new Map<string, number>()
  const byPredicate = new Map<string, number[]>()
  const add = (f: Omit<DerivedFact, 'text'>) => {
    const text = termToString(f.term)
    if (known.has(text)) return
    known.set(text, facts.length)
    const key = indicator(f.term)
    if (!byPredicate.has(key)) byPredicate.set(key, [])
    byPredicate.get(key)!.push(facts.length)
    facts.push({ ...f, text })
  }
  const stop = (rounds: number, error: string | null, fixpoint = false): LeastModelResult => ({
    facts,
    rounds,
    fixpoint,
    error,
  })

  // Round 1: T_P ↑ 1, the program's facts.
  for (let c = 0; c < clauses.length; c++) {
    const { head, body } = clauses[c]
    if (body.length > 0) continue
    if (!isGround(head)) return stop(0, `the fact ${termToString(head)} has a variable: facts must be ground`)
    add({ term: head, round: 1, clause: c, substitution: [], premises: [] })
  }
  let rounds = facts.length > 0 ? 1 : 0
  for (let round = 2; round <= maxRounds + 1; round++) {
    const before = facts.length
    // Facts known at the start of the round: T_P is applied to T_P ↑ (round − 1) only.
    const visible = (key: string) => (byPredicate.get(key) ?? []).filter((i) => i < before)
    const found: Omit<DerivedFact, 'text'>[] = []
    for (let c = 0; c < clauses.length; c++) {
      const { head, body } = clauses[c]
      if (body.length === 0) continue
      let partial: { theta: Substitution; premises: number[] }[] = [{ theta: [], premises: [] }]
      try {
        for (const goal of body) {
          if (goal.kind === 'var' || goal.kind === 'number')
            throw new DomainError('leastHerbrandModel', `the goal ${termToString(goal)} cannot be called`)
          const key = indicator(goal)
          const next: { theta: Substitution; premises: number[] }[] = []
          if (TESTS.has(key)) {
            for (const p of partial) {
              const theta = builtin(goal, p.theta)
              if (theta) next.push({ theta, premises: p.premises })
            }
          } else {
            if (key.startsWith('\\+/'))
              throw new DomainError('leastHerbrandModel', 'negation (\\+) is not part of a definite program')
            for (const p of partial)
              for (const i of visible(key)) {
                const theta = matchTerm(goal, facts[i].term, p.theta)
                if (theta) next.push({ theta, premises: [...p.premises, i] })
              }
          }
          partial = next
          if (partial.length === 0) break
        }
      } catch (e) {
        if (!(e instanceof DomainError)) throw e
        return stop(rounds, `clause ${c + 1}: ${e.message}`)
      }
      for (const p of partial) {
        const term = applySubstitution(head, p.theta)
        if (!isGround(term))
          return stop(
            rounds,
            `clause ${c + 1} derives ${termToString(term)}, which has a variable: every head variable must occur in a body atom`,
          )
        found.push({ term, round, clause: c, substitution: p.theta, premises: p.premises })
      }
    }
    for (const f of found) {
      add(f)
      if (facts.length > maxFacts) return stop(round, `more than ${maxFacts} facts: stopped`)
    }
    if (facts.length === before) return stop(rounds, null, true)
    rounds = round
  }
  return stop(rounds, `no fixpoint after ${maxRounds} rounds`)
}

/** The outcome of `herbrandBaseSize`. */
export interface HerbrandBase {
  /** The constants that occur in the program (atoms, and numbers as text). */
  readonly constants: string[]
  /** The predicates of the program as `name/arity`, built-in tests left out. */
  readonly predicates: string[]
  /** The number of ground atoms, $\sum_p \lvert C \rvert^{\operatorname{arity}(p)}$. */
  readonly size: number
}

/**
 * The size of the Herbrand base of a function-free program: $\sum_p \lvert C \rvert^{\operatorname{arity}(p)}$ over
 * its predicates $p$, with $C$ the constants that occur in it (one invented constant when there are none). The least
 * model is a subset of it.
 *
 * @param clauses The program's clauses, as `parseProgram(source).clauses`.
 * @returns The constants, the predicates and the size of the base.
 *
 * @example Four binary predicates over five constants
 * import { parseProgram } from 'aifn-compute/logic/terms'
 * const { clauses } = parseProgram(`
 *   parent(ann, bob). parent(bob, cid). parent(bob, dee). parent(cid, eve).
 *   grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
 *   ancestor(X, Y) :- parent(X, Y).
 *   sibling(X, Y) :- parent(P, X), parent(P, Y), X \\= Y.`)
 * const base = herbrandBaseSize(clauses)
 * print('constants:', base.constants, 'predicates:', base.predicates)
 * print('size:', base.size)
 */
export function herbrandBaseSize(clauses: readonly Clause[]): HerbrandBase {
  const constants = new Set<string>()
  const predicates = new Map<string, number>()
  const walk = (t: Term) => {
    if (t.kind === 'atom') constants.add(t.name)
    else if (t.kind === 'number') constants.add(String(t.value))
    else if (t.kind === 'compound') t.args.forEach(walk)
  }
  for (const { head, body } of clauses)
    for (const g of [head, ...body]) {
      const key = g.kind === 'atom' || g.kind === 'compound' ? indicator(g) : null
      if (!key || TESTS.has(key)) continue
      predicates.set(key, g.kind === 'compound' ? g.args.length : 0)
      if (g.kind === 'compound') g.args.forEach(walk)
    }
  const c = Math.max(1, constants.size)
  const size = [...predicates.values()].reduce((s, arity) => s + c ** arity, 0)
  return { constants: [...constants], predicates: [...predicates.keys()], size }
}
