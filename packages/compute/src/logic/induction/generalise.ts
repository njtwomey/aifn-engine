/**
 * Generality of terms and clauses (Plotkin 1970): θ-subsumption and the least general generalisation.
 *
 * A clause C θ-subsumes D when some substitution θ maps every literal of C onto a literal of D (Cθ ⊆ D): C is then at
 * least as general as D, and C ⊨ D. The least general generalisation (lgg) of two terms replaces each pair of
 * differing subterms by a variable, the same pair always by the same variable: lgg(f(a, g(a)), f(b, g(b))) = f(X, g(X)).
 * The lgg of two clauses pairs every two literals of the same predicate and sign; it θ-subsumes both clauses, and every
 * clause that θ-subsumes both θ-subsumes it.
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
  readonly head: Term
  readonly body: readonly Term[]
}

/** The shared table of an lgg: each pair of differing subterms gets one variable. */
class Pairs {
  private readonly table = new Map<string, Term>()
  private count = 0
  private readonly names: (i: number) => string
  constructor(names: (i: number) => string) {
    this.names = names
  }
  get(a: Term, b: Term): Term {
    const key = `${termToString(a)}\u0000${termToString(b)}`
    let v = this.table.get(key)
    if (!v) {
      v = variable(this.names(this.count), this.count)
      this.count++
      this.table.set(key, v)
    }
    return v
  }
}

const defaultName = (i: number) => (i < 26 ? String.fromCharCode(65 + i) : `V${i}`)

function lggWith(a: Term, b: Term, pairs: Pairs): Term {
  if (termsEqual(a, b) && a.kind !== 'var') return a
  if (a.kind === 'compound' && b.kind === 'compound' && a.functor === b.functor && a.args.length === b.args.length)
    return compound(
      a.functor,
      a.args.map((x, i) => lggWith(x, b.args[i], pairs)),
    )
  return pairs.get(a, b)
}

/** The least general generalisation of two terms (Plotkin; Reynolds 1970). Its variables are named A, B, …. */
export function termLgg(a: Term, b: Term): Term {
  return lggWith(a, b, new Pairs(defaultName))
}

const predicateOf = (t: Term) =>
  t.kind === 'compound' ? `${t.functor}/${t.args.length}` : t.kind === 'atom' ? `${t.name}/0` : ''

/**
 * The least general generalisation of two clauses: the lgg of the heads, and the lggs of every pair of body literals
 * of the same predicate, with one table of variables throughout, so a pair of constants is one variable everywhere.
 * Not reduced (it may hold literals that others θ-subsume; `reduceClause` removes them). Returns null when the heads
 * have different predicates.
 */
export function clauseLgg(c: HornClause, d: HornClause): HornClause | null {
  if (predicateOf(c.head) !== predicateOf(d.head)) return null
  const pairs = new Pairs(defaultName)
  const head = lggWith(c.head, d.head, pairs)
  const body: Term[] = []
  const seen = new Set<string>()
  for (const x of c.body)
    for (const y of d.body) {
      if (predicateOf(x) !== predicateOf(y)) continue
      const g = lggWith(x, y, pairs)
      const key = termToString(g)
      if (!seen.has(key)) {
        seen.add(key)
        body.push(g)
      }
    }
  return { head, body }
}

/** The largest variable id in the terms, or −1. */
const maxId = (terms: readonly Term[]) => Math.max(-1, ...termVariables(...terms).map((v) => v.id))

/**
 * A substitution θ with Cθ ⊆ D (head onto head, each body literal of C onto some body literal of D), or null. D's
 * variables are held fixed (renamed apart first). Backtracking search over the choices for each literal.
 */
export function thetaSubsumption(c: HornClause, d: HornClause): Substitution | null {
  const offset = maxId([c.head, ...c.body]) + 1
  const renamed = renameVariables(compound('$clause', [d.head, ...d.body]), offset)
  const dHead = renamed.kind === 'compound' ? renamed.args[0] : d.head
  const dBody = renamed.kind === 'compound' ? renamed.args.slice(1) : []
  const start = matchTerm(c.head, dHead)
  if (start === null) return null
  // Most constrained literals first: fewer candidates in D.
  const order = [...c.body].sort(
    (x, y) =>
      dBody.filter((z) => predicateOf(z) === predicateOf(x)).length -
      dBody.filter((z) => predicateOf(z) === predicateOf(y)).length,
  )
  const search = (i: number, theta: Substitution): Substitution | null => {
    if (i === order.length) return theta
    for (const z of dBody) {
      if (predicateOf(z) !== predicateOf(order[i])) continue
      const next = matchTerm(order[i], z, theta)
      if (next !== null) {
        const found = search(i + 1, next)
        if (found !== null) return found
      }
    }
    return null
  }
  return search(0, start)
}

/** True when C θ-subsumes D (C is at least as general as D). */
export function thetaSubsumes(c: HornClause, d: HornClause): boolean {
  return thetaSubsumption(c, d) !== null
}

/**
 * The clause with redundant body literals removed (Plotkin's reduction): a literal is dropped when the clause with it
 * θ-subsumes the clause without it (the converse always holds), so both are equivalent. Greedy, last literal first.
 */
export function reduceClause(c: HornClause): HornClause {
  let body = [...c.body]
  for (let i = body.length - 1; i >= 0; i--) {
    const without = { head: c.head, body: body.filter((_, j) => j !== i) }
    if (thetaSubsumes({ head: c.head, body }, without)) body = without.body
  }
  return { head: c.head, body }
}
