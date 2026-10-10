/**
 * Generality of terms and clauses (Plotkin 1970): $\theta$-subsumption and the least general generalisation.
 *
 * A clause $C$ $\theta$-subsumes $D$ when some substitution $\theta$ maps every literal of $C$ onto a literal of $D$
 * ($C\theta \subseteq D$): $C$ is then at least as general as $D$, and $C \models D$. The least general generalisation
 * (lgg) of two terms replaces each pair of differing subterms by a variable, the same pair always by the same
 * variable: $\operatorname{lgg}(f(a, g(a)), f(b, g(b))) = f(X, g(X))$. The lgg of two clauses pairs every two
 * literals of the same predicate and sign; it $\theta$-subsumes both clauses, and every clause that $\theta$-subsumes
 * both $\theta$-subsumes it.
 *
 * Clauses are Horn clauses, a head and a body; terms are those of `aifn-compute/logic/terms`. The examples build
 * terms as plain objects, with helpers `a` (an atom), `v` (a variable), `f` (a compound) and `text` (to print one).
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

/**
 * The name of the lgg variable numbered `i`: A to Z, then `V26`, `V27`, ….
 *
 * @param i The variable's number, from 0.
 * @returns Its name.
 */
const defaultName = (i: number) => (i < 26 ? String.fromCharCode(65 + i) : `V${i}`)

/**
 * The lgg of two terms with a shared table of pairs: identical terms other than variables are kept, compounds with
 * the same functor and arity are generalised argument by argument, and any other pair becomes the table's variable
 * for it.
 *
 * @param a The first term.
 * @param b The second term.
 * @param pairs The table of pairs met so far, extended with the new ones.
 * @returns The generalisation.
 */
function lggWith(a: Term, b: Term, pairs: Pairs): Term {
  if (termsEqual(a, b) && a.kind !== 'var') return a
  if (a.kind === 'compound' && b.kind === 'compound' && a.functor === b.functor && a.args.length === b.args.length)
    return compound(
      a.functor,
      a.args.map((x, i) => lggWith(x, b.args[i], pairs)),
    )
  return pairs.get(a, b)
}

/**
 * The least general generalisation of two terms (Plotkin; Reynolds 1970). Its variables are named A, B, …, with ids
 * 0, 1, … in order of creation.
 *
 * @param a The first term. Its variables are treated like constants: a pair of different variables becomes a variable
 *   of the result.
 * @param b The second term.
 * @returns The most specific term that both `a` and `b` are instances of.
 *
 * @example The same pair of constants becomes the same variable
 * const a = (name) => ({ kind: 'atom', name })
 * const f = (functor, ...args) => ({ kind: 'compound', functor, args })
 * const text = (t) => (t.args ? `${t.functor}(${t.args.map(text).join(', ')})` : t.name)
 * print(text(termLgg(f('f', a('a'), f('g', a('a'))), f('f', a('b'), f('g', a('b'))))))
 * print(text(termLgg(f('f', a('a'), a('b')), f('f', a('b'), a('a')))))
 */
export function termLgg(a: Term, b: Term): Term {
  return lggWith(a, b, new Pairs(defaultName))
}

/**
 * `name/arity` of an atom or compound literal, or `''` for a variable or number.
 *
 * @param t The literal.
 * @returns Its predicate indicator.
 */
const predicateOf = (t: Term) =>
  t.kind === 'compound' ? `${t.functor}/${t.args.length}` : t.kind === 'atom' ? `${t.name}/0` : ''

/**
 * The least general generalisation of two clauses: the lgg of the heads, and the lggs of every pair of body literals
 * of the same predicate, with one table of variables throughout, so a pair of constants is one variable everywhere.
 * Not reduced (it may hold literals that others $\theta$-subsume; `reduceClause` removes them). Returns null when the
 * heads have different predicates.
 *
 * @param c The first clause. Its variables are generalised like constants.
 * @param d The second clause.
 * @returns The lgg, with repeated body literals dropped, or null.
 *
 * @example Two instances of the same pattern
 * const a = (name) => ({ kind: 'atom', name })
 * const f = (functor, ...args) => ({ kind: 'compound', functor, args })
 * const text = (t) => (t.args ? `${t.functor}(${t.args.map(text).join(', ')})` : t.name)
 * const c = { head: f('p', a('a')), body: [f('q', a('a'), a('b')), f('r', a('b'))] }
 * const d = { head: f('p', a('c')), body: [f('q', a('c'), a('d')), f('r', a('d'))] }
 * const g = clauseLgg(c, d)
 * print(text(g.head), ':-', g.body.map(text).join(', '))
 * print('different heads:', clauseLgg(c, { head: f('s', a('a')), body: [] }))
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

/**
 * The largest variable id in the terms, or $-1$.
 *
 * @param terms The terms.
 * @returns The largest id.
 */
const maxId = (terms: readonly Term[]) => Math.max(-1, ...termVariables(...terms).map((v) => v.id))

/**
 * A substitution $\theta$ with $C\theta \subseteq D$ (head onto head, each body literal of $C$ onto some body literal
 * of $D$), or null. $D$'s variables are held fixed (renamed apart first). Backtracking search over the choices for
 * each literal, the literals with fewest candidates first.
 *
 * @param c The clause $C$, the more general one, whose variables are bound.
 * @param d The clause $D$. Its variables are renamed to ids above those of $C$ (names kept), and act as constants.
 * @returns $\theta$, binding the variables of $C$ to terms of the renamed $D$, or null when $C$ does not
 *   $\theta$-subsume $D$.
 *
 * @example A general clause onto a specific one
 * const a = (name) => ({ kind: 'atom', name })
 * const v = (name, id) => ({ kind: 'var', name, id })
 * const f = (functor, ...args) => ({ kind: 'compound', functor, args })
 * const text = (t) => (t.args ? `${t.functor}(${t.args.map(text).join(', ')})` : t.name)
 * const general = { head: f('p', v('X', 0)), body: [f('q', v('X', 0), v('Y', 1))] }
 * const specific = { head: f('p', a('a')), body: [f('q', a('a'), a('b')), f('r', a('b'))] }
 * print(thetaSubsumption(general, specific).map((b) => `${b.variable.name} = ${text(b.value)}`).join(', '))
 * print('converse:', thetaSubsumption(specific, general))
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

/**
 * True when $C$ $\theta$-subsumes $D$ ($C$ is at least as general as $D$).
 *
 * @param c The clause $C$.
 * @param d The clause $D$.
 * @returns Whether `thetaSubsumption` finds a substitution.
 *
 * @example Subsumption goes one way
 * const a = (name) => ({ kind: 'atom', name })
 * const v = (name, id) => ({ kind: 'var', name, id })
 * const f = (functor, ...args) => ({ kind: 'compound', functor, args })
 * const general = { head: f('p', v('X', 0)), body: [f('q', v('X', 0), v('Y', 1))] }
 * const specific = { head: f('p', a('a')), body: [f('q', a('a'), a('b')), f('r', a('b'))] }
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
 * const v = (name, id) => ({ kind: 'var', name, id })
 * const f = (functor, ...args) => ({ kind: 'compound', functor, args })
 * const text = (t) => (t.args ? `${t.functor}(${t.args.map(text).join(', ')})` : t.name)
 * const c = { head: f('p', v('X', 0)), body: [f('q', v('X', 0), v('Y', 1)), f('q', v('X', 0), v('Z', 2))] }
 * const r = reduceClause(c)
 * print(text(r.head), ':-', r.body.map(text).join(', '))
 */
export function reduceClause(c: HornClause): HornClause {
  let body = [...c.body]
  for (let i = body.length - 1; i >= 0; i--) {
    const without = { head: c.head, body: body.filter((_, j) => j !== i) }
    if (thetaSubsumes({ head: c.head, body }, without)) body = without.body
  }
  return { head: c.head, body }
}
