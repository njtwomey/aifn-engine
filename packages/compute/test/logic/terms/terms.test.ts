/** Terms, unification laws, matching, reading and printing (aifn-compute/logic/terms). */
import { describe, expect, it } from 'vitest'
import {
  applySubstitution,
  atom,
  clauseToString,
  composeSubstitutions,
  compound,
  listItems,
  listTerm,
  matchTerm,
  numeral,
  parseProgram,
  parseQuery,
  parseTerm,
  PrologSyntaxError,
  termsEqual,
  termToString,
  termVariables,
  unify,
  variable,
  type Term,
} from 'aifn-compute/logic/terms'

const t = (s: string) => parseTerm(s)
/** Two terms read together, so they share variables by name (`a = b`). */
const pair = (s: string): [Term, Term] => {
  const e = t(s)
  if (e.kind !== 'compound') throw new Error('expected a = b')
  return [e.args[0], e.args[1]]
}
const show = (x: Term) => termToString(x)

/** Random terms over a small signature, for the laws. */
function randomTerms(seed: number, n: number): Term[] {
  let s = seed
  const rand = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648
  const make = (depth: number): Term => {
    const r = rand()
    if (depth <= 0 || r < 0.35)
      return r < 0.2
        ? variable('XYZW'[Math.floor(rand() * 4)], Math.floor(rand() * 4))
        : atom('ab'[Math.floor(rand() * 2)])
    const f = rand() < 0.5 ? 'f' : 'g'
    return compound(f, f === 'f' ? [make(depth - 1), make(depth - 1)] : [make(depth - 1)])
  }
  return Array.from({ length: n }, () => make(3))
}

describe('unify', () => {
  it('finds the most general unifier of classic pairs', () => {
    const [a, b] = pair('f(X, g(Y), Y) = f(a, g(b), Z)')
    const theta = unify(a, b)!
    expect(theta).not.toBeNull()
    expect(show(applySubstitution(a, theta))).toBe('f(a, g(b), b)')
    expect(show(applySubstitution(b, theta))).toBe('f(a, g(b), b)')
    expect(unify(t('f(X, X)'), t('f(a, b)'))).toBeNull()
    expect(unify(t('f(a)'), t('g(a)'))).toBeNull()
    expect(unify(t('f(a, b)'), t('f(a)'))).toBeNull()
    expect(unify(numeral(1), numeral(1))).toEqual([])
    expect(unify(t('[H|T]'), t('[1, 2, 3]'))!.map((b) => `${b.variable.name}=${show(b.value)}`)).toEqual([
      'H=1',
      'T=[2, 3]',
    ])
  })

  it('applies the occurs check only when asked', () => {
    expect(unify(...pair('X = f(X)'), { occursCheck: true })).toBeNull()
    const cyclic = unify(...pair('X = f(X)'))
    expect(cyclic).not.toBeNull()
    expect(show(cyclic![0].value)).toBe('f(X)')
  })

  it('obeys the laws: a unifier, idempotent, symmetric up to renaming, most general', () => {
    const terms = randomTerms(7, 160)
    let unified = 0
    for (let i = 0; i + 1 < terms.length; i += 2) {
      const [a, b] = [terms[i], terms[i + 1]]
      const ab = unify(a, b, { occursCheck: true })
      const ba = unify(b, a, { occursCheck: true })
      expect(ab === null).toBe(ba === null)
      if (ab === null || ba === null) continue
      unified++
      const aa = applySubstitution(a, ab)
      // A unifier.
      expect(termsEqual(aa, applySubstitution(b, ab))).toBe(true)
      // Idempotent: θθ = θ.
      expect(termsEqual(applySubstitution(aa, ab), aa)).toBe(true)
      // Symmetric: both results are variants of each other (each matches the other).
      const other = applySubstitution(a, ba)
      expect(matchTerm(aa, other)).not.toBeNull()
      expect(matchTerm(other, aa)).not.toBeNull()
      // Most general: the common instance a ground unifier gives is an instance of aθ.
      const ground = termVariables(aa).map((v) => ({ variable: v, value: atom('c') }))
      const sigma = composeSubstitutions(ab, ground)
      expect(matchTerm(aa, applySubstitution(a, sigma))).not.toBeNull()
    }
    expect(unified).toBeGreaterThan(10)
  })
})

describe('substitutions', () => {
  it('composes as sequential application', () => {
    const [h, fx, fz, z, a] = (
      t('h(X, Y, Z, W) = [f(X, Y), f(g(Z), Z), Z, a]') as unknown as { args: Term[] }
    ).args.flatMap((x, i) => (i === 0 ? [x] : listItems(x).items))
    const theta = unify(fx, fz)!
    const sigma = unify(z, a)!
    const term = h
    expect(show(applySubstitution(term, composeSubstitutions(theta, sigma)))).toBe(
      show(applySubstitution(applySubstitution(term, theta), sigma)),
    )
  })
  it('matches one way only', () => {
    expect(matchTerm(t('f(X, X)'), t('f(a, a)'))).not.toBeNull()
    expect(matchTerm(t('f(X, X)'), t('f(a, b)'))).toBeNull()
    expect(matchTerm(t('f(a)'), t('f(X)'))).toBeNull()
  })
})

describe('reading and printing', () => {
  it('reads operators with standard priorities', () => {
    expect(show(t('X is 1 + 2 * 3 - 4'))).toBe('X is 1 + 2 * 3 - 4')
    const e = t('1 - 2 - 3')
    expect(e.kind === 'compound' && e.functor === '-' && show(e.args[0])).toBe('1 - 2')
    expect(show(t('(a :- b, c ; d)'))).toBe('a :- b, c ; d')
    expect(show(t('\\+ member(X, [a, b|T])'))).toBe('\\+ member(X, [a, b|T])')
    expect(show(t('f(-1, - 2, -(a))'))).toBe('f(-1, -2, -a)')
    expect(show(t("'hello world'"))).toBe("'hello world'")
    expect(show(t('X = (a, b)'))).toBe('X = (a, b)')
  })
  it('reads lists, strings of clauses and queries', () => {
    const { items, tail } = listItems(t('[1, 2 | T]'))
    expect(items.map(show)).toEqual(['1', '2'])
    expect(show(tail)).toBe('T')
    expect(show(listTerm([atom('a')]))).toBe('[a]')
    const { clauses, queries } = parseProgram(`
      % a comment
      parent(tom, bob).   /* block */
      grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
      ?- grandparent(tom, W).
    `)
    expect(clauses.map((c) => clauseToString(c))).toEqual([
      'parent(tom, bob).',
      'grandparent(X, Z) :- parent(X, Y), parent(Y, Z).',
    ])
    expect(clauses[1].variableNames).toEqual(['X', 'Z', 'Y'])
    expect(queries[0].variableNames).toEqual(['W'])
    expect(parseQuery('?- member(X, [a]).').goals.length).toBe(1)
    expect(parseQuery('a(X), b(X, _), c(_)').variableNames).toEqual(['X', '_', '_'])
  })
  it('reports syntax errors with line and column', () => {
    try {
      parseProgram('a.\nb(X :- c.')
      expect.unreachable()
    } catch (e) {
      expect(e).toBeInstanceOf(PrologSyntaxError)
      expect((e as PrologSyntaxError).line).toBe(2)
    }
    expect(() => parseProgram('a :- b')).toThrow(PrologSyntaxError)
  })
  it('parseQuery reports the line and column of the source text', () => {
    expect(parseQuery('\n\n  ?- p(X).').line).toBe(3)
    expect(parseQuery('p(X)').line).toBe(1)
    try {
      parseQuery('\n?- p(X ; .')
      expect.unreachable()
    } catch (e) {
      expect(e).toBeInstanceOf(PrologSyntaxError)
      expect((e as PrologSyntaxError).line).toBe(2)
      expect((e as PrologSyntaxError).column).toBe(8)
    }
  })
})
