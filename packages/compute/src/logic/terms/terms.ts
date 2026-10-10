/**
 * First-order terms (the data of logic programs) and unification.
 *
 * A term is an atom (`mary`, `[]`), a number, a variable (`X`, identified by an integer id; the name is for printing)
 * or a compound $f(t_1, \dots, t_n)$. Lists are compounds of the functor `.` with two arguments, ending in the atom
 * `[]`: `[a, b]` is `.(a, .(b, []))`. Terms are plain data, so they survive a structured clone. Variables of
 * different terms are the same variable when their ids are equal, so terms from different sources are renamed apart
 * (`renameVariables`) before they are unified.
 *
 * A substitution is a finite set of bindings $\{X_1 \mapsto t_1, \dots\}$. `unify` returns the most general unifier of
 * two terms (Robinson, 1965) in solved form: no bound variable occurs in any bound term, so applying it once gives the
 * final result (it is idempotent, $\theta\theta = \theta$). The one exception is a cyclic binding such as
 * $X \mapsto f(X)$, which only unification without the occurs check makes.
 */
import { DomainError } from 'aifn-compute/foundation/errors'

/** A constant: a name, such as `mary`, `[]` or `+`. */
export interface Atom {
  /** Always `'atom'`. */
  readonly kind: 'atom'
  /** The atom's name, unquoted. */
  readonly name: string
}
/** A number (integers and floats share one kind; integer-valued numbers print as integers). */
export interface Numeral {
  /** Always `'number'`. */
  readonly kind: 'number'
  /** The number's value. */
  readonly value: number
}
/** A logic variable: equal ids are the same variable. `name` is how it prints (`X`, `_G3`). */
export interface Variable {
  /** Always `'var'`. */
  readonly kind: 'var'
  /** How the variable prints. */
  readonly name: string
  /** The variable's identity: substitutions and unification key variables by it. */
  readonly id: number
}
/** A compound term `functor(args…)`. */
export interface Compound {
  /** Always `'compound'`. */
  readonly kind: 'compound'
  /** The name of the function symbol (`.` for a list cell). */
  readonly functor: string
  /** The arguments, at least one (a functor with none is an atom). */
  readonly args: readonly Term[]
}
/** A first-order term: an atom, a number, a variable or a compound, told apart by `kind`. */
export type Term = Atom | Numeral | Variable | Compound

/** One binding $X \mapsto t$ of a substitution. */
export interface Binding {
  /** The variable $X$ bound. */
  readonly variable: Variable
  /** The term $t$ it is bound to. */
  readonly value: Term
}
/** A substitution as its bindings, in the order the variables were bound. */
export type Substitution = readonly Binding[]

/**
 * The atom named `name`.
 *
 * @param name The atom's name, such as `mary`, `[]` or `+`: any string; `termToString` quotes it when it needs to be.
 * @returns The atom.
 *
 * @example Atoms, quoted when printed only where Prolog needs it
 * print(termToString(atom('mary')), termToString(atom('New York')), termToString(atom('[]')))
 */
export const atom = (name: string): Atom => ({ kind: 'atom', name })
/**
 * The number `value` as a term.
 *
 * @param value Any number: integers and floats share one kind, and an integer-valued number prints as an integer.
 * @returns The number term.
 *
 * @example Numbers print as Prolog writes them
 * print(termToString(numeral(3)), termToString(numeral(-2.5)), termToString(numeral(0.1 + 0.2)))
 */
export const numeral = (value: number): Numeral => ({ kind: 'number', value })
/**
 * A logic variable. Two variables are the same variable when their ids are equal, whatever their names.
 *
 * @param name How the variable prints (`X`, `_G3`).
 * @param id Its identity: the integer that unification, substitutions and `termsEqual` compare.
 * @returns The variable.
 *
 * @example The id, not the name, is the variable
 * print('same variable:', termsEqual(variable('X', 0), variable('Y', 0)))
 * print('same name only:', termsEqual(variable('X', 0), variable('X', 1)))
 */
export const variable = (name: string, id: number): Variable => ({ kind: 'var', name, id })
/**
 * The compound term `functor(args…)`, or the atom `functor` when there are no arguments.
 *
 * @param functor The name of the function symbol.
 * @param args The arguments, in order; with none, the result is an atom.
 * @returns The compound term (an atom for no arguments).
 *
 * @example Build terms and print them
 * print(termToString(compound('f', [atom('a'), variable('X', 0)])))
 * print(termToString(compound('+', [numeral(1), numeral(2)])))
 * print('no arguments:', compound('g', []).kind)
 */
export const compound = (functor: string, args: readonly Term[]): Term =>
  args.length === 0 ? atom(functor) : { kind: 'compound', functor, args }

/** The empty list `[]`. */
export const NIL: Atom = atom('[]')

/**
 * The list `[items… | tail]`: a chain of `.` cells, one per item, ending in `tail`.
 *
 * @param items The elements, first to last.
 * @param tail What follows the last item: `[]` (the default) for a proper list, a variable for a partial list. With no
 *   items the result is `tail` itself.
 * @returns The list term.
 *
 * @example A proper list and a partial one
 * print(termToString(listTerm([atom('a'), atom('b')])))
 * print(termToString(listTerm([atom('a'), atom('b')], variable('T', 0))))
 */
export function listTerm(items: readonly Term[], tail: Term = NIL): Term {
  let out = tail
  for (let i = items.length - 1; i >= 0; i--) out = { kind: 'compound', functor: '.', args: [items[i], out] }
  return out
}

/**
 * The items of a list term and its tail (`[]` for a proper list, a variable for a partial one).
 *
 * @param term Any term, read as a chain of `.` cells with two arguments; a term that is not one has no items and is
 *   its own tail.
 * @returns `items`, the elements in order, and `tail`, the term after the last cell.
 *
 * @example Take a partial list apart
 * const { items, tail } = listItems(parseTerm('[a, b | T]'))
 * print('items:', items.map((t) => termToString(t)).join(', '))
 * print('tail:', termToString(tail))
 */
export function listItems(term: Term): { items: Term[]; tail: Term } {
  const items: Term[] = []
  let t = term
  while (t.kind === 'compound' && t.functor === '.' && t.args.length === 2) {
    items.push(t.args[0])
    t = t.args[1]
  }
  return { items, tail: t }
}

/**
 * `name/arity` of an atom or compound: the key of a predicate. Throws `DomainError` for a variable or a number, which
 * cannot be called.
 *
 * @param term The goal or clause head, an atom (arity 0) or a compound.
 * @returns The predicate indicator, such as `parent/2`.
 *
 * @example Predicate indicators
 * print(indicator(parseTerm('parent(tom, bob)')), indicator(atom('true')))
 */
export function indicator(term: Term): string {
  if (term.kind === 'atom') return `${term.name}/0`
  if (term.kind === 'compound') return `${term.functor}/${term.args.length}`
  throw new DomainError('indicator', `indicator: ${termToString(term)} is not callable`)
}

/**
 * The distinct variables of the terms, in order of first occurrence (left to right, depth first).
 *
 * @param terms The terms to scan, in order; a variable is told apart from another by its id.
 * @returns Each variable once, as first met.
 *
 * @example The variables of a term
 * print(termVariables(parseTerm('f(X, g(Y, X), Z)')).map((v) => v.name).join(', '))
 */
export function termVariables(...terms: readonly Term[]): Variable[] {
  const seen = new Set<number>()
  const out: Variable[] = []
  const walk = (t: Term) => {
    if (t.kind === 'var') {
      if (!seen.has(t.id)) {
        seen.add(t.id)
        out.push(t)
      }
    } else if (t.kind === 'compound') t.args.forEach(walk)
  }
  terms.forEach(walk)
  return out
}

/**
 * True when the term has no variables.
 *
 * @param term The term to test.
 * @returns Whether no variable occurs in `term`.
 *
 * @example Ground and non-ground terms
 * print(isGround(parseTerm('f(a, [1, 2])')), isGround(parseTerm('f(a, X)')))
 */
export function isGround(term: Term): boolean {
  if (term.kind === 'var') return false
  return term.kind !== 'compound' || term.args.every(isGround)
}

/**
 * Structural identity (Prolog's `==`): the same term, variables compared by id. No variable is bound.
 *
 * @param a The first term.
 * @param b The second term.
 * @returns Whether `a` and `b` are identical.
 *
 * @example Identity is not unifiability
 * print('same term:', termsEqual(parseTerm('f(a, [b])'), parseTerm('f(a, [b])')))
 * print('variables differ:', termsEqual(variable('X', 0), variable('X', 1)))
 */
export function termsEqual(a: Term, b: Term): boolean {
  if (a === b) return true
  if (a.kind !== b.kind) return false
  switch (a.kind) {
    case 'atom':
      return a.name === (b as Atom).name
    case 'number':
      return a.value === (b as Numeral).value
    case 'var':
      return a.id === (b as Variable).id
    case 'compound': {
      const c = b as Compound
      return (
        a.functor === c.functor && a.args.length === c.args.length && a.args.every((x, i) => termsEqual(x, c.args[i]))
      )
    }
  }
}

/**
 * The term with each variable replaced by `map` of it when that is defined; shares unchanged subterms.
 *
 * @param term The term to rewrite; not modified.
 * @param map Called on every variable occurrence: a term replaces the variable, `undefined` keeps it. Its result is
 *   not rewritten again.
 * @returns The rewritten term, `term` itself when nothing changed.
 */
function mapVariables(term: Term, map: (v: Variable) => Term | undefined): Term {
  switch (term.kind) {
    case 'var':
      return map(term) ?? term
    case 'compound': {
      let changed = false
      const args = term.args.map((a) => {
        const b = mapVariables(a, map)
        if (b !== a) changed = true
        return b
      })
      return changed ? { kind: 'compound', functor: term.functor, args } : term
    }
    default:
      return term
  }
}

/**
 * $t\theta$: the term with every bound variable replaced by its value (one pass; a solved-form $\theta$ needs no
 * more).
 *
 * @param term The term $t$; not modified.
 * @param theta The substitution $\theta$. A value is put in place as it is, without substituting into it again.
 * @returns The term $t\theta$ (`term` itself when $\theta$ binds none of its variables).
 *
 * @example Apply a unifier
 * const { goals: [t, u] } = parseQuery('f(X, g(Y)), f(a, g(b))')
 * const theta = unify(t, u)
 * print('theta =', substitutionToString(theta))
 * print('t theta =', termToString(applySubstitution(t, theta)))
 */
export function applySubstitution(term: Term, theta: Substitution): Term {
  if (theta.length === 0) return term
  const map = new Map(theta.map((b) => [b.variable.id, b.value]))
  return mapVariables(term, (v) => map.get(v.id))
}

/**
 * The composition $\theta\sigma$: applying it equals applying $\theta$ then $\sigma$. Bindings of $\theta$ have
 * $\sigma$ applied to their values; bindings of $\sigma$ for variables $\theta$ does not bind are added; trivial
 * bindings $X \mapsto X$ are dropped.
 *
 * @param theta The substitution $\theta$, applied first.
 * @param sigma The substitution $\sigma$, applied second.
 * @returns The composition $\theta\sigma$: the bindings of $\theta$ first, then the new ones of $\sigma$.
 *
 * @example Compose two substitutions
 * const X = variable('X', 0)
 * const Y = variable('Y', 1)
 * const theta = [{ variable: X, value: compound('f', [Y]) }]
 * const sigma = [{ variable: Y, value: atom('a') }]
 * print(substitutionToString(composeSubstitutions(theta, sigma)))
 */
export function composeSubstitutions(theta: Substitution, sigma: Substitution): Substitution {
  const bound = new Set(theta.map((b) => b.variable.id))
  const out: Binding[] = []
  for (const b of theta) {
    const value = applySubstitution(b.value, sigma)
    if (!(value.kind === 'var' && value.id === b.variable.id)) out.push({ variable: b.variable, value })
  }
  for (const b of sigma) if (!bound.has(b.variable.id)) out.push(b)
  return out
}

/** Options of `unify`. */
export interface UnifyOptions {
  /** Refuse to bind a variable $X$ to a term containing $X$ (default false, as in Prolog's `=/2`). */
  occursCheck?: boolean
}

/**
 * The most general unifier of `a` and `b` (Robinson 1965, by Martelli and Montanari's rules), or null when they do not
 * unify. With the occurs check (`unify_with_occurs_check/2`) the result is idempotent: no variable it binds occurs in
 * any value, so $\theta\theta = \theta$. Without it (Prolog's `=/2`, the default), `X = f(X)` succeeds with the
 * binding $X \mapsto f(X)$, which stands for the infinite term $f(f(\dots))$; such a binding is returned unfolded once,
 * and only it breaks idempotence.
 *
 * @param a The first term; not modified.
 * @param b The second term. Its variables share one space of ids with those of `a`: rename the terms apart first
 *   unless a shared variable is meant.
 * @param options Whether to apply the occurs check (see `UnifyOptions`).
 * @returns The unifier as bindings in the order the variables were bound (empty when the terms are already
 *   identical), or null when the terms do not unify.
 *
 * @example The most general unifier of two terms
 * const { goals: [s, t] } = parseQuery('f(X, g(b)), f(a, g(Y))')
 * print('mgu =', substitutionToString(unify(s, t)))
 * print('f(a) and g(a):', unify(parseTerm('f(a)'), parseTerm('g(a)')))
 *
 * @example The occurs check
 * const { goals: [x, fx] } = parseQuery('X, f(X)')
 * print('without:', substitutionToString(unify(x, fx)))
 * print('with:', unify(x, fx, { occursCheck: true }))
 */
export function unify(a: Term, b: Term, options: UnifyOptions = {}): Substitution | null {
  const check = options.occursCheck ?? false
  // Triangular bindings while unifying; resolved to solved form at the end.
  const bound = new Map<number, Term>()
  const order: Variable[] = []
  const walk = (t: Term): Term => {
    while (t.kind === 'var') {
      const v = bound.get(t.id)
      if (v === undefined) return t
      t = v
    }
    return t
  }
  const occurs = (id: number, t: Term): boolean => {
    const s = walk(t)
    if (s.kind === 'var') return s.id === id
    return s.kind === 'compound' && s.args.some((x) => occurs(id, x))
  }
  const stack: [Term, Term][] = [[a, b]]
  while (stack.length) {
    const [x0, y0] = stack.pop()!
    const x = walk(x0)
    const y = walk(y0)
    if (x === y) continue
    if (x.kind === 'var' || y.kind === 'var') {
      if (x.kind === 'var' && y.kind === 'var' && x.id === y.id) continue
      const [v, t] = x.kind === 'var' ? [x, y] : [y as Variable, x]
      if (check && occurs(v.id, t)) return null
      bound.set(v.id, t)
      order.push(v)
      continue
    }
    if (x.kind !== y.kind) return null
    if (x.kind === 'atom') {
      if (x.name !== (y as Atom).name) return null
    } else if (x.kind === 'number') {
      if (x.value !== (y as Numeral).value) return null
    } else {
      const c = y as Compound
      if (x.functor !== c.functor || x.args.length !== c.args.length) return null
      for (let i = x.args.length - 1; i >= 0; i--) stack.push([x.args[i], c.args[i]])
    }
  }
  // Resolve the triangular bindings; a variable met again inside its own value (a cycle, only without the occurs
  // check) is left as the variable.
  const open = new Set<number>()
  const resolve = (t: Term): Term =>
    mapVariables(t, (v) => {
      if (!bound.has(v.id) || open.has(v.id)) return undefined
      open.add(v.id)
      const out = resolve(bound.get(v.id)!)
      open.delete(v.id)
      return out
    })
  return order.map((v) => {
    open.add(v.id)
    const value = resolve(bound.get(v.id)!)
    open.delete(v.id)
    return { variable: v, value }
  })
}

/**
 * One-way matching: a substitution $\theta$ binding only variables of `pattern` with $p\theta = t$ for `pattern` $p$
 * and `term` $t$, or null. Variables of `term` are treated as constants. The test inside $\theta$-subsumption.
 *
 * @param pattern The pattern $p$, whose variables may be bound. Its variable ids must differ from those of `term`.
 * @param term The term $t$ to match; a variable in it matches only a pattern variable.
 * @param theta Bindings of pattern variables already made, which every new binding must agree with (default none).
 *   Not modified.
 * @returns `theta` followed by the new bindings, or null when `pattern` does not match `term`.
 *
 * @example Matching goes one way
 * const { goals: [p, t] } = parseQuery('f(X, Y, X), f(a, g(Z), a)')
 * print('pattern onto term:', substitutionToString(matchTerm(p, t)))
 * print('term onto pattern:', matchTerm(t, p))
 */
export function matchTerm(pattern: Term, term: Term, theta: Substitution = []): Substitution | null {
  const bound = new Map(theta.map((b) => [b.variable.id, b.value]))
  const out = [...theta]
  const go = (p: Term, t: Term): boolean => {
    if (p.kind === 'var') {
      const v = bound.get(p.id)
      if (v !== undefined) return termsEqual(v, t)
      bound.set(p.id, t)
      out.push({ variable: p, value: t })
      return true
    }
    if (p.kind !== t.kind) return false
    if (p.kind === 'atom') return p.name === (t as Atom).name
    if (p.kind === 'number') return p.value === (t as Numeral).value
    const c = t as Compound
    return p.functor === c.functor && p.args.length === c.args.length && p.args.every((x, i) => go(x, c.args[i]))
  }
  return go(pattern, term) ? out : null
}

/**
 * The term with its variables renamed to fresh ids `offset`, `offset + 1`, … in order of first occurrence; names are
 * kept.
 *
 * @param term The term to rename; not modified.
 * @param offset The first new id; the caller chooses it above every id in use (a counter of fresh ids).
 * @returns The renamed term.
 *
 * @example Rename apart from ids below 10
 * const t = renameVariables(parseTerm('f(X, Y, X)'), 10)
 * print(termVariables(t).map((v) => `${v.name} has id ${v.id}`).join(', '))
 */
export function renameVariables(term: Term, offset: number): Term {
  const ids = new Map<number, number>()
  return mapVariables(term, (v) => {
    if (!ids.has(v.id)) ids.set(v.id, offset + ids.size)
    return variable(v.name, ids.get(v.id)!)
  })
}

// ── Printing ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Operators printed infix, with their priority and type (for brackets). */
export const INFIX: Readonly<Record<string, { priority: number; type: 'xfx' | 'xfy' | 'yfx' }>> = {
  ':-': { priority: 1200, type: 'xfx' },
  ';': { priority: 1100, type: 'xfy' },
  '->': { priority: 1050, type: 'xfy' },
  ',': { priority: 1000, type: 'xfy' },
  '=': { priority: 700, type: 'xfx' },
  '\\=': { priority: 700, type: 'xfx' },
  '==': { priority: 700, type: 'xfx' },
  '\\==': { priority: 700, type: 'xfx' },
  is: { priority: 700, type: 'xfx' },
  '=:=': { priority: 700, type: 'xfx' },
  '=\\=': { priority: 700, type: 'xfx' },
  '<': { priority: 700, type: 'xfx' },
  '>': { priority: 700, type: 'xfx' },
  '=<': { priority: 700, type: 'xfx' },
  '>=': { priority: 700, type: 'xfx' },
  '+': { priority: 500, type: 'yfx' },
  '-': { priority: 500, type: 'yfx' },
  '*': { priority: 400, type: 'yfx' },
  '/': { priority: 400, type: 'yfx' },
  '//': { priority: 400, type: 'yfx' },
  mod: { priority: 400, type: 'yfx' },
  rem: { priority: 400, type: 'yfx' },
  '**': { priority: 200, type: 'xfx' },
  '^': { priority: 200, type: 'xfy' },
}
/** Operators printed prefix, with their priority and type (for brackets). */
export const PREFIX: Readonly<Record<string, { priority: number; type: 'fy' | 'fx' }>> = {
  ':-': { priority: 1200, type: 'fx' },
  '?-': { priority: 1200, type: 'fx' },
  '\\+': { priority: 900, type: 'fy' },
  '-': { priority: 200, type: 'fy' },
}

/** Atoms that print without quotes although they are neither a name nor a run of symbol characters. */
const SOLO = new Set(['[]', '!', ';', ',', '{}'])
/**
 * An atom's name as Prolog text: bare when it is a name starting with a lowercase letter, a solo atom (`[]`, `!`,
 * `;`, `,`, `{}`) or a run of symbol characters; otherwise in single quotes, with backslashes and quotes escaped.
 *
 * @param name The atom's name.
 * @returns The name, quoted when needed.
 */
const quoteAtom = (name: string): string => {
  if (/^[a-z][A-Za-z0-9_]*$/.test(name) || SOLO.has(name) || /^[+\-*/\\^<>=~:.?@#&$]+$/.test(name)) return name
  return `'${name.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`
}
/**
 * A number as Prolog text: an integer as it is, another finite number to 15 significant digits (so `0.1 + 0.2` prints
 * as `0.3`), and `Infinity` and `NaN` as JavaScript prints them.
 *
 * @param x The number.
 * @returns Its text.
 */
const formatNumber = (x: number): string =>
  Number.isInteger(x) ? String(x) : Number.isFinite(x) ? String(Number(x.toPrecision(15))) : String(x)

/** Options of `termToString`. */
export interface PrintOptions {
  /** How a variable prints; default its name. */
  variableName?: (v: Variable) => string
  /** Terms nested deeper than this print as `…` (default 64). */
  maxDepth?: number
}

/**
 * A term in Prolog syntax: operators infix or prefix with standard priorities (brackets only where needed), lists in
 * brackets, atoms quoted where needed, and a negative number bracketed when it is an operand.
 *
 * @param term The term to print.
 * @param options How variables print and how deep to go (see `PrintOptions`).
 * @returns The text.
 *
 * @example Operators, brackets and lists
 * const t = parseTerm('X is (1 + 2) * 3, L = [a, b | T]')
 * print(termToString(t))
 * print(termToString(t, { variableName: (v) => `_G${v.id}` }))
 */
export function termToString(term: Term, options: PrintOptions = {}): string {
  const name = options.variableName ?? ((v: Variable) => v.name)
  const maxDepth = options.maxDepth ?? 64
  const go = (t: Term, max: number, depth: number): string => {
    if (depth > maxDepth) return '…'
    switch (t.kind) {
      case 'atom':
        return quoteAtom(t.name)
      case 'number':
        return t.value < 0 && max < 200 ? `(${formatNumber(t.value)})` : formatNumber(t.value)
      case 'var':
        return name(t)
      case 'compound': {
        if (t.functor === '.' && t.args.length === 2) {
          const { items, tail } = listItems(t)
          const body = items.map((x) => go(x, 999, depth + 1)).join(', ')
          return tail.kind === 'atom' && tail.name === '[]' ? `[${body}]` : `[${body}|${go(tail, 999, depth + 1)}]`
        }
        const op = t.args.length === 2 ? INFIX[t.functor] : undefined
        if (op) {
          const left = op.type === 'yfx' ? op.priority : op.priority - 1
          const right = op.type === 'xfy' ? op.priority : op.priority - 1
          const sep = t.functor === ',' ? ', ' : /^[a-z]/.test(t.functor) ? ` ${t.functor} ` : ` ${t.functor} `
          const s = `${go(t.args[0], left, depth + 1)}${sep}${go(t.args[1], right, depth + 1)}`
          return op.priority > max ? `(${s})` : s
        }
        const pre = t.args.length === 1 ? PREFIX[t.functor] : undefined
        if (pre) {
          const arg = go(t.args[0], pre.type === 'fy' ? pre.priority : pre.priority - 1, depth + 1)
          const s = `${t.functor}${/^[0-9(+\-*/\\^<>=~:.?@#&$]/.test(arg) || /^[a-z]/.test(t.functor) || t.functor === '\\+' ? ' ' : ''}${arg}`
          return pre.priority > max ? `(${s})` : s
        }
        return `${quoteAtom(t.functor)}(${t.args.map((x) => go(x, 999, depth + 1)).join(', ')})`
      }
    }
  }
  return go(term, 1200, 0)
}

/**
 * A substitution as `{X = a, Y = f(Z)}`, its bindings in order.
 *
 * @param theta The substitution.
 * @param options How variables and values print, as in `termToString`.
 * @returns The text.
 *
 * @example Print a unifier
 * const { goals: [s, t] } = parseQuery('p(X, [Y | T]), p(1, [2, 3])')
 * print(substitutionToString(unify(s, t)))
 */
export function substitutionToString(theta: Substitution, options: PrintOptions = {}): string {
  const name = options.variableName ?? ((v: Variable) => v.name)
  return `{${theta.map((b) => `${name(b.variable)} = ${termToString(b.value, options)}`).join(', ')}}`
}
