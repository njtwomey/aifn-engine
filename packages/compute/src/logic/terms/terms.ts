/**
 * First-order terms (the data of logic programs) and unification.
 *
 * A term is an atom (`mary`, `[]`), a number, a variable (`X`, identified by an integer id; the name is for printing)
 * or a compound `f(t₁, …, tₙ)`. Lists are compounds of the functor `.` with two arguments, ending in the atom `[]`:
 * `[a, b]` is `.(a, .(b, []))`. Terms are plain data, so they survive a structured clone.
 *
 * A substitution is a finite set of bindings {X₁ ↦ t₁, …}. `unify` returns the most general unifier of two terms in
 * solved form: no bound variable occurs in any bound term, so applying it once gives the final result (it is
 * idempotent, θθ = θ).
 */
import { DomainError } from 'aifn-compute/foundation/errors'

/** A constant: a name, such as `mary`, `[]` or `+`. */
export interface Atom {
  readonly kind: 'atom'
  readonly name: string
}
/** A number (integers and floats share one kind; integer-valued numbers print as integers). */
export interface Numeral {
  readonly kind: 'number'
  readonly value: number
}
/** A logic variable: equal ids are the same variable. `name` is how it prints (`X`, `_G3`). */
export interface Variable {
  readonly kind: 'var'
  readonly name: string
  readonly id: number
}
/** A compound term `functor(args…)`. */
export interface Compound {
  readonly kind: 'compound'
  readonly functor: string
  readonly args: readonly Term[]
}
export type Term = Atom | Numeral | Variable | Compound

/** One binding X ↦ t of a substitution. */
export interface Binding {
  readonly variable: Variable
  readonly value: Term
}
/** A substitution as its bindings, in the order the variables were bound. */
export type Substitution = readonly Binding[]

export const atom = (name: string): Atom => ({ kind: 'atom', name })
export const numeral = (value: number): Numeral => ({ kind: 'number', value })
export const variable = (name: string, id: number): Variable => ({ kind: 'var', name, id })
export const compound = (functor: string, args: readonly Term[]): Term =>
  args.length === 0 ? atom(functor) : { kind: 'compound', functor, args }

/** The empty list `[]`. */
export const NIL: Atom = atom('[]')

/** The list `[items… | tail]` (tail `[]` by default). */
export function listTerm(items: readonly Term[], tail: Term = NIL): Term {
  let out = tail
  for (let i = items.length - 1; i >= 0; i--) out = { kind: 'compound', functor: '.', args: [items[i], out] }
  return out
}

/** The items of a list term and its tail (`[]` for a proper list, a variable for a partial one). */
export function listItems(term: Term): { items: Term[]; tail: Term } {
  const items: Term[] = []
  let t = term
  while (t.kind === 'compound' && t.functor === '.' && t.args.length === 2) {
    items.push(t.args[0])
    t = t.args[1]
  }
  return { items, tail: t }
}

/** `name/arity` of an atom or compound: the key of a predicate. */
export function indicator(term: Term): string {
  if (term.kind === 'atom') return `${term.name}/0`
  if (term.kind === 'compound') return `${term.functor}/${term.args.length}`
  throw new DomainError('indicator', `indicator: ${termToString(term)} is not callable`)
}

/** The distinct variables of the terms, in order of first occurrence (left to right, depth first). */
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

/** True when the term has no variables. */
export function isGround(term: Term): boolean {
  if (term.kind === 'var') return false
  return term.kind !== 'compound' || term.args.every(isGround)
}

/** Structural identity (`==`): the same term, variables compared by id. */
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

/** The term with each variable `id` replaced by `map(id)` when that is defined; shares unchanged subterms. */
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

/** tθ: the term with every bound variable replaced by its value (one pass; a solved-form θ needs no more). */
export function applySubstitution(term: Term, theta: Substitution): Term {
  if (theta.length === 0) return term
  const map = new Map(theta.map((b) => [b.variable.id, b.value]))
  return mapVariables(term, (v) => map.get(v.id))
}

/**
 * The composition θσ: applying it equals applying θ then σ. Bindings of θ have σ applied to their values; bindings of
 * σ for variables θ does not bind are added; trivial bindings X ↦ X are dropped.
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
  /** Refuse to bind X to a term containing X (default false, as in Prolog's `=/2`). */
  occursCheck?: boolean
}

/**
 * The most general unifier of `a` and `b` (Robinson 1965, by Martelli and Montanari's rules), or null when they do not
 * unify. With the occurs check (`unify_with_occurs_check/2`) the result is idempotent: no variable it binds occurs in
 * any value, so θθ = θ. Without it (Prolog's `=/2`, the default), X = f(X) succeeds with the binding X ↦ f(X), which
 * stands for the infinite term f(f(…)); such a binding is returned unfolded once, and only it breaks idempotence.
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
 * One-way matching: a substitution θ binding only variables of `pattern` with patternθ = `term`, or null. Variables of
 * `term` are treated as constants. The test inside θ-subsumption.
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

/** The term with its variables renamed to fresh ids `offset`, `offset + 1`, … in order of first occurrence. */
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
/** Operators printed prefix. */
export const PREFIX: Readonly<Record<string, { priority: number; type: 'fy' | 'fx' }>> = {
  ':-': { priority: 1200, type: 'fx' },
  '?-': { priority: 1200, type: 'fx' },
  '\\+': { priority: 900, type: 'fy' },
  '-': { priority: 200, type: 'fy' },
}

const SOLO = new Set(['[]', '!', ';', ',', '{}'])
const quoteAtom = (name: string): string => {
  if (/^[a-z][A-Za-z0-9_]*$/.test(name) || SOLO.has(name) || /^[+\-*/\\^<>=~:.?@#&$]+$/.test(name)) return name
  return `'${name.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`
}
const formatNumber = (x: number): string =>
  Number.isInteger(x) ? String(x) : Number.isFinite(x) ? String(Number(x.toPrecision(15))) : String(x)

/** Options of `termToString`. */
export interface PrintOptions {
  /** How a variable prints; default its name. */
  variableName?: (v: Variable) => string
  /** Terms nested deeper than this print as `…` (default 64). */
  maxDepth?: number
}

/** A term in Prolog syntax: operators infix with standard priorities, lists in brackets. */
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

/** A substitution as `{X = a, Y = f(Z)}`. */
export function substitutionToString(theta: Substitution, options: PrintOptions = {}): string {
  const name = options.variableName ?? ((v: Variable) => v.name)
  return `{${theta.map((b) => `${name(b.variable)} = ${termToString(b.value, options)}`).join(', ')}}`
}
