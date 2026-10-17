/**
 * Unification step by step: Martelli and Montanari's transformation rules, one rule per state, so a figure can show the
 * equations being solved.
 *
 * Martelli and Montanari (1982), "An Efficient Unification Algorithm", ACM TOPLAS 4(2), §2. The state is a list of
 * equations still to solve and the solved bindings $X \mapsto t$. Each step takes the first equation $s = t$ and
 * applies one rule: delete ($s$ and $t$ identical), decompose (the same function symbol and arity: replace the
 * equation by the equations of the arguments), conflict (different symbols: fail), swap ($t$ a variable and $s$ not:
 * write it as $t = s$), eliminate ($s$ a variable not occurring in $t$: bind it and substitute it everywhere), or the
 * occurs check ($s$ a variable occurring in $t$: fail). Without the occurs check the last case binds the variable
 * anyway, as Prolog's `=/2` does, which stands for an infinite term.
 *
 * The solved bindings are composed eagerly (each new binding is substituted into the earlier values), so the final list
 * is in solved form and is the unifier `unify` returns, up to the direction of variable-to-variable bindings.
 */
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  applySubstitution,
  termsEqual,
  termToString,
  termVariables,
  type Binding,
  type Term,
  type Variable,
} from './terms'

/** The rule a step of `unificationTrace` applied (`start` and `done` mark the first and last states). */
export type UnificationRule = 'start' | 'delete' | 'decompose' | 'conflict' | 'swap' | 'eliminate' | 'occurs' | 'done'

/** One state of `unificationTrace`. */
export interface UnificationStep {
  /** The equations still to solve, first to last; the first is the next one treated. */
  readonly equations: readonly (readonly [Term, Term])[]
  /** The solved bindings, in the order made, with each later binding already substituted into the earlier values. */
  readonly solved: readonly Binding[]
  /** The rule this step applied. */
  readonly rule: UnificationRule
  /** The equation the rule applied to, as it was before the step (null for `start` and `done`). */
  readonly equation: readonly [Term, Term] | null
  /** What happened, in one sentence. */
  readonly note: string
  /** `unified` when no equation is left, `failed` after a conflict or a failed occurs check, else `running`. */
  readonly status: 'running' | 'unified' | 'failed'
}

/** Options of `unificationTrace`. */
export interface UnificationTraceOptions {
  /**
   * Fail on an equation $X = t$ with $X$ occurring in $t$ (default false, as `unify` and Prolog's `=/2`); without it
   * the variable is bound to $t$ and stands for an infinite term.
   */
  occursCheck?: boolean
  /** The most states made before giving up with `DomainError` (default 10 000). */
  maxSteps?: number
}

/**
 * The arguments of a term (none unless it is a compound).
 *
 * @param t The term.
 * @returns Its arguments.
 */
const argsOf = (t: Term): readonly Term[] => (t.kind === 'compound' ? t.args : [])

/**
 * The symbol a term starts with, as `name/arity` (`f/2`, `a/0`), or the number itself for a number.
 *
 * @param t A term that is not a variable.
 * @returns Its symbol as text; two terms decompose against each other exactly when their symbols are equal.
 */
const symbol = (t: Term): string =>
  t.kind === 'compound' ? `${t.functor}/${t.args.length}` : t.kind === 'atom' ? `${t.name}/0` : termToString(t)

/**
 * Whether variable $v$ occurs in term $t$.
 *
 * @param v The variable.
 * @param t The term searched.
 * @returns True when a variable of `t` has `v`'s id.
 */
const occursIn = (v: Variable, t: Term): boolean => termVariables(t).some((w) => w.id === v.id)

/**
 * An equation as Prolog text, `s = t`.
 *
 * @param s The left-hand side.
 * @param t The right-hand side.
 * @returns The text.
 */
const eq = (s: Term, t: Term): string => `${termToString(s)} = ${termToString(t)}`

/**
 * Every state of Martelli and Montanari's unification algorithm on the equation $a = b$, from the start to `unified`
 * or `failed`, one rule per state. The final state's `solved` bindings are a most general unifier (in solved form);
 * a failure is reported in `status`, never thrown. The step-by-step companion of `unify`.
 *
 * @param a The first term.
 * @param b The second term; a variable with the same id as one of `a`'s is the same variable.
 * @param options Whether to apply the occurs check, and the step limit (see `UnificationTraceOptions`). A run longer
 *   than the limit throws `DomainError`.
 * @returns The states, one per rule applied: the first holds the single equation $a = b$ (rule `start`), the last
 *   has status `unified` (rule `done`) or `failed` (rule `conflict` or `occurs`).
 *
 * @example Solve an equation rule by rule
 * const { goals: [s, t] } = parseQuery('p(f(X), Y, b), p(Z, g(Z), W)')
 * const states = unificationTrace(s, t)
 * for (const st of states) print(st.rule, '|', substitutionToString(st.solved))
 * print('final status:', states[states.length - 1].status)
 *
 * @example The occurs check
 * const { goals: [x, fx] } = parseQuery('X, f(X)')
 * print('with:', unificationTrace(x, fx, { occursCheck: true }).map((st) => st.rule))
 * print('without:', unificationTrace(x, fx).map((st) => st.rule))
 */
export function unificationTrace(a: Term, b: Term, options: UnificationTraceOptions = {}): UnificationStep[] {
  const occursCheck = options.occursCheck ?? false
  const maxSteps = options.maxSteps ?? 10_000
  let equations: (readonly [Term, Term])[] = [[a, b]]
  let solved: Binding[] = []
  const steps: UnificationStep[] = [
    {
      equations,
      solved,
      rule: 'start',
      equation: null,
      note: `Start with the equation ${eq(a, b)}.`,
      status: 'running',
    },
  ]
  const push = (
    rule: UnificationRule,
    equation: readonly [Term, Term] | null,
    note: string,
    status: UnificationStep['status'] = 'running',
  ) => steps.push({ equations, solved, rule, equation, note, status })

  for (let guard = 0; guard < maxSteps; guard++) {
    if (equations.length === 0) {
      push('done', null, 'No equations are left: the solved bindings are a most general unifier.', 'unified')
      return steps
    }
    const [s, t] = equations[0]
    const rest = equations.slice(1)
    if (termsEqual(s, t)) {
      equations = rest
      push('delete', [s, t], `Delete ${eq(s, t)}: both sides are the same term.`)
    } else if (s.kind === 'var') {
      const cyclic = occursIn(s, t)
      if (cyclic && occursCheck) {
        equations = rest
        push(
          'occurs',
          [s, t],
          `Fail: ${s.name} occurs in ${termToString(t)}, so no finite term satisfies ${eq(s, t)}.`,
          'failed',
        )
        return steps
      }
      const theta: Binding[] = [{ variable: s, value: t }]
      const sub = (u: Term) => applySubstitution(u, theta)
      equations = rest.map(([l, r]) => [sub(l), sub(r)] as const)
      solved = [...solved.map((x) => ({ variable: x.variable, value: sub(x.value) })), { variable: s, value: t }]
      push(
        'eliminate',
        [s, t],
        cyclic
          ? `Bind ${s.name} ↦ ${termToString(t)} without the occurs check: ${s.name} stands for an infinite term.`
          : `Eliminate ${s.name}: bind ${s.name} ↦ ${termToString(t)} and replace ${s.name} by it in every other equation and binding.`,
      )
    } else if (t.kind === 'var') {
      equations = [[t, s], ...rest]
      push('swap', [s, t], `Swap ${eq(s, t)} to ${eq(t, s)}, so the variable is on the left.`)
    } else if (symbol(s) !== symbol(t)) {
      equations = rest
      push(
        'conflict',
        [s, t],
        `Fail: ${termToString(s)} and ${termToString(t)} start with different symbols (${symbol(s)} and ${symbol(t)}).`,
        'failed',
      )
      return steps
    } else {
      const tArgs = argsOf(t)
      const pairs = argsOf(s).map((x, i) => [x, tArgs[i]] as const)
      equations = [...pairs, ...rest]
      push(
        'decompose',
        [s, t],
        `Decompose ${eq(s, t)}: the same symbol ${symbol(s)}, so the arguments must be equal pairwise: ${pairs.map(([x, y]) => eq(x, y)).join(', ')}.`,
      )
    }
  }
  throw new DomainError('unificationTrace', `unificationTrace: no end after ${maxSteps} steps`)
}
