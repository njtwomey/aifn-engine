/**
 * A small Prolog: SLD resolution with depth-first search and backtracking, as a step-through `Algorithm`.
 *
 * The state is the textbook one. A resolvent is a list of goals; resolving its first goal G against a clause H :- B
 * (renamed apart) with the most general unifier θ of G and H gives the resolvent (B, rest)θ. The query's variables are
 * kept instantiated in `answer`, so the empty resolvent is a solution with the answer's bindings. When G has several
 * clauses whose heads unify, the remaining ones are kept on a stack of choice points; a goal with none fails, and the
 * search backtracks to the most recent choice point (depth first, left to right, clauses in program order: the order
 * of standard Prolog's solutions).
 *
 * Each step does one thing and draws one node of the SLD tree: resolves a goal with a clause, runs a built-in, resumes
 * an alternative after backtracking, records a success, or fails. Cut (`!`) removes the choice points made since its
 * clause was called; the branches it removes are reported as pruned nodes. Negation as failure (`\+ G`), if-then-else,
 * disjunction, `call/1` and `findall/3` are built from choice points and internal goals, so they show in the tree too.
 *
 * Built-ins: `true`, `fail`/`false`, `!`, `,`, `;`, `->`, `\+`/`not`, `call/1`, `findall/3`, `=`, `\=`, `==`, `\==`,
 * `is`, `=:=`, `=\=`, `<`, `>`, `=<`, `>=`, `var`, `nonvar`, `atom`, `number`, `integer`, `atomic`, `write`, `print`,
 * `nl`. The library (`member`, `append`, `select`, `permutation`, `reverse`, `length`, `last`, `sum_list`, `between`)
 * is Prolog source in `library.ts`.
 */
import { DomainError } from 'aifn-compute/foundation/errors'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import {
  applySubstitution,
  compound,
  indicator,
  isGround,
  listTerm,
  numeral,
  parseProgram,
  parseQuery,
  renameVariables,
  termsEqual,
  termToString,
  unify,
  variable,
  type Clause,
  type Query,
  type Substitution,
  type Term,
} from 'aifn-compute/logic/terms'
import { LIBRARY_SOURCE } from './library'

// ── Programs ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A program ready to run: its clauses (the program's, then the library's it does not replace) indexed by predicate. */
export interface PrologProgram {
  readonly clauses: readonly Clause[]
  /** Where each clause came from: the program, or the library. */
  readonly sources: readonly ('program' | 'library')[]
  /** Clause indices of each predicate `name/arity`, in program order. */
  readonly predicates: Readonly<Record<string, readonly number[]>>
}

const LIBRARY = parseProgram(LIBRARY_SOURCE).clauses

/** A program from Prolog text or clauses, with the library predicates it does not define (`library: false` for none). */
export function prologProgram(source: string | readonly Clause[], options: { library?: boolean } = {}): PrologProgram {
  const own = typeof source === 'string' ? parseProgram(source).clauses : [...source]
  const defined = new Set(own.map((c) => indicator(c.head)))
  const lib = options.library === false ? [] : LIBRARY.filter((c) => !defined.has(indicator(c.head)))
  const clauses = [...own, ...lib]
  const predicates: Record<string, number[]> = {}
  clauses.forEach((c, i) => (predicates[indicator(c.head)] ??= []).push(i))
  return {
    clauses,
    sources: clauses.map((_, i) => (i < own.length ? 'program' : 'library')),
    predicates,
  }
}

// ── State ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A goal of a resolvent and the choice-point height a cut inside it cuts back to. */
export interface SldGoal {
  readonly term: Term
  readonly cut: number
}

/** A choice point: where the search resumes on backtracking. `node` is the tree node where the choice was made. */
export type SldChoice =
  | {
      readonly kind: 'clauses'
      readonly node: number
      readonly depth: number
      /** The goal being resolved, its rest of the resolvent, and the answer then. */
      readonly goal: Term
      readonly rest: readonly SldGoal[]
      readonly answer: readonly Term[]
      /** Clauses still to try, all with heads that unify with the goal. */
      readonly clauses: readonly number[]
    }
  | {
      readonly kind: 'alternative'
      readonly node: number
      readonly depth: number
      readonly goals: readonly SldGoal[]
      readonly answer: readonly Term[]
      /** What resuming means, e.g. "\+ G succeeds: G has no proof". */
      readonly note: string
    }
  | {
      readonly kind: 'findall'
      readonly node: number
      readonly depth: number
      readonly rest: readonly SldGoal[]
      readonly answer: readonly Term[]
      /** The list argument, unified with the collected results when the search inside is exhausted. */
      readonly result: Term
      readonly results: readonly Term[]
    }

/** A branch removed by a cut before it was tried: a node in the tree, never expanded. */
export interface PrunedBranch {
  readonly node: number
  readonly parent: number
  /** The clause it would have used, or null for an alternative of `;`, `->` or `\+`. */
  readonly clause: number | null
  readonly note: string
}

/** What one step did. Every kind but `exhausted` names the tree node it created or closed. */
export type SldEvent =
  | { readonly kind: 'start'; readonly node: number }
  | {
      readonly kind: 'resolve'
      readonly node: number
      readonly parent: number
      /** The goal resolved and the clause used (renamed apart), with their most general unifier. */
      readonly goal: Term
      readonly clause: number
      readonly head: Term
      readonly unifier: Substitution
      /** True when the clause was taken from a choice point after backtracking. */
      readonly retry: boolean
      /** Clauses left at this choice point. */
      readonly alternatives: number
    }
  | {
      readonly kind: 'builtin'
      readonly node: number
      readonly parent: number
      readonly goal: Term
      readonly unifier: Substitution
      readonly note: string
      readonly pruned: readonly PrunedBranch[]
    }
  | { readonly kind: 'alternative'; readonly node: number; readonly parent: number; readonly note: string }
  | { readonly kind: 'success'; readonly node: number; readonly solution: number }
  | {
      readonly kind: 'fail'
      readonly node: number
      readonly goal: Term | null
      readonly reason: string
      readonly pruned: readonly PrunedBranch[]
    }
  | { readonly kind: 'error'; readonly node: number; readonly goal: Term; readonly message: string }
  | { readonly kind: 'exhausted' }

/** A solution: the named query variables' values. */
export interface SldSolution {
  readonly bindings: readonly { readonly name: string; readonly value: Term }[]
  readonly node: number
  readonly step: number
}

/** The state of `sldSteps`. */
export interface SldState {
  t: number
  terminated?: boolean
  /** The current resolvent, or null after a success or failure (the next step backtracks). */
  readonly goals: readonly SldGoal[] | null
  /** The query's variables (by id) under the current resolvent's substitution. */
  readonly answer: readonly Term[]
  /** The tree node of the current resolvent, and its depth (the number of steps from the query). */
  readonly node: number
  readonly depth: number
  readonly choices: readonly SldChoice[]
  /** The next fresh variable id, and the number of tree nodes so far. */
  readonly fresh: number
  readonly nodes: number
  /** Clause uses so far: renamed variables are suffixed with it (`X_3`). */
  readonly renames: number
  readonly event: SldEvent
  readonly solutions: readonly SldSolution[]
  /** What `write`, `print` and `nl` produced. */
  readonly output: string
  /** True once a branch was cut off at the depth limit (answers may then be missing). */
  readonly depthLimited: boolean
  readonly error: string | null
  /** Why the search ended: every branch explored, enough solutions, or an error. */
  readonly stopped: 'exhausted' | 'solutions' | 'error' | null
}

/** Options of `sldSteps` and `solveQuery`. */
export interface SldOptions {
  /** Branches deeper than this many steps fail (and set `depthLimited`). Default 200. */
  maxDepth?: number
  /** Stop after this many solutions. Default: all. */
  maxSolutions?: number
  /** Unify with the occurs check (default false, as standard Prolog). */
  occursCheck?: boolean
}

// ── Arithmetic ───────────────────────────────────────────────────────────────────────────────────────────────────────

class PrologError extends Error {}

function evaluate(t: Term): number {
  switch (t.kind) {
    case 'number':
      return t.value
    case 'var':
      throw new PrologError(`arguments are not sufficiently instantiated: ${t.name}`)
    case 'atom':
      if (t.name === 'pi') return Math.PI
      if (t.name === 'e') return Math.E
      if (t.name === 'inf' || t.name === 'infinite') return Infinity
      throw new PrologError(`type error: ${t.name} is not a number`)
    case 'compound': {
      const a = t.args.map(evaluate)
      const int = (x: number) => {
        if (!Number.isInteger(x)) throw new PrologError(`type error: ${x} is not an integer`)
        return x
      }
      if (a.length === 1)
        switch (t.functor) {
          case '-':
            return -a[0]
          case '+':
            return a[0]
          case 'abs':
            return Math.abs(a[0])
          case 'sign':
            return Math.sign(a[0])
          case 'sqrt':
            return Math.sqrt(a[0])
          case 'exp':
            return Math.exp(a[0])
          case 'log':
            return Math.log(a[0])
          case 'floor':
            return Math.floor(a[0])
          case 'ceiling':
            return Math.ceil(a[0])
          case 'round':
            return Math.round(a[0])
          case 'truncate':
            return Math.trunc(a[0])
        }
      if (a.length === 2)
        switch (t.functor) {
          case '+':
            return a[0] + a[1]
          case '-':
            return a[0] - a[1]
          case '*':
            return a[0] * a[1]
          case '/': {
            if (a[1] === 0) throw new PrologError('evaluation error: division by zero')
            return a[0] / a[1]
          }
          case '//': {
            if (int(a[1]) === 0) throw new PrologError('evaluation error: division by zero')
            return Math.trunc(int(a[0]) / a[1])
          }
          case 'mod': {
            if (int(a[1]) === 0) throw new PrologError('evaluation error: division by zero')
            const m = int(a[0]) % a[1]
            return m !== 0 && Math.sign(m) !== Math.sign(a[1]) ? m + a[1] : m
          }
          case 'rem':
            return int(a[0]) % int(a[1])
          case 'min':
            return Math.min(a[0], a[1])
          case 'max':
            return Math.max(a[0], a[1])
          case '**':
          case '^':
            return a[0] ** a[1]
        }
      throw new PrologError(`unknown arithmetic function ${t.functor}/${t.args.length}`)
    }
  }
}

const COMPARE: Record<string, (a: number, b: number) => boolean> = {
  '=:=': (a, b) => a === b,
  '=\\=': (a, b) => a !== b,
  '<': (a, b) => a < b,
  '>': (a, b) => a > b,
  '=<': (a, b) => a <= b,
  '>=': (a, b) => a >= b,
}

const TYPE_TESTS: Record<string, (t: Term) => boolean> = {
  var: (t) => t.kind === 'var',
  nonvar: (t) => t.kind !== 'var',
  atom: (t) => t.kind === 'atom',
  number: (t) => t.kind === 'number',
  integer: (t) => t.kind === 'number' && Number.isInteger(t.value),
  atomic: (t) => t.kind === 'atom' || t.kind === 'number',
  ground: (t) => isGround(t),
}

// ── The algorithm ────────────────────────────────────────────────────────────────────────────────────────────────────

const show = (t: Term) => termToString(t)

/**
 * SLD resolution of `query` against `program`, one tree event per step (see the module comment). Step 0 is the query
 * (node 0); the run ends (`terminated`) when the choice points are exhausted, `maxSolutions` are found, or a goal
 * raises an error (an unknown predicate, unbound arithmetic). Branches deeper than `maxDepth` fail with a note.
 */
export function sldSteps(
  program: PrologProgram,
  query: Query | string,
  options: SldOptions = {},
): Algorithm<void, SldState> {
  const q: Query = typeof query === 'string' ? parseQuery(query) : query
  const maxDepth = options.maxDepth ?? 200
  const maxSolutions = options.maxSolutions ?? Infinity
  const occursCheck = options.occursCheck ?? false
  const named = q.variableNames.map((name, id) => ({ name, id })).filter((v) => !v.name.startsWith('_'))

  /** Apply θ to a list of goals and the answer. */
  const applyAll = (goals: readonly SldGoal[], theta: Substitution) =>
    theta.length === 0 ? goals : goals.map((g) => ({ term: applySubstitution(g.term, theta), cut: g.cut }))
  const applyAnswer = (answer: readonly Term[], theta: Substitution) =>
    theta.length === 0 ? answer : answer.map((x) => applySubstitution(x, theta))

  /** The clause renamed apart: variables get fresh ids from `fresh` and names suffixed with `_<use>`. */
  const renamed = (i: number, fresh: number, use: number) => {
    const c = program.clauses[i]
    const term = renameVariables(compound(':-', [c.head, compound('$body', c.body)]), fresh) as unknown as {
      args: [Term, Term]
    }
    const names = (t: Term): Term =>
      t.kind === 'var'
        ? variable(`${t.name === '_' ? '_G' : t.name}_${use}`, t.id)
        : t.kind === 'compound'
          ? { ...t, args: t.args.map(names) }
          : t
    const head = names(term.args[0])
    const bodyTerm = names(term.args[1])
    const body = bodyTerm.kind === 'compound' ? bodyTerm.args : []
    return { head, body, size: c.variableNames.length }
  }

  /** Clauses of the goal's predicate whose head unifies with it, in program order. */
  const viable = (goal: Term, fresh: number): number[] | null => {
    const key = indicator(goal)
    const clauses = program.predicates[key]
    if (!clauses) return null
    return clauses.filter((i) => unify(goal, renameVariables(program.clauses[i].head, fresh), { occursCheck }) !== null)
  }

  /** Remove the choice points above `height`; clause and alternative branches they held become pruned nodes. */
  const cutTo = (s: SldState, height: number, nodes: number) => {
    const pruned: PrunedBranch[] = []
    for (const c of s.choices.slice(height)) {
      if (c.kind === 'clauses')
        for (const clause of c.clauses)
          pruned.push({ node: nodes + pruned.length, parent: c.node, clause, note: 'cut' })
      else if (c.kind === 'alternative')
        pruned.push({ node: nodes + pruned.length, parent: c.node, clause: null, note: c.note })
    }
    return { choices: s.choices.slice(0, height), pruned, nodes: nodes + pruned.length }
  }

  /** Resolve `goal` with clause `i`: the child node's state. */
  const resolveWith = (
    s: SldState,
    i: number,
    goal: Term,
    rest: readonly SldGoal[],
    answer: readonly Term[],
    parent: number,
    depth: number,
    choices: readonly SldChoice[],
    height: number,
    retry: boolean,
    alternatives: number,
  ): SldState => {
    const { head, body, size } = renamed(i, s.fresh, s.renames + 1)
    const theta = unify(goal, head, { occursCheck })!
    const goals = applyAll([...body.map((term) => ({ term, cut: height })), ...rest], theta)
    const node = s.nodes
    return {
      ...s,
      t: s.t + 1,
      goals,
      answer: applyAnswer(answer, theta),
      node,
      depth,
      choices,
      fresh: s.fresh + size,
      nodes: s.nodes + 1,
      renames: s.renames + 1,
      event: { kind: 'resolve', node, parent, goal, clause: i, head, unifier: theta, retry, alternatives },
    }
  }

  /** A failure of the current node: the next step backtracks. */
  const fail = (s: SldState, goal: Term | null, reason: string, extra: Partial<SldState> = {}): SldState => ({
    ...s,
    t: s.t + 1,
    goals: null,
    event: { kind: 'fail', node: s.node, goal, reason, pruned: [] },
    ...extra,
  })

  /** A built-in that succeeded: a child node with the given resolvent. */
  const builtin = (
    s: SldState,
    goal: Term,
    goals: readonly SldGoal[],
    note: string,
    extra: {
      unifier?: Substitution
      choices?: readonly SldChoice[]
      pruned?: readonly PrunedBranch[]
      nodes?: number
    } = {},
  ): SldState => {
    const theta = extra.unifier ?? []
    const node = extra.nodes ?? s.nodes
    return {
      ...s,
      t: s.t + 1,
      goals: applyAll(goals, theta),
      answer: applyAnswer(s.answer, theta),
      node,
      depth: s.depth + 1,
      choices: extra.choices ?? s.choices,
      nodes: node + 1,
      event: { kind: 'builtin', node, parent: s.node, goal, unifier: theta, note, pruned: extra.pruned ?? [] },
    }
  }

  const backtrack = (s: SldState): SldState => {
    if (s.choices.length === 0)
      return { ...s, t: s.t + 1, terminated: true, stopped: 'exhausted', event: { kind: 'exhausted' } }
    const c = s.choices[s.choices.length - 1]
    const below = s.choices.slice(0, -1)
    if (c.kind === 'clauses') {
      const [i, ...left] = c.clauses
      const choices = left.length ? [...below, { ...c, clauses: left }] : below
      return resolveWith(s, i, c.goal, c.rest, c.answer, c.node, c.depth + 1, choices, below.length, true, left.length)
    }
    const node = s.nodes
    if (c.kind === 'alternative')
      return {
        ...s,
        t: s.t + 1,
        goals: c.goals,
        answer: c.answer,
        node,
        depth: c.depth + 1,
        choices: below,
        nodes: node + 1,
        event: { kind: 'alternative', node, parent: c.node, note: c.note },
      }
    // findall: the search inside is exhausted; unify the list of results with the third argument.
    const list = listTerm(c.results)
    const theta = unify(c.result, list, { occursCheck })
    const base = { ...s, node, depth: c.depth + 1, choices: below, nodes: node + 1 }
    const note = `findall collected ${show(list)}`
    if (theta === null)
      return {
        ...base,
        t: s.t + 1,
        goals: null,
        answer: c.answer,
        event: {
          kind: 'alternative',
          node,
          parent: c.node,
          note: `${note}, which does not unify with ${show(c.result)}`,
        },
      }
    return {
      ...base,
      t: s.t + 1,
      goals: applyAll(c.rest, theta),
      answer: applyAnswer(c.answer, theta),
      event: { kind: 'alternative', node, parent: c.node, note },
    }
  }

  const raise = (s: SldState, goal: Term, message: string): SldState => ({
    ...s,
    t: s.t + 1,
    goals: null,
    terminated: true,
    stopped: 'error',
    error: message,
    event: { kind: 'error', node: s.node, goal, message },
  })

  const solveGoal = (s: SldState, first: SldGoal, rest: readonly SldGoal[]): SldState => {
    const g = first.term
    if (g.kind === 'var')
      return raise(s, g, `arguments are not sufficiently instantiated: the goal is the variable ${g.name}`)
    if (g.kind === 'number') return raise(s, g, `type error: ${show(g)} is not callable`)
    const name = g.kind === 'atom' ? g.name : g.functor
    const args = g.kind === 'compound' ? g.args : []
    const key = `${name}/${args.length}`
    const height = s.choices.length
    const unifyGoal = (a: Term, b: Term, note: string) => {
      const theta = unify(a, b, { occursCheck })
      return theta === null
        ? fail(s, g, `${show(a)} and ${show(b)} do not unify`)
        : builtin(s, g, rest, note, { unifier: theta })
    }
    try {
      switch (key) {
        case 'true/0':
          return builtin(s, g, rest, 'true always succeeds')
        case 'fail/0':
        case 'false/0':
          return fail(s, g, `${name} always fails`)
        case '!/0': {
          const cut = cutTo(s, first.cut, s.nodes)
          const n = cut.pruned.length
          return builtin(
            s,
            g,
            rest,
            n === 0 ? 'cut: no choice points to remove' : `cut: ${n} untried branch${n > 1 ? 'es' : ''} removed`,
            { choices: cut.choices, pruned: cut.pruned, nodes: cut.nodes },
          )
        }
        case ',/2':
          return builtin(
            s,
            g,
            [{ term: args[0], cut: first.cut }, { term: args[1], cut: first.cut }, ...rest],
            'conjunction',
          )
        case ';/2': {
          const [left, right] = args
          if (left.kind === 'compound' && left.functor === '->' && left.args.length === 2) {
            // If-then-else: the else branch waits on a choice point that the condition's success removes.
            const alt: SldChoice = {
              kind: 'alternative',
              node: s.node,
              depth: s.depth,
              goals: [{ term: right, cut: first.cut }, ...rest],
              answer: s.answer,
              note: `the condition ${show(left.args[0])} failed: take the else branch`,
            }
            return builtin(
              s,
              g,
              [
                { term: left.args[0], cut: height + 1 },
                { term: compound('$then', [numeral(height)]), cut: first.cut },
                { term: left.args[1], cut: first.cut },
                ...rest,
              ],
              'if-then-else: try the condition',
              { choices: [...s.choices, alt] },
            )
          }
          const alt: SldChoice = {
            kind: 'alternative',
            node: s.node,
            depth: s.depth,
            goals: [{ term: right, cut: first.cut }, ...rest],
            answer: s.answer,
            note: `the right branch of ;`,
          }
          return builtin(s, g, [{ term: left, cut: first.cut }, ...rest], 'disjunction: left branch first', {
            choices: [...s.choices, alt],
          })
        }
        case '->/2':
          return builtin(
            s,
            g,
            [
              { term: args[0], cut: height },
              { term: compound('$then', [numeral(height)]), cut: first.cut },
              { term: args[1], cut: first.cut },
              ...rest,
            ],
            'if-then: try the condition',
          )
        case '$then/1': {
          const cut = cutTo(s, (args[0] as { value: number }).value, s.nodes)
          return builtin(s, g, rest, 'the condition succeeded: commit to the then branch', {
            choices: cut.choices,
            pruned: cut.pruned,
            nodes: cut.nodes,
          })
        }
        case '\\+/1':
        case 'not/1': {
          const alt: SldChoice = {
            kind: 'alternative',
            node: s.node,
            depth: s.depth,
            goals: rest,
            answer: s.answer,
            note: `${show(args[0])} has no proof, so \\+ succeeds`,
          }
          return builtin(
            s,
            g,
            [
              { term: args[0], cut: height + 1 },
              { term: compound('$naf', [numeral(height), args[0]]), cut: 0 },
            ],
            `negation as failure: try to prove ${show(args[0])}`,
            { choices: [...s.choices, alt] },
          )
        }
        case '$naf/2': {
          const cut = cutTo(s, (args[0] as { value: number }).value, s.nodes)
          return {
            ...fail(s, null, `${show(args[1])} was proved, so \\+ ${show(args[1])} fails`),
            choices: cut.choices,
            nodes: cut.nodes,
            event: {
              kind: 'fail',
              node: s.node,
              goal: null,
              reason: `${show(args[1])} was proved, so \\+ ${show(args[1])} fails`,
              pruned: cut.pruned,
            },
          }
        }
        case 'call/1':
          return builtin(s, g, [{ term: args[0], cut: height }, ...rest], 'call')
        case 'findall/3': {
          const [template, goal, result] = args
          const box: SldChoice = {
            kind: 'findall',
            node: s.node,
            depth: s.depth,
            rest,
            answer: s.answer,
            result,
            results: [],
          }
          return builtin(
            s,
            g,
            [
              { term: goal, cut: height + 1 },
              { term: compound('$collect', [numeral(height), template]), cut: 0 },
            ],
            `findall: collect ${show(template)} for every proof of ${show(goal)}`,
            { choices: [...s.choices, box] },
          )
        }
        case '$collect/2': {
          const k = (args[0] as { value: number }).value
          const box = s.choices[k]
          if (box.kind !== 'findall') throw new PrologError('internal: findall choice point lost')
          const choices = s.choices.map((c, i) => (i === k ? { ...box, results: [...box.results, args[1]] } : c))
          return fail(s, null, `findall collected ${show(args[1])}; backtrack for more`, { choices })
        }
        case '=/2':
          return unifyGoal(args[0], args[1], 'unify')
        case '\\=/2':
          return unify(args[0], args[1], { occursCheck }) === null
            ? builtin(s, g, rest, 'the terms do not unify')
            : fail(s, g, `${show(args[0])} and ${show(args[1])} unify`)
        case '==/2':
          return termsEqual(args[0], args[1]) ? builtin(s, g, rest, 'identical') : fail(s, g, 'not identical')
        case '\\==/2':
          return termsEqual(args[0], args[1]) ? fail(s, g, 'identical') : builtin(s, g, rest, 'not identical')
        case 'is/2': {
          const value = evaluate(args[1])
          return unifyGoal(args[0], numeral(value), `evaluate ${show(args[1])} = ${show(numeral(value))}`)
        }
        case 'write/1':
        case 'print/1':
          return { ...builtin(s, g, rest, 'write'), output: s.output + show(args[0]) }
        case 'nl/0':
          return { ...builtin(s, g, rest, 'new line'), output: s.output + '\n' }
      }
      if (args.length === 2 && name in COMPARE) {
        const a = evaluate(args[0])
        const b = evaluate(args[1])
        return COMPARE[name](a, b) ? builtin(s, g, rest, `${a} ${name} ${b}`) : fail(s, g, `${a} ${name} ${b} is false`)
      }
      if (args.length === 1 && name in TYPE_TESTS)
        return TYPE_TESTS[name](args[0]) ? builtin(s, g, rest, `${name} test`) : fail(s, g, `${name} test fails`)
    } catch (e) {
      if (e instanceof PrologError) return raise(s, g, e.message)
      throw e
    }
    const options = viable(g, s.fresh)
    if (options === null) return raise(s, g, `unknown procedure ${key}`)
    if (options.length === 0) return fail(s, g, `no clause head for ${key} unifies with ${show(g)}`)
    const [i, ...left] = options
    const choices: readonly SldChoice[] = left.length
      ? [
          ...s.choices,
          { kind: 'clauses', node: s.node, depth: s.depth, goal: g, rest, answer: s.answer, clauses: left },
        ]
      : s.choices
    return resolveWith(s, i, g, rest, s.answer, s.node, s.depth + 1, choices, height, false, left.length)
  }

  return {
    name: 'sld',
    init: () => ({
      t: 0,
      goals: q.goals.map((term) => ({ term, cut: 0 })),
      answer: q.variableNames.map((name, id) => variable(name, id)),
      node: 0,
      depth: 0,
      choices: [],
      fresh: q.variableNames.length,
      nodes: 1,
      renames: 0,
      event: { kind: 'start', node: 0 },
      solutions: [],
      output: '',
      depthLimited: false,
      error: null,
      stopped: null,
    }),
    step: (s) => {
      if (s.terminated) return { ...s, t: s.t + 1 }
      if (s.goals === null) return backtrack(s)
      if (s.goals.length === 0) {
        const solutions = [
          ...s.solutions,
          { bindings: named.map((v) => ({ name: v.name, value: s.answer[v.id] })), node: s.node, step: s.t + 1 },
        ]
        const enough = solutions.length >= maxSolutions
        return {
          ...s,
          t: s.t + 1,
          goals: null,
          solutions,
          event: { kind: 'success', node: s.node, solution: solutions.length - 1 },
          ...(enough ? { terminated: true, stopped: 'solutions' as const } : {}),
        }
      }
      if (s.depth >= maxDepth)
        return fail(s, s.goals[0].term, `depth limit ${maxDepth} reached`, { depthLimited: true })
      return solveGoal(s, s.goals[0], s.goals.slice(1))
    },
  }
}

// ── Running to the end ───────────────────────────────────────────────────────────────────────────────────────────────

/** The outcome of `solveQuery`. */
export interface SolveResult {
  readonly solutions: readonly SldSolution[]
  /** Solutions as text, e.g. `X = tom, Y = ann` (`true` for a solution without named variables). */
  readonly answers: readonly string[]
  readonly steps: number
  readonly stopped: 'exhausted' | 'solutions' | 'error' | 'steps'
  /** Why the search stopped, for a reader: an error, the step limit, or the depth limit cutting branches off. */
  readonly message: string
  readonly depthLimited: boolean
  readonly output: string
}

/** Prints the bindings of a solution as `X = a, Y = [b, c]` (`true` when none is bound). */
export function formatSolution(solution: SldSolution): string {
  // Variables left unbound are omitted, as Prolog's top level does.
  const bound = solution.bindings.filter((b) => !(b.value.kind === 'var' && b.value.name === b.name))
  if (bound.length === 0) return 'true'
  return bound.map((b) => `${b.name} = ${termToString(b.value)}`).join(', ')
}

/**
 * Run `sldSteps` to the end: every solution in standard Prolog order (or the first `maxSolutions`), stopping after
 * `maxSteps` steps (default 100 000) with a message saying so.
 */
export function solveQuery(
  program: PrologProgram | string,
  query: Query | string,
  options: SldOptions & { maxSteps?: number } = {},
): SolveResult {
  const prog = typeof program === 'string' ? prologProgram(program) : program
  const maxSteps = options.maxSteps ?? 100_000
  if (!(maxSteps >= 1)) throw new DomainError('solveQuery', 'solveQuery: maxSteps must be at least 1')
  const s = run(sldSteps(prog, query, options), undefined, maxSteps)
  const stopped = s.stopped ?? 'steps'
  const depthNote = s.depthLimited ? ' Some branches were cut off at the depth limit, so answers may be missing.' : ''
  const message =
    stopped === 'error'
      ? `Error: ${s.error}`
      : stopped === 'steps'
        ? `Stopped after the step limit of ${maxSteps} steps with ${s.solutions.length} solution${s.solutions.length === 1 ? '' : 's'}; the search was not finished.${depthNote}`
        : `${s.solutions.length} solution${s.solutions.length === 1 ? '' : 's'} in ${s.t} steps.${depthNote}`
  return {
    solutions: s.solutions,
    answers: s.solutions.map(formatSolution),
    steps: s.t,
    stopped,
    message,
    depthLimited: s.depthLimited,
    output: s.output,
  }
}
