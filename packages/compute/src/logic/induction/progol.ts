/**
 * Mode-directed inverse entailment: the bottom clause of an example, and the general-to-specific search of the clauses
 * below it (Progol and Aleph).
 *
 * Muggleton (1995), "Inverse entailment and Progol", New Generation Computing 13, §§5 to 7; Srinivasan, "The Aleph
 * Manual". `bottomClause` saturates one positive example against ground background facts: the head is the example
 * with its constants replaced by variables; at each depth every body mode is applied with its input (`+`) arguments
 * filled by every known term of the right type, every background fact that fits gives a literal, and the terms in its
 * output (`-`) positions become known for the next depth. Constants become variables, the same constant always the
 * same variable; arguments marked `#` keep their constant. `bottomClauseSearch` then searches the clauses whose body
 * is a subset of the bottom clause's body, breadth-first by length, each subset reached once (a literal is added only
 * after the last literal added, and only when its input variables occur in the head or an earlier literal). Each
 * clause is scored by Aleph's compression $P - N - L + 1$ ($P$ positives and $N$ negatives covered, $L$ literals with
 * the head). A clause that covers no negative is not refined further, and one whose refinements cannot beat the best
 * score so far ($P - L \le$ best) is pruned.
 *
 * Differences from Progol: the background is extensional (ground facts, no rules), recall is unbounded, constants are
 * told apart by name only (a constant met with two types gets one variable), and the search is breadth-first instead
 * of Progol's best-first order. Coverage is decided by SLD resolution (`solveQuery`).
 */
import { DomainError } from 'aifn-compute/foundation/errors'
import { prologProgram, solveQuery } from 'aifn-compute/logic/resolution'
import { clauseToString, compound, termToString, variable, type Term, type Variable } from 'aifn-compute/logic/terms'

/** One argument of a mode: input (`+`), output (`-`) or constant (`#`), with a type name. */
export interface ModeArgument {
  /** `+` an input (a known term), `-` an output (may be new), `#` a constant kept as it is. */
  readonly io: '+' | '-' | '#'
  /** The argument's type: only terms met with this type fill an input of it. */
  readonly type: string
}

/** A mode declaration such as `parent(+person, -person)`. */
export interface ModeDeclaration {
  /** The predicate's name. */
  readonly predicate: string
  /** Its arguments, in order. */
  readonly args: readonly ModeArgument[]
}

/**
 * A mode declaration from text such as `parent(+person, -person)`. Text of another shape throws `DomainError`.
 *
 * @param text The declaration: a predicate name and its arguments, each `+`, `-` or `#` followed by a type name.
 * @returns The parsed mode.
 *
 * @example Read a mode
 * print(parseMode('parent(+person, -person)'))
 */
export function parseMode(text: string): ModeDeclaration {
  const m = /^\s*([a-z]\w*)\s*\((.*)\)\s*$/.exec(text)
  if (!m) throw new DomainError('parseMode', `parseMode: not a mode declaration: ${text}`)
  const args = m[2].split(',').map((s) => {
    const a = /^\s*([+\-#])\s*([a-z]\w*)\s*$/.exec(s)
    if (!a) throw new DomainError('parseMode', `parseMode: bad mode argument: ${s}`)
    return { io: a[1] as '+' | '-' | '#', type: a[2] }
  })
  return { predicate: m[1], args }
}

/** A literal of a bottom clause with the depth at which it was found and its input and output variables. */
export interface BottomLiteral {
  /** The literal. */
  readonly literal: Term
  /** The depth (round of saturation) that added it, from 1. */
  readonly depth: number
  /** Ids of the variables in its input (`+`) positions. */
  readonly inputs: readonly number[]
  /** Ids of the variables in its output (`-`) positions. */
  readonly outputs: readonly number[]
}

/** A bottom clause: the head, the literals in order, and the constant each variable stands for. */
export interface BottomClause {
  /** The head: the example with its constants replaced by variables. */
  readonly head: Term
  /** The body literals, in the order found. */
  readonly literals: readonly BottomLiteral[]
  /** The constant each variable replaced, by variable id. */
  readonly constants: readonly string[]
  /** Variable names by id. */
  readonly names: readonly string[]
}

/**
 * The name of variable `i`: A to Z, then `V26`, `V27`, ….
 *
 * @param i The variable's id.
 * @returns Its name.
 */
const nameOf = (i: number) => (i < 26 ? String.fromCharCode(65 + i) : `V${i}`)

/**
 * The arguments of a term (none unless it is a compound).
 *
 * @param t The term.
 * @returns Its arguments.
 */
const argsOf = (t: Term): readonly Term[] => (t.kind === 'compound' ? t.args : [])

/**
 * A constant's name: an atom's name, a number's value, or the printed term.
 *
 * @param t A ground term.
 * @returns The name variables are keyed by.
 */
const constantName = (t: Term): string =>
  t.kind === 'atom' ? t.name : t.kind === 'number' ? String(t.value) : termToString(t)

/**
 * The bottom clause of a ground example: the most specific clause, within the modes and the depth, that with the
 * background facts entails the example (Muggleton, 1995).
 *
 * @param facts The background knowledge: ground facts.
 * @param example The positive example, a ground atom.
 * @param head The head mode (its predicate is the example's).
 * @param body The body modes, applied in order at each depth.
 * @param depth How many rounds of new terms to allow (Progol's variable depth $i$).
 * @returns The bottom clause.
 *
 * @example The bottom clause of a daughter example
 * import { parseProgram, parseTerm } from 'aifn-compute/logic/terms'
 * const facts = parseProgram('parent(ann, bob). parent(ann, cid). parent(cid, fay). female(ann). female(cid).')
 *   .clauses.map((c) => c.head)
 * const modes = ['female(+person)', 'parent(+person, -person)', 'parent(-person, +person)'].map(parseMode)
 * const bc = bottomClause(facts, parseTerm('daughter(cid, ann)'), parseMode('daughter(+person, +person)'), modes)
 * print(subsetText(bc, bc.literals.map((_, i) => i)))
 * print('constants:', bc.constants)
 */
export function bottomClause(
  facts: readonly Term[],
  example: Term,
  head: ModeDeclaration,
  body: readonly ModeDeclaration[],
  depth = 1,
): BottomClause {
  const constants: string[] = []
  const vars: Variable[] = []
  const varFor = (c: string): Variable => {
    let i = constants.indexOf(c)
    if (i < 0) {
      i = constants.length
      constants.push(c)
      vars.push(variable(nameOf(i), i))
    }
    return vars[i]
  }
  const headArgs = argsOf(example).map((a, k) => (head.args[k]?.io === '#' ? a : varFor(constantName(a))))
  const headTerm = compound(head.predicate, headArgs)
  // Known terms: (constant, type) pairs whose variable may fill an input position.
  const known = new Set<string>()
  argsOf(example).forEach((a, k) => {
    if (head.args[k] && head.args[k].io !== '#') known.add(`${constantName(a)}:${head.args[k].type}`)
  })
  const literals: BottomLiteral[] = []
  const seen = new Set<string>()
  for (let d = 1; d <= depth; d++) {
    const fresh: string[] = []
    for (const m of body) {
      for (const f of facts) {
        if (f.kind !== 'compound' || f.functor !== m.predicate || f.args.length !== m.args.length) continue
        const names = f.args.map(constantName)
        // Every input must already be known with the mode's type.
        if (!m.args.every((a, k) => a.io !== '+' || known.has(`${names[k]}:${a.type}`))) continue
        const lit = compound(
          m.predicate,
          f.args.map((a, k) => (m.args[k].io === '#' ? a : varFor(names[k]))),
        )
        const key = termToString(lit)
        if (seen.has(key)) continue
        seen.add(key)
        const ids = (io: '+' | '-') => m.args.flatMap((a, k) => (a.io === io ? [constants.indexOf(names[k])] : []))
        literals.push({ literal: lit, depth: d, inputs: ids('+'), outputs: ids('-') })
        m.args.forEach((a, k) => {
          if (a.io === '-') fresh.push(`${names[k]}:${a.type}`)
        })
      }
    }
    fresh.forEach((k) => known.add(k))
  }
  return { head: headTerm, literals, constants, names: vars.map((v) => v.name) }
}

/** A clause of the search: a subset of the bottom clause's literals, with its coverage and score. */
export interface SearchedClause {
  /** Indices of the bottom clause's literals in the body, increasing. */
  readonly subset: readonly number[]
  /** The clause as Prolog text. */
  readonly text: string
  /** Indices of the positive examples it covers. */
  readonly positives: readonly number[]
  /** Indices of the negative examples it covers. */
  readonly negatives: readonly number[]
  /** Compression $P - N - L + 1$. */
  readonly score: number
}

/** One step of the search: the clause evaluated, what was decided, and the best clause so far. */
export interface BottomSearchStep {
  /** The clause evaluated. */
  readonly clause: SearchedClause
  /**
   * `refine`: its refinements join the queue; `consistent`: it covers no negative and is not refined; `pruned`: its
   * refinements cannot beat the best; `leaf`: no literal can be added.
   */
  readonly decision: 'refine' | 'consistent' | 'pruned' | 'leaf'
  /** The best clause covering no negative so far, or null. */
  readonly best: SearchedClause | null
  /** What was decided, in words. */
  readonly note: string
}

/**
 * The clause of a subset of a bottom clause's literals, as Prolog text.
 *
 * @param bottom The bottom clause.
 * @param subset Indices of its literals to keep in the body, in order.
 * @returns The clause's text.
 *
 * @example The head alone and with one literal
 * import { parseProgram, parseTerm } from 'aifn-compute/logic/terms'
 * const facts = parseProgram('parent(ann, cid). female(cid).').clauses.map((c) => c.head)
 * const modes = ['female(+person)', 'parent(-person, +person)'].map(parseMode)
 * const bc = bottomClause(facts, parseTerm('daughter(cid, ann)'), parseMode('daughter(+person, +person)'), modes)
 * print(subsetText(bc, []), '|', subsetText(bc, [0]))
 */
export function subsetText(bottom: BottomClause, subset: readonly number[]): string {
  return clauseToString({ head: bottom.head, body: subset.map((i) => bottom.literals[i].literal) })
}

/** Options of `bottomClauseSearch`. */
export interface BottomSearchOptions {
  /** Stop after evaluating this many clauses (default 500). */
  maxClauses?: number
  /** The step limit of each coverage test by SLD resolution (default 20 000); a test that hits it counts as failed. */
  maxSteps?: number
}

/**
 * The general-to-specific search of the clauses below a bottom clause, one evaluated clause per step, breadth-first
 * by length and scored by compression (see the file comment). Each clause's coverage is decided by running it with
 * the background facts on the Prolog engine. A plain function of plain data, so it can run in a worker.
 *
 * @param bottom The bottom clause, as `bottomClause` returns it.
 * @param facts The background facts, used for the coverage tests.
 * @param positives The positive examples (ground atoms).
 * @param negatives The negative examples (ground atoms).
 * @param options The limits (see `BottomSearchOptions`).
 * @returns The steps in the order the clauses were evaluated; the last step's `best` is the clause returned.
 *
 * @example Search below the bottom clause of daughter(cid, ann)
 * import { parseProgram, parseTerm } from 'aifn-compute/logic/terms'
 * const facts = parseProgram(`parent(ann, bob). parent(ann, cid). parent(bob, dee). parent(bob, eve).
 *   parent(cid, fay). parent(eve, gus). female(ann). female(cid). female(dee). female(eve). female(fay).`)
 *   .clauses.map((c) => c.head)
 * const modes = ['female(+person)', 'parent(+person, -person)', 'parent(-person, +person)'].map(parseMode)
 * const bc = bottomClause(facts, parseTerm('daughter(cid, ann)'), parseMode('daughter(+person, +person)'), modes)
 * const pos = ['daughter(cid, ann)', 'daughter(dee, bob)', 'daughter(eve, bob)', 'daughter(fay, cid)'].map(parseTerm)
 * const neg = ['daughter(bob, ann)', 'daughter(gus, eve)', 'daughter(ann, cid)', 'daughter(dee, ann)'].map(parseTerm)
 * const steps = bottomClauseSearch(bc, facts, pos, neg)
 * print(steps.length, 'clauses evaluated; best:', steps[steps.length - 1].best.text)
 */
export function bottomClauseSearch(
  bottom: BottomClause,
  facts: readonly Term[],
  positives: readonly Term[],
  negatives: readonly Term[],
  options: BottomSearchOptions = {},
): BottomSearchStep[] {
  const maxClauses = options.maxClauses ?? 500
  const maxSteps = options.maxSteps ?? 20_000
  const factText = facts.map((f) => `${termToString(f)}.`).join('\n')
  const headVars = new Set(argsOf(bottom.head).flatMap((a) => (a.kind === 'var' ? [a.id] : [])))
  const evaluate = (subset: readonly number[]): SearchedClause => {
    const text = subsetText(bottom, subset)
    const program = prologProgram(`${factText}\n${text}`, { library: false })
    const covers = (e: Term) => solveQuery(program, termToString(e), { maxSolutions: 1, maxSteps }).solutions.length > 0
    const pos = positives.flatMap((e, i) => (covers(e) ? [i] : []))
    const neg = negatives.flatMap((e, i) => (covers(e) ? [i] : []))
    return { subset, text, positives: pos, negatives: neg, score: pos.length - neg.length - (subset.length + 1) + 1 }
  }
  // Literals that may follow a subset: after its last index, with every input variable already bound.
  const extensions = (subset: readonly number[]): number[] => {
    const bound = new Set(headVars)
    subset.forEach((i) => {
      bottom.literals[i].inputs.forEach((v) => bound.add(v))
      bottom.literals[i].outputs.forEach((v) => bound.add(v))
    })
    const last = subset.length ? subset[subset.length - 1] : -1
    return bottom.literals.flatMap((l, i) => (i > last && l.inputs.every((v) => bound.has(v)) ? [i] : []))
  }
  const steps: BottomSearchStep[] = []
  let best: SearchedClause | null = null
  const queue: number[][] = [[]]
  while (queue.length && steps.length < maxClauses) {
    const subset = queue.shift()!
    const clause = evaluate(subset)
    const P = clause.positives.length
    const N = clause.negatives.length
    const L = subset.length + 1
    let decision: BottomSearchStep['decision']
    let note: string
    if (N === 0 && P > 0) {
      if (!best || clause.score > best.score) best = clause
      decision = 'consistent'
      note = `Covers ${P} positives and no negative: score ${P} − 0 − ${L} + 1 = ${clause.score}. Adding literals cannot cover more positives, so it is not refined.`
    } else if (P === 0) {
      decision = 'pruned'
      note = `Covers no positive, and no refinement can cover one: pruned.`
    } else if (best && P - L <= best.score) {
      decision = 'pruned'
      note = `Covers ${P}+ ${N}−. A refinement covers at most ${P} positives with at least ${L + 1} literals, so it scores at most ${P - L}, no better than the best (${best.score}): pruned.`
    } else {
      const ext = extensions(subset)
      decision = ext.length ? 'refine' : 'leaf'
      for (const i of ext) queue.push([...subset, i])
      note = ext.length
        ? `Covers ${P}+ ${N}−, score ${clause.score}. It covers a negative, so its ${ext.length} refinement${ext.length > 1 ? 's' : ''} join the queue.`
        : `Covers ${P}+ ${N}−, score ${clause.score}. No literal of the bottom clause can be added.`
    }
    steps.push({ clause, decision, best, note })
  }
  return steps
}
