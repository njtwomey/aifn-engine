/**
 * Learning from failures: a Popper-style learner that generates the smallest program no constraint prunes, tests it,
 * and turns each failure into constraints that prune the hypothesis space.
 *
 * Cropper and Morel (2021), "Learning programs by learning from failures", Machine Learning 110, §§3 and 4. The
 * hypothesis space is every set of at most `maxClauses` definite clauses over the declared predicates, with at most
 * `maxVars` variables and `maxBody` body literals per clause. The learner generates the smallest program (fewest
 * literals) that no constraint prunes, tests it on the examples with SLD resolution, and turns each failure into
 * constraints (the paper's Table 2): a program that proves a negative example is too general, so every generalisation
 * of it is pruned; a program that misses a positive is too specific, so every specialisation of it is pruned; a
 * program that proves no positive also prunes every non-recursive (separable) program that contains a specialisation
 * of one of its clauses (the elimination constraint). Generality is theory subsumption: $T_1 \succeq T_2$ when every
 * clause of $T_2$ is $\theta$-subsumed by a clause of $T_1$. The first program that proves every positive and no
 * negative is returned; it has the fewest literals of any solution in the space.
 *
 * Differences from Popper: the space is enumerated explicitly and the constraints are kept as a pruned set, instead of
 * answer-set constraints solved by Clingo, so it suits small spaces (a few thousand programs); ties in size are broken
 * by clause order, not by the solver; the bias also requires every head variable to occur in the body, every other
 * variable to occur at least twice, variables to respect declared argument types, a recursive program to have a
 * non-recursive clause, and no clause of a program to subsume another. Every test is recorded, for a stepped figure.
 * Problems and results are plain data, so `learnFromFailures` can run in a worker.
 */
import { DomainError } from 'aifn-compute/foundation/errors'
import { prologProgram, solveQuery } from 'aifn-compute/logic/resolution'
import { compound, variable, type Term } from 'aifn-compute/logic/terms'
import { thetaSubsumes, type HornClause } from './generalise'

/** A predicate the learner may use, with an optional type per argument (variables must then agree in type). */
export interface LffPredicate {
  /** The predicate's name. */
  readonly name: string
  /** Its number of arguments. */
  readonly arity: number
  /** A type name per argument; a variable may only fill arguments of one type. Left out, any variable fits. */
  readonly types?: readonly string[]
}

/** A learning-from-failures problem: background knowledge, the target, the body predicates and the examples. */
export interface LffProblem {
  /** The background knowledge as Prolog text. */
  readonly background: string
  /** The target predicate (the head of every clause). */
  readonly target: LffPredicate
  /** The predicates allowed in bodies (the target is added when `recursion` is on). */
  readonly body: readonly LffPredicate[]
  /** Positive examples: ground atoms of the target, as Prolog text. */
  readonly positives: readonly string[]
  /** Negative examples: ground atoms of the target, as Prolog text. */
  readonly negatives: readonly string[]
}

/** Bounds of the hypothesis space. */
export interface LffBias {
  /** Variables per clause at most (default 3); at least the target's arity. */
  readonly maxVars?: number
  /** Body literals per clause at most (default 2). */
  readonly maxBody?: number
  /** Clauses per program at most (default 2). */
  readonly maxClauses?: number
  /** Allow the target in clause bodies (default true). */
  readonly recursion?: boolean
}

/** Which constraints the learner adds (all by default; all off is plain generate-and-test), and its limits. */
export interface LffOptions {
  /** Prune the generalisations of a program that proves a negative (default true). */
  readonly generalisation?: boolean
  /** Prune the specialisations of a program that misses a positive (default true). */
  readonly specialisation?: boolean
  /** Prune separable programs with a specialisation of a clause of a program that proves no positive (default true). */
  readonly elimination?: boolean
  /** SLD steps per example at most (default 2000); a proof cut off counts as a failure. */
  readonly maxProofSteps?: number
  /** SLD depth at most (default 40). */
  readonly maxProofDepth?: number
  /** Tests at most (default 500). */
  readonly maxTests?: number
}

/** A body or head literal: a predicate and its variables (ids 0, 1, …; the head uses 0 to arity $- 1$). */
export interface LffLiteral {
  /** The predicate's name. */
  readonly pred: string
  /** The variable ids of its arguments. */
  readonly args: readonly number[]
}

/** A clause of the hypothesis space. */
export interface LffClause {
  /** The body literals, in a canonical order. */
  readonly body: readonly LffLiteral[]
  /** Literals including the head. */
  readonly size: number
  /** Whether the target occurs in the body. */
  readonly recursive: boolean
  /** As Prolog text, the body in proof order. */
  readonly text: string
}

/** A program of the hypothesis space. */
export interface LffProgram {
  /** Its clauses, as indices into the clauses in increasing order. */
  readonly clauses: readonly number[]
  /** Literals in all its clauses, heads included. */
  readonly size: number
  /** Whether a clause is recursive. */
  readonly recursive: boolean
}

/** The kinds of constraint. */
export type LffConstraint = 'generalisation' | 'specialisation' | 'elimination'

/** One generate-test-constrain cycle. */
export interface LffStep {
  /** The program tested (an index into `programs`). */
  readonly program: number
  /** Which positives it proves. */
  readonly positives: readonly boolean[]
  /** Which negatives it proves. */
  readonly negatives: readonly boolean[]
  /** `all`, `some` or `none` of the positives proved; `none` or `some` of the negatives. */
  readonly outcome: { readonly positive: 'all' | 'some' | 'none'; readonly negative: 'none' | 'some' }
  /** The constraints learned from it. */
  readonly constraints: readonly LffConstraint[]
  /** Programs newly pruned by each constraint (a program pruned by two counts for the first). */
  readonly pruned: Readonly<Record<LffConstraint, number>>
  /** Programs neither tested nor pruned after this step. */
  readonly remaining: number
  /** Whether the program proves every positive and no negative. */
  readonly solution: boolean
}

/** The outcome of `learnFromFailures`. */
export interface LffResult {
  /** The clauses of the hypothesis space. */
  readonly clauses: readonly LffClause[]
  /** The programs of the hypothesis space, in generation order. */
  readonly programs: readonly LffProgram[]
  /** `subsumes[i][j]`: clause $i$ $\theta$-subsumes clause $j$. */
  readonly subsumes: readonly (readonly boolean[])[]
  /** The tests, in order. */
  readonly steps: readonly LffStep[]
  /** For each program: the step that tested it, or $-1$. */
  readonly testedAt: readonly number[]
  /** For each program: the step whose constraint pruned it, or $-1$. */
  readonly prunedAt: readonly number[]
  /** For each program: the constraint that pruned it, or null. */
  readonly prunedBy: readonly (LffConstraint | null)[]
  /** The solution's program index, or $-1$. */
  readonly solution: number
  /** Programs up to the solution in the generation order: the tests plain generate-and-test would make. */
  readonly enumerateTests: number
}

/**
 * The name of variable `i`: A to Z, then `V26`, `V27`, ….
 *
 * @param i The variable's id.
 * @returns Its name.
 */
const varName = (i: number) => (i < 26 ? String.fromCharCode(65 + i) : `V${i}`)

/**
 * A literal as Prolog text.
 *
 * @param l The literal.
 * @returns The text, such as `head(A, B)`.
 */
const literalText = (l: LffLiteral) => `${l.pred}(${l.args.map(varName).join(', ')})`

/**
 * A literal's identity as a string, for sorting and deduplication.
 *
 * @param l The literal.
 * @returns The key.
 */
const literalKey = (l: LffLiteral) => `${l.pred}/${l.args.join(',')}`

/**
 * The body in proof order: starting from the head's first argument, repeatedly the literal with the most variables
 * already bound, background predicates before the target, so recursive calls come after the literals that bind them.
 *
 * @param target The target predicate.
 * @param body The body literals.
 * @returns The literals reordered.
 */
function proofOrder(target: LffPredicate, body: readonly LffLiteral[]): LffLiteral[] {
  const bound = new Set<number>([0])
  const left = [...body]
  const out: LffLiteral[] = []
  while (left.length) {
    let best = 0
    let bestScore = -Infinity
    left.forEach((l, i) => {
      const score = 2 * l.args.filter((a) => bound.has(a)).length - (l.pred === target.name ? 1 : 0)
      if (score > bestScore) {
        best = i
        bestScore = score
      }
    })
    const [l] = left.splice(best, 1)
    l.args.forEach((a) => bound.add(a))
    out.push(l)
  }
  return out
}

/**
 * A clause of the target with the given body as Prolog text, the body in proof order.
 *
 * @param target The target predicate.
 * @param body The body literals.
 * @returns The text.
 */
function clauseText(target: LffPredicate, body: readonly LffLiteral[]): string {
  const head = literalText({ pred: target.name, args: Array.from({ length: target.arity }, (_, i) => i) })
  return body.length ? `${head} :- ${proofOrder(target, body).map(literalText).join(', ')}.` : `${head}.`
}

/**
 * The $k$-subsets of $\{0, \dots, n - 1\}$ in lexicographic order.
 *
 * @param n The size of the set.
 * @param k The size of the subsets.
 * @param start The smallest element allowed.
 * @returns A generator of the subsets, each increasing.
 */
function* combinations(n: number, k: number, start = 0): Generator<number[]> {
  if (k === 0) {
    yield []
    return
  }
  for (let i = start; i <= n - k; i++) for (const rest of combinations(n, k - 1, i + 1)) yield [i, ...rest]
}

/**
 * Every tuple of `length` values in $\{0, \dots, \text{values} - 1\}$, the last varying fastest.
 *
 * @param values The number of values.
 * @param length The tuple length.
 * @returns A generator of the tuples.
 */
function* tuples(values: number, length: number): Generator<number[]> {
  if (length === 0) {
    yield []
    return
  }
  for (const rest of tuples(values, length - 1)) for (let v = 0; v < values; v++) yield [...rest, v]
}

/**
 * The canonical form of a body: the non-head variables renamed by the permutation that makes the sorted body's text
 * smallest, so two renamings of one clause are the same.
 *
 * @param body The body literals.
 * @param arity The target's arity (head variables are kept).
 * @param maxVars The number of variables.
 * @returns The canonical body, sorted.
 */
function canonical(body: readonly LffLiteral[], arity: number, maxVars: number): LffLiteral[] {
  const extra = Array.from({ length: maxVars - arity }, (_, i) => arity + i)
  const perms: number[][] = []
  const permute = (rest: number[], acc: number[]) => {
    if (!rest.length) perms.push(acc)
    rest.forEach((v, i) => permute([...rest.slice(0, i), ...rest.slice(i + 1)], [...acc, v]))
  }
  permute(extra, [])
  let best: LffLiteral[] | null = null
  let bestKey = ''
  for (const p of perms) {
    const map = (a: number) => (a < arity ? a : p[a - arity])
    const renamed = body
      .map((l) => ({ pred: l.pred, args: l.args.map(map) }))
      .sort((x, y) => (literalKey(x) < literalKey(y) ? -1 : 1))
    const key = renamed.map(literalKey).join(';')
    if (best === null || key < bestKey) {
      best = renamed
      bestKey = key
    }
  }
  return best ?? [...body]
}

/**
 * The clauses of the hypothesis space, smallest first: every clause the bias allows, up to renaming of variables.
 * Throws `DomainError` when `maxVars` is below the target's arity.
 *
 * @param problem The target and the body predicates (the background and examples are not read).
 * @param bias The bounds on variables, body literals and recursion (see `LffBias`).
 * @returns The clauses, ordered by size and then by text.
 *
 * @example The clauses of a tiny space
 * const problem = {
 *   background: '',
 *   target: { name: 'p', arity: 1 },
 *   body: [{ name: 'q', arity: 1 }, { name: 'r', arity: 2 }],
 *   positives: [],
 *   negatives: [],
 * }
 * print(lffClauses(problem, { maxVars: 2, maxBody: 2, recursion: false }).map((c) => c.text))
 */
export function lffClauses(problem: LffProblem, bias: LffBias = {}): LffClause[] {
  const maxVars = bias.maxVars ?? 3
  const maxBody = bias.maxBody ?? 2
  const recursion = bias.recursion ?? true
  const { target } = problem
  if (maxVars < target.arity)
    throw new DomainError('lffClauses', `lffClauses: maxVars ${maxVars} is below the target's arity ${target.arity}`)
  const preds = recursion ? [...problem.body, target] : [...problem.body]
  const literals: LffLiteral[] = []
  for (const p of preds)
    for (const args of tuples(maxVars, p.arity)) {
      if (p.name === target.name && args.every((a, i) => a === i)) continue
      literals.push({ pred: p.name, args })
    }
  const typeOf = new Map(preds.map((p) => [p.name, p.types]))
  const seen = new Set<string>()
  const out: LffClause[] = []
  for (let k = 1; k <= maxBody; k++)
    for (const pick of combinations(literals.length, k)) {
      const body = pick.map((i) => literals[i])
      const count = new Array<number>(maxVars).fill(0)
      for (const l of body) for (const a of l.args) count[a]++
      if (count.slice(0, target.arity).some((c) => c === 0)) continue
      if (count.slice(target.arity).some((c) => c === 1)) continue
      // Variables in use must be the first ones (A, B, C, … with no gap), else the clause is a renaming of another.
      const used = count.map((c, i) => c > 0 || i < target.arity)
      if (used.some((u, i) => !u && used.slice(i + 1).some(Boolean))) continue
      // Types: every variable has one type across the head and body.
      const types = new Map<number, string>()
      let typed = true
      const assign = (v: number, t: string | undefined) => {
        if (t === undefined) return
        const cur = types.get(v)
        if (cur !== undefined && cur !== t) typed = false
        types.set(v, t)
      }
      target.types?.forEach((t, i) => assign(i, t))
      for (const l of body) typeOf.get(l.pred)?.forEach((t, i) => assign(l.args[i], t))
      if (!typed) continue
      const canon = canonical(body, target.arity, maxVars)
      const key = canon.map(literalKey).join(';')
      if (seen.has(key)) continue
      seen.add(key)
      out.push({
        body: canon,
        size: canon.length + 1,
        recursive: canon.some((l) => l.pred === target.name),
        text: clauseText(target, canon),
      })
    }
  return out.sort((a, b) => a.size - b.size || (a.text < b.text ? -1 : a.text > b.text ? 1 : 0))
}

/**
 * A clause of the space as a `HornClause` of terms.
 *
 * @param target The target predicate.
 * @param c The clause.
 * @returns The clause with variables named and numbered by their ids.
 */
function hornClause(target: LffPredicate, c: LffClause): HornClause {
  const v = (a: number) => variable(varName(a), a)
  const head = compound(
    target.name,
    Array.from({ length: target.arity }, (_, i) => v(i)),
  )
  const body: Term[] = c.body.map((l) => compound(l.pred, l.args.map(v)))
  return { head, body }
}

/**
 * The $\theta$-subsumption matrix of the clauses: entry $(i, j)$ is true when clause $i$ subsumes clause $j$.
 *
 * @param target The head predicate of the clauses.
 * @param clauses The clauses, as `lffClauses` returns them.
 * @returns The matrix, true on the diagonal.
 *
 * @example A more general clause subsumes a more specific one
 * const target = { name: 'p', arity: 1 }
 * const body = [{ name: 'q', arity: 1 }, { name: 'r', arity: 1 }]
 * const problem = { background: '', target, body, positives: [], negatives: [] }
 * const clauses = lffClauses(problem, { maxVars: 1, maxBody: 2, recursion: false })
 * print(clauses.map((c) => c.text))
 * print(lffSubsumption(target, clauses))
 */
export function lffSubsumption(target: LffPredicate, clauses: readonly LffClause[]): boolean[][] {
  const horn = clauses.map((c) => hornClause(target, c))
  return horn.map((ci, i) => horn.map((cj, j) => i === j || thetaSubsumes(ci, cj)))
}

/**
 * The programs of the hypothesis space, smallest first: sets of 1 to `maxClauses` clauses, without a recursive program
 * lacking a non-recursive clause and without a clause that another clause of the program subsumes.
 *
 * @param clauses The clauses, as `lffClauses` returns them.
 * @param subsumes Their subsumption matrix, as `lffSubsumption` returns it.
 * @param maxClauses Clauses per program at most.
 * @returns The programs, ordered by size, then by number of clauses, then by clause indices.
 *
 * @example Programs of a tiny space
 * const target = { name: 'p', arity: 1 }
 * const body = [{ name: 'q', arity: 1 }, { name: 'r', arity: 1 }]
 * const problem = { background: '', target, body, positives: [], negatives: [] }
 * const clauses = lffClauses(problem, { maxVars: 1, maxBody: 2, recursion: false })
 * const programs = lffPrograms(clauses, lffSubsumption(target, clauses))
 * print(programs.map((p) => p.clauses.map((i) => clauses[i].text).join(' ')))
 */
export function lffPrograms(
  clauses: readonly LffClause[],
  subsumes: readonly (readonly boolean[])[],
  maxClauses = 2,
): LffProgram[] {
  const out: LffProgram[] = []
  for (let k = 1; k <= maxClauses; k++)
    for (const pick of combinations(clauses.length, k)) {
      const recursive = pick.some((i) => clauses[i].recursive)
      if (recursive && pick.every((i) => clauses[i].recursive)) continue
      if (pick.some((i) => pick.some((j) => i !== j && subsumes[i][j]))) continue
      out.push({ clauses: pick, size: pick.reduce((s, i) => s + clauses[i].size, 0), recursive })
    }
  return out.sort(
    (a, b) =>
      a.size - b.size ||
      a.clauses.length - b.clauses.length ||
      a.clauses.join(',').localeCompare(b.clauses.join(','), 'en', { numeric: true }),
  )
}

/**
 * Learn a program by generate, test and constrain (Cropper and Morel, 2021), recording every test (see the file
 * comment). Each test runs the program with the background knowledge on the Prolog engine.
 *
 * @param problem The background knowledge, the target, the body predicates and the examples.
 * @param bias The bounds of the hypothesis space (see `LffBias`).
 * @param options Which constraints to learn, and the proof and test limits (see `LffOptions`).
 * @returns The space (clauses, programs, subsumption), the steps, when each program was tested or pruned, and the
 *   solution ($-1$ when the tests ran out first).
 *
 * @example Learn ancestor from parent, and compare with plain generate-and-test
 * const problem = {
 *   background: `parent(ann, bob). parent(ann, cid). parent(bob, dee).
 *     parent(bob, eve). parent(cid, fay). parent(eve, gus).`,
 *   target: { name: 'ancestor', arity: 2 },
 *   body: [{ name: 'parent', arity: 2 }],
 *   positives: ['ancestor(ann, bob)', 'ancestor(ann, dee)', 'ancestor(ann, gus)', 'ancestor(cid, fay)'],
 *   negatives: ['ancestor(bob, ann)', 'ancestor(dee, bob)', 'ancestor(cid, gus)'],
 * }
 * const r = learnFromFailures(problem)
 * print(r.programs[r.solution].clauses.map((i) => r.clauses[i].text))
 * print(r.steps.length, 'tests; generate-and-test would make', r.enumerateTests)
 */
export function learnFromFailures(problem: LffProblem, bias: LffBias = {}, options: LffOptions = {}): LffResult {
  const useGen = options.generalisation ?? true
  const useSpec = options.specialisation ?? true
  const useElim = options.elimination ?? true
  const maxProofSteps = options.maxProofSteps ?? 2000
  const maxProofDepth = options.maxProofDepth ?? 40
  const maxTests = options.maxTests ?? 500
  const clauses = lffClauses(problem, bias)
  const subsumes = lffSubsumption(problem.target, clauses)
  const programs = lffPrograms(clauses, subsumes, bias.maxClauses ?? 2)
  const testedAt = new Array<number>(programs.length).fill(-1)
  const prunedAt = new Array<number>(programs.length).fill(-1)
  const prunedBy = new Array<LffConstraint | null>(programs.length).fill(null)
  const steps: LffStep[] = []
  let remaining = programs.length
  let solution = -1

  // T1 ⪰ T2: every clause of T2 is subsumed by a clause of T1.
  const generalises = (t1: readonly number[], t2: readonly number[]) =>
    t2.every((c2) => t1.some((c1) => subsumes[c1][c2]))

  // Which positives and negatives the program proves; the program is built once for both.
  const proves = (program: LffProgram) => {
    const text = `${problem.background}\n${program.clauses.map((i) => clauses[i].text).join('\n')}`
    const p = prologProgram(text)
    const test = (e: string) =>
      solveQuery(p, e, { maxSolutions: 1, maxSteps: maxProofSteps, maxDepth: maxProofDepth }).solutions.length > 0
    return { pos: problem.positives.map(test), neg: problem.negatives.map(test) }
  }

  for (let next = 0; next < programs.length && steps.length < maxTests; next++) {
    if (testedAt[next] >= 0 || prunedAt[next] >= 0) continue
    const t = steps.length
    const h = programs[next]
    testedAt[next] = t
    remaining--
    const { pos, neg } = proves(h)
    const nPos = pos.filter(Boolean).length
    const positive = nPos === pos.length ? 'all' : nPos === 0 ? 'none' : 'some'
    const negative = neg.some(Boolean) ? 'some' : 'none'
    const isSolution = positive === 'all' && negative === 'none'
    const constraints: LffConstraint[] = []
    if (!isSolution) {
      if (negative === 'some' && useGen) constraints.push('generalisation')
      if (positive !== 'all' && useSpec) constraints.push('specialisation')
      if (positive === 'none' && useElim) constraints.push('elimination')
    }
    const pruned: Record<LffConstraint, number> = { generalisation: 0, specialisation: 0, elimination: 0 }
    for (let j = 0; j < programs.length && constraints.length; j++) {
      if (testedAt[j] >= 0 || prunedAt[j] >= 0) continue
      const g = programs[j]
      for (const c of constraints) {
        const hit =
          c === 'generalisation'
            ? generalises(g.clauses, h.clauses)
            : c === 'specialisation'
              ? generalises(h.clauses, g.clauses)
              : !g.recursive && g.clauses.some((cg) => h.clauses.some((ch) => subsumes[ch][cg]))
        if (hit) {
          prunedAt[j] = t
          prunedBy[j] = c
          pruned[c]++
          remaining--
          break
        }
      }
    }
    steps.push({
      program: next,
      positives: pos,
      negatives: neg,
      outcome: { positive, negative },
      constraints,
      pruned,
      remaining,
      solution: isSolution,
    })
    if (isSolution) {
      solution = next
      break
    }
  }
  return {
    clauses,
    programs,
    subsumes,
    steps,
    testedAt,
    prunedAt,
    prunedBy,
    solution,
    enumerateTests: solution >= 0 ? solution + 1 : programs.length,
  }
}
