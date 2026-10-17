/**
 * Meta-interpretive learning: a Metagol-style learner of dyadic Datalog programs, with predicate invention and every
 * event of its search recorded.
 *
 * Muggleton, Lin and Tamaddoni-Nezhad (2015), "Meta-interpretive learning of higher-order dyadic datalog: predicate
 * invention revisited", Machine Learning 100, §3; Cropper and Muggleton's Metagol. The learner proves the positive
 * examples with a meta-interpreter. A goal whose predicate is a background predicate is proved by its facts. A goal
 * whose predicate is the target or an invented predicate is proved by a clause already in the program, or by a new
 * clause: a metarule, such as chain $P(A, B) \leftarrow Q(A, C), R(C, B)$, with $P$ bound to the goal's predicate and
 * the body's predicate variables bound while the body is proved. Each new clause is a metasubstitution added to the
 * program. The search is depth first with backtracking, under iterative deepening on the number of clauses
 * ($1, 2, \dots$ up to `maxClauses`); a program that proves every positive is accepted only if it proves no negative,
 * else the search backtracks for another.
 *
 * Differences from Metagol: instead of Metagol's ordering constraints on predicate symbols, a depth bound on the
 * proof, a ban on a clause whose first body predicate is its own head (left recursion) and on an ident clause calling
 * its own head keep the search finite; a binding that makes a clause a copy of another is skipped. Body predicate
 * variables are bound to background predicates first, then to the target and the invented predicates, then to a new
 * invented predicate (`inv1`, `inv2`, …). Problems and results are plain data, so `metagol` can run in a worker.
 */

/** A first-order argument: a constant (`c`) or a variable (by number, `v`). */
export type Argument = { readonly c: string } | { readonly v: number }

/** A predicate position: a predicate symbol (`p`) or a predicate variable (by number, `pv`). */
export type PredicateTerm = { readonly p: string } | { readonly pv: number }

/** A dyadic atom $P(a, b)$. */
export interface DyadicAtom {
  /** The predicate. */
  readonly pred: PredicateTerm
  /** The two arguments. */
  readonly args: readonly [Argument, Argument]
}

/**
 * A metarule $P(A, B) \leftarrow \dots$ over predicate variables 0, 1, 2 ($P$ is 0) and first-order variables 0
 * ($A$), 1 ($B$), 2 ($C$), ….
 */
export interface Metarule {
  /** Its name, as `MilProblem.metarules` lists it. */
  readonly name: string
  /** The body atoms: each a predicate variable and two first-order variables. */
  readonly body: readonly { readonly pred: number; readonly args: readonly [number, number] }[]
}

/**
 * The standard metarules over dyadic predicates: ident $P(A, B) \leftarrow Q(A, B)$, inverse
 * $P(A, B) \leftarrow Q(B, A)$ and chain $P(A, B) \leftarrow Q(A, C), R(C, B)$.
 */
export const METARULES: Readonly<Record<string, Metarule>> = {
  ident: { name: 'ident', body: [{ pred: 1, args: [0, 1] }] },
  inverse: { name: 'inverse', body: [{ pred: 1, args: [1, 0] }] },
  chain: {
    name: 'chain',
    body: [
      { pred: 1, args: [0, 2] },
      { pred: 2, args: [2, 1] },
    ],
  },
}

/** A meta-interpretive learning problem. */
export interface MilProblem {
  /** Background facts by predicate: pairs of constants. */
  readonly background: Readonly<Record<string, readonly (readonly [string, string])[]>>
  /** The target predicate. */
  readonly target: string
  /** The positive examples: pairs the target must hold of. */
  readonly positives: readonly (readonly [string, string])[]
  /** The negative examples: pairs the target must not hold of. */
  readonly negatives: readonly (readonly [string, string])[]
  /** The metarules, by name (keys of `METARULES`), in the order they are tried. */
  readonly metarules: readonly string[]
}

/** Options of `metagol`. */
export interface MilOptions {
  /** Clauses in the program at most (default 3). */
  maxClauses?: number
  /** Invented predicates at most (default 1). */
  maxInvented?: number
  /** Nested proof depth at most (default 8). */
  maxDepth?: number
  /** Events recorded at most (default 3000); the search stops there with `stopped: 'limit'`. */
  maxEvents?: number
}

/** One event of the search. */
export interface MilEvent {
  /**
   * What happened: a new clause bound (`bound`), a background `fact` used or none (`fail`), a program clause tried
   * (`reuse`), a `new-clause` from a metarule, an open predicate bound (`bind`) or invented (`invent`), a program
   * rejected for proving a negative (`reject`), accepted (`accept`), or no program found (`none`).
   */
  readonly kind: 'bound' | 'fact' | 'fail' | 'reuse' | 'new-clause' | 'bind' | 'invent' | 'reject' | 'accept' | 'none'
  /** The goal being proved, as text (empty for bound, reject, accept and none). */
  readonly goal: string
  /** The program at this point, one clause per line. */
  readonly program: readonly string[]
  /** What happened, in words. */
  readonly note: string
  /** The clause bound in use (iterative deepening). */
  readonly clauses: number
}

/** The outcome of `metagol`. */
export interface MilResult {
  /** Every event of the search, in order. */
  readonly events: readonly MilEvent[]
  /** The program found, one clause per line as Prolog text, or null. */
  readonly program: readonly string[] | null
  /** `found`, `none` (no program within the bounds) or `limit` (the event limit was reached first). */
  readonly stopped: 'found' | 'none' | 'limit'
}

/** Bindings: first-order variables to constants and predicate variables to symbols, as two persistent maps. */
interface Env {
  /** First-order variables to constants. */
  readonly fo: ReadonlyMap<number, string>
  /** Predicate variables to predicate symbols. */
  readonly ho: ReadonlyMap<number, string>
}

/** A clause of the program being built: a metarule and a predicate term for each of its predicate variables. */
interface MetaClause {
  /** The metarule's name. */
  readonly metarule: string
  /** Predicate terms for $P$, $Q$, $R$, …. */
  readonly preds: readonly PredicateTerm[]
}

/**
 * The constant an argument stands for, or null while it is an unbound variable.
 *
 * @param a The argument.
 * @param env The bindings.
 * @returns The constant or null.
 */
const argValue = (a: Argument, env: Env): string | null => ('c' in a ? a.c : (env.fo.get(a.v) ?? null))

/**
 * The predicate symbol a predicate term stands for, or null while it is an unbound variable.
 *
 * @param p The predicate term.
 * @param env The bindings.
 * @returns The symbol or null.
 */
const predValue = (p: PredicateTerm, env: Env): string | null => ('p' in p ? p.p : (env.ho.get(p.pv) ?? null))

/**
 * Bind an argument to a constant.
 *
 * @param a The argument.
 * @param value The constant.
 * @param env The bindings.
 * @returns The extended bindings (`env` itself when already bound to `value`), or null on a clash.
 */
function bindArg(a: Argument, value: string, env: Env): Env | null {
  const cur = argValue(a, env)
  if (cur !== null) return cur === value ? env : null
  if ('c' in a) return null
  const fo = new Map(env.fo)
  fo.set(a.v, value)
  return { fo, ho: env.ho }
}

/**
 * Bind a predicate term to a symbol (a symbol stays as it is).
 *
 * @param p The predicate term.
 * @param value The symbol.
 * @param env The bindings.
 * @returns The extended bindings.
 */
function bindPred(p: PredicateTerm, value: string, env: Env): Env {
  if ('p' in p) return env
  const ho = new Map(env.ho)
  ho.set(p.pv, value)
  return { fo: env.fo, ho }
}

/**
 * Learn a dyadic Datalog program with a meta-interpreter (Muggleton, Lin and Tamaddoni-Nezhad, 2015), recording
 * every event of the search (see the file comment). Iterative deepening returns a program with the fewest clauses.
 *
 * @param problem The background facts, the target, the examples and the metarules.
 * @param options The limits on clauses, invented predicates, proof depth and recorded events (see `MilOptions`).
 * @returns The events, the program found (as text, one clause per line) or null, and why the search stopped.
 *
 * @example Learn ancestor, a recursive program
 * const parent = [['ann', 'bob'], ['ann', 'cid'], ['bob', 'dee'], ['bob', 'eve'], ['cid', 'fay'], ['eve', 'gus']]
 * const r = metagol({
 *   background: { parent },
 *   target: 'ancestor',
 *   positives: [['ann', 'bob'], ['ann', 'dee'], ['ann', 'gus'], ['cid', 'fay']],
 *   negatives: [['bob', 'ann'], ['dee', 'bob'], ['cid', 'gus']],
 *   metarules: ['ident', 'chain'],
 * })
 * print(r.program)
 * print(r.events.length, 'events, stopped:', r.stopped)
 *
 * @example Great-grandparent with an invented predicate
 * const parent = [['ann', 'bob'], ['ann', 'cid'], ['bob', 'dee'], ['bob', 'eve'], ['cid', 'fay'], ['eve', 'gus']]
 * const r = metagol({
 *   background: { parent },
 *   target: 'greatgrandparent',
 *   positives: [['ann', 'gus']],
 *   negatives: [['ann', 'eve'], ['bob', 'gus'], ['ann', 'dee']],
 *   metarules: ['ident', 'chain'],
 * })
 * print(r.program)
 * print('rejected programs:', r.events.filter((e) => e.kind === 'reject').length)
 */
export function metagol(problem: MilProblem, options: MilOptions = {}): MilResult {
  const maxClauses = options.maxClauses ?? 3
  const maxInvented = options.maxInvented ?? 1
  const maxDepth = options.maxDepth ?? 8
  const maxEvents = options.maxEvents ?? 3000
  const prims = Object.keys(problem.background)
  const events: MilEvent[] = []
  let counter = 0
  let bound = 0
  let full = false
  // While positive, events are not recorded (the coverage tests of `proves`), so they never reach the limit.
  let muted = 0

  const showArg = (a: Argument, env: Env) => argValue(a, env) ?? `_${'v' in a ? a.v - 1000 : ''}`
  const showPred = (p: PredicateTerm, env: Env) => predValue(p, env) ?? `P${'pv' in p ? p.pv : ''}`
  const showAtom = (g: DyadicAtom, env: Env) =>
    `${showPred(g.pred, env)}(${showArg(g.args[0], env)}, ${showArg(g.args[1], env)})`
  const VARS = ['A', 'B', 'C', 'D']
  const showClause = (c: MetaClause, env: Env) => {
    const mr = METARULES[c.metarule]
    const p = (i: number) => showPred(c.preds[i], env)
    const body = mr.body.map((b) => `${p(b.pred).replace(/^P\d+$/, '?')}(${VARS[b.args[0]]}, ${VARS[b.args[1]]})`)
    return `${p(0)}(A, B) :- ${body.join(', ')}.`
  }
  const showProgram = (prog: readonly MetaClause[], env: Env) => prog.map((c) => showClause(c, env))
  const emit = (kind: MilEvent['kind'], goal: string, prog: readonly MetaClause[], env: Env, note: string) => {
    if (muted > 0) return
    if (events.length >= maxEvents) {
      full = true
      return
    }
    events.push({ kind, goal, program: showProgram(prog, env), note, clauses: bound })
  }

  const learnt = (env: Env, prog: readonly MetaClause[], invented: number) =>
    [problem.target, ...Array.from({ length: invented }, (_, i) => `inv${i + 1}`)].filter(
      (p) => p === problem.target || prog.some((c) => predValue(c.preds[0], env) === p),
    )

  type State = { env: Env; prog: readonly MetaClause[]; invented: number }

  // Prove the goals left to right; yield every state in which all are proved.
  function* proveAll(goals: readonly DyadicAtom[], s: State, depth: number): Generator<State> {
    if (full) return
    if (goals.length === 0) {
      yield s
      return
    }
    for (const s1 of proveOne(goals[0], s, depth)) yield* proveAll(goals.slice(1), s1, depth)
  }

  function* proveFacts(g: DyadicAtom, pred: string, s: State): Generator<State> {
    let any = false
    for (const [a, b] of problem.background[pred]) {
      const e1 = bindArg(g.args[0], a, s.env)
      const e2 = e1 && bindArg(g.args[1], b, e1)
      if (!e2) continue
      const env = bindPred(g.pred, pred, e2)
      any = true
      emit('fact', showAtom(g, env), s.prog, env, `${pred}(${a}, ${b}) is a background fact.`)
      yield { ...s, env }
    }
    if (!any) emit('fail', showAtom(g, bindPred(g.pred, pred, s.env)), s.prog, s.env, `No background fact matches.`)
  }

  // Instantiate a clause's metarule for goal g: A and B are the goal's arguments, fresh variables for the rest.
  function instantiate(c: MetaClause, g: DyadicAtom): DyadicAtom[] {
    const mr = METARULES[c.metarule]
    const fresh = [-1, -1, counter++, counter++]
    const local: Argument[] = [g.args[0], g.args[1], { v: 1000 + fresh[2] }, { v: 1000 + fresh[3] }]
    return mr.body.map((b) => ({
      pred: c.preds[b.pred],
      args: [local[b.args[0]], local[b.args[1]]] as [Argument, Argument],
    }))
  }

  function* proveDefined(g: DyadicAtom, pred: string, s: State, depth: number): Generator<State> {
    const env0 = bindPred(g.pred, pred, s.env)
    // Clauses already in the program.
    for (const [k, c] of s.prog.entries()) {
      if (predValue(c.preds[0], s.env) !== pred) continue
      emit('reuse', showAtom(g, env0), s.prog, env0, `Try clause ${k + 1}: ${showClause(c, env0)}`)
      yield* proveAll(instantiate(c, g), { ...s, env: env0 }, depth + 1)
    }
    // A new clause from each metarule.
    if (s.prog.length >= bound) return
    for (const name of problem.metarules) {
      const mr = METARULES[name]
      const nPreds = 1 + Math.max(...mr.body.map((b) => b.pred))
      const preds: PredicateTerm[] = [{ p: pred }]
      for (let i = 1; i < nPreds; i++) preds.push({ pv: counter++ })
      const c: MetaClause = { metarule: name, preds }
      const prog = [...s.prog, c]
      emit(
        'new-clause',
        showAtom(g, env0),
        prog,
        env0,
        `Add clause ${prog.length} from the ${name} metarule, its body predicates still open.`,
      )
      yield* proveAll(instantiate(c, g), { ...s, env: env0, prog }, depth + 1)
    }
  }

  function* proveOne(g: DyadicAtom, s: State, depth: number): Generator<State> {
    if (full) return
    if (depth > maxDepth) {
      emit('fail', showAtom(g, s.env), s.prog, s.env, `Proof depth limit ${maxDepth} reached.`)
      return
    }
    const known = predValue(g.pred, s.env)
    if (known !== null) {
      if (prims.includes(known)) yield* proveFacts(g, known, s)
      else yield* proveDefined(g, known, s, depth)
      return
    }
    // An open body predicate: background predicates, then learnt ones, then a new invented predicate.
    const owner = s.prog.find((c) => c.preds.some((p) => 'pv' in p && p === g.pred))
    const head = owner ? predValue(owner.preds[0], s.env) : null
    const firstOfOwner =
      owner !== undefined &&
      METARULES[owner.metarule].body.length > 1 &&
      owner.preds[METARULES[owner.metarule].body[0].pred] === g.pred
    // A binding that makes the owning clause a copy of another clause of the program is skipped.
    const duplicate = (env: Env) =>
      owner !== undefined &&
      s.prog.some(
        (c) =>
          c !== owner &&
          c.metarule === owner.metarule &&
          c.preds.every((p, i) => {
            const a = predValue(p, env)
            return a !== null && a === predValue(owner.preds[i], env)
          }),
      )
    for (const p of prims) {
      const env = bindPred(g.pred, p, s.env)
      if (duplicate(env)) continue
      emit('bind', showAtom(g, env), s.prog, env, `Bind the open predicate to the background predicate ${p}.`)
      yield* proveFacts(g, p, { ...s, env })
    }
    for (const p of learnt(s.env, s.prog, s.invented)) {
      if (firstOfOwner && p === head) continue
      if (owner && METARULES[owner.metarule].body.length === 1 && p === head) continue
      const env = bindPred(g.pred, p, s.env)
      if (duplicate(env)) continue
      emit('bind', showAtom(g, env), s.prog, env, `Bind the open predicate to ${p}, so the clause calls ${p}.`)
      yield* proveDefined(g, p, { ...s, env }, depth)
    }
    if (s.invented < maxInvented && s.prog.length < bound) {
      const p = `inv${s.invented + 1}`
      const env = bindPred(g.pred, p, s.env)
      emit('invent', showAtom(g, env), s.prog, env, `Invent a new predicate ${p} and define it by a new clause.`)
      yield* proveDefined(g, p, { ...s, env, invented: s.invented + 1 }, depth)
    }
  }

  // Does the program prove a ground goal (no new clauses)? Events are not recorded, and the event limit's state is
  // left as it was.
  function proves(goal: readonly [string, string], s: State): boolean {
    const saved = bound
    const savedFull = full
    bound = 0
    muted++
    const g: DyadicAtom = { pred: { p: problem.target }, args: [{ c: goal[0] }, { c: goal[1] }] }
    const ok = !proveAll([g], s, 0).next().done
    muted--
    full = savedFull
    bound = saved
    return ok
  }

  const goals: DyadicAtom[] = problem.positives.map(([a, b]) => ({
    pred: { p: problem.target },
    args: [{ c: a }, { c: b }],
  }))
  const empty: Env = { fo: new Map(), ho: new Map() }
  for (bound = 1; bound <= maxClauses; bound++) {
    emit('bound', '', [], empty, `Search for programs of at most ${bound} clause${bound > 1 ? 's' : ''}.`)
    for (const s of proveAll(goals, { env: empty, prog: [], invented: 0 }, 0)) {
      const covered = problem.negatives.find((n) => proves(n, s))
      if (covered) {
        emit(
          'reject',
          '',
          s.prog,
          s.env,
          `The program proves every positive but also the negative ${problem.target}(${covered[0]}, ${covered[1]}): backtrack.`,
        )
        continue
      }
      const program = showProgram(s.prog, s.env)
      emit('accept', '', s.prog, s.env, 'The program proves every positive and no negative.')
      return { events, program, stopped: 'found' }
    }
    if (full) return { events, program: null, stopped: 'limit' }
  }
  emit('none', '', [], empty, `No program of at most ${maxClauses} clauses proves the positives and no negative.`)
  return { events, program: null, stopped: 'none' }
}
