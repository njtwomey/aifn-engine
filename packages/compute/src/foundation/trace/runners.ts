import type {
  Algorithm,
  Checkpoints,
  Index,
  Key,
  Recorded,
  Recorder,
  Size,
  Status,
  StopReason,
  Stream,
  Trace,
  TraceOptions,
} from 'aifn-compute/foundation/contracts'
import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { initStream, rootKey, stepContext, stopReason } from './protocol'
import { flattenRecorded, makeSeries, seriesData } from './series'

/** A monotone clock in milliseconds: `performance.now()` where it exists (browsers, workers, Node), else `Date.now()`. */
export const now: () => number =
  typeof globalThis.performance?.now === 'function' ? () => globalThis.performance.now() : () => Date.now()

// ---------------------------------------------------------------------------------------------------------------------
// Profiling. A step runs synchronously, so a module-level "current phases" record is enough: the runner sets it around
// each `step` call and `profile` adds to it. Outside a traced step, `profile` just calls `fn`.

let currentPhases: Record<string, number> | null = null

/**
 * Times `fn` as the named phase of the current step: inside a traced step it adds the elapsed milliseconds to
 * `trace.timing.phases[name]`; elsewhere it only calls `fn`. Nested phases each count their own inclusive time.
 */
export function profile<T>(name: string, fn: () => T): T {
  const phases = currentPhases
  if (!phases) return fn()
  const start = now()
  try {
    return fn()
  } finally {
    phases[name] = (phases[name] ?? 0) + (now() - start)
  }
}

/** Options of the runners without history: the root stream (only its key is used; default `stream(0)`). */
export type RunOptions = { stream?: Stream }

// ---------------------------------------------------------------------------------------------------------------------
// Runners without history.

/**
 * Runs `alg` for at most `n` steps and returns the final state. Stops early on a `Status` flag or `done`, so
 * `run(alg, start, n)` is the state a trace of `n` steps ends on.
 */
export function run<Start, S extends Status>(
  alg: Algorithm<Start, S>,
  start: Start,
  n: Size,
  options: RunOptions = {},
): S {
  const key = rootKey(options.stream)
  let state = alg.init(start, initStream(key))
  for (let t = 0; t < n && !stopReason(alg, state); t++) state = alg.step(state, stepContext(key, t))
  return state
}

/**
 * The state at step `i`, equal to `run(alg, start, i)` with the same root stream. With `checkpoints` (a trace, or
 * stored checkpoints), it starts from the latest stored state at or before step `i` rather than from `init`, so
 * scrubbing a long run is cheap; a trace also supplies its root key.
 */
export function seek<Start, S extends Status>(
  alg: Algorithm<Start, S>,
  start: Start,
  i: Index,
  options: { checkpoints?: Trace<S> | Checkpoints<S>; stream?: Stream } = {},
): S {
  const c = options.checkpoints
  // A trace's states were drawn with its own key: continuing them with another stream would mix two runs.
  if (options.stream && c && 'meta' in c && options.stream.key.path !== c.meta.key.path)
    throw new AifnError(
      'seek',
      `seek: the stream (${options.stream.key.path}) is not the trace's (${c.meta.key.path}); omit it to use the trace's`,
    )
  const key = options.stream ? options.stream.key : c && 'meta' in c ? c.meta.key : rootKey(undefined)
  let t = 0
  let state: S | undefined
  const sources: Checkpoints<S>[] = []
  if (c) {
    if ('meta' in c) {
      sources.push(c.checkpoints, { index: [c.meta.steps], states: [c.final] })
      if (c.steps.length === c.index.length) sources.push({ index: Array.from(c.index), states: c.steps })
    } else sources.push(c)
  }
  for (const source of sources) {
    // Indices are ascending: take the last one at or before i.
    for (let k = source.index.length - 1; k >= 0; k--) {
      if (source.index[k] <= i) {
        if (state === undefined || source.index[k] > t) {
          t = source.index[k]
          state = source.states[k]
        }
        break
      }
    }
  }
  if (state === undefined) state = alg.init(start, initStream(key))
  for (; t < i && !stopReason(alg, state); t++) state = alg.step(state, stepContext(key, t))
  return state
}

/**
 * A generator of `{ step, state }` for play loops, starting at step 0. It ends after yielding a state that stops the
 * run (with `stopped` set); otherwise it runs for as long as it is pulled.
 */
export function* live<Start, S extends Status>(
  alg: Algorithm<Start, S>,
  start: Start,
  options: RunOptions = {},
): Generator<{ step: Index; state: S; stopped?: StopReason }, void, unknown> {
  const key = rootKey(options.stream)
  let state = alg.init(start, initStream(key))
  for (let step = 0; ; step++) {
    const stopped = stopReason(alg, state)
    if (stopped) {
      yield { step, state, stopped }
      return
    }
    yield { step, state }
    state = alg.step(state, stepContext(key, step))
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Growable columns. Rows are appended to a typed-array buffer that doubles when full; a snapshot is a view of the
// filled prefix, so it costs O(1). Appending never writes inside a prefix that a snapshot has seen, with one exception
// handled by copy-on-write: a snapshot may show one provisional row past the committed rows (the final state off the
// `every` grid), and the slot it used is copied away before it is overwritten.

type ArrayCtor = Float64ArrayConstructor | Int32ArrayConstructor

class Rows<A extends Float64Array | Int32Array> {
  data: A
  rows = 0
  /** Rows (committed or provisional) that some snapshot's view covers in the current buffer. */
  private lent = 0

  private readonly make: ArrayCtor
  width: number

  constructor(make: ArrayCtor, width: number) {
    this.make = make
    this.width = width
    this.data = new make(16 * Math.max(1, width)) as A
  }

  /** Make the slot of row `rows` writable: grow the buffer if full, or copy it if a snapshot covers the slot. */
  private prepare(): void {
    const need = (this.rows + 1) * this.width
    if (need > this.data.length) {
      const bigger = new this.make(Math.max(need, 2 * this.data.length)) as A
      bigger.set(this.data.subarray(0, this.rows * this.width))
      this.data = bigger
      this.lent = 0
    } else if (this.lent > this.rows) {
      this.data = this.data.slice() as A
      this.lent = 0
    }
  }

  push(values: ArrayLike<number>): void {
    this.prepare()
    this.data.set(values, this.rows * this.width)
    this.rows++
  }

  /** A view of the committed rows, plus `extra` as one provisional row when given. */
  view(extra?: ArrayLike<number>): A {
    let rows = this.rows
    if (extra) {
      this.prepare()
      this.data.set(extra, rows * this.width)
      rows++
    }
    this.lent = Math.max(this.lent, rows)
    return this.data.subarray(0, rows * this.width) as A
  }
}

type Column = { valueShape: number[] | null; rows: Rows<Float64Array>; finiteSeen: boolean }

/** A getter memoised on first read (and replaceable by assignment), so a snapshot copies no state arrays up front. */
function lazy<T extends object, K extends string, V>(target: T, name: K, compute: () => V): T & Record<K, V> {
  let value: V | undefined
  let done = false
  Object.defineProperty(target, name, {
    enumerable: true,
    configurable: true,
    get: () => {
      if (!done) {
        value = compute()
        done = true
      }
      return value
    },
    set: (v: V) => {
      value = v
      done = true
    },
  })
  return target as T & Record<K, V>
}

// ---------------------------------------------------------------------------------------------------------------------
// The trace builder: one incremental engine behind `trace`, `extend` and `timeSliced`.

/** Recorders a trace was made with, so `extend` can reuse them in the same thread. Functions are not part of a trace. */
const madeWith = new WeakMap<object, Record<string, Recorder<never>>>()

interface Builder<S> {
  /** Step until step `target`, a stop, or the clock passes `deadline` (at least one step per call). */
  advance(target: Size, deadline?: number): void
  readonly t: Size
  readonly stopped: StopReason | null
  /** The trace so far, in O(1); the current state is included as the final kept step. Does not change the builder. */
  snapshot(): Trace<S>
}

type Resume<S> = {
  columns: Map<string, Column>
  index: Rows<Int32Array>
  elapsed: Rows<Float64Array>
  stepMs: Rows<Float64Array>
  states: S[]
  checkpoints: { index: Index[]; states: S[] }
  phases: Record<string, number>
  totalMs: number
  elapsedOffset: number
}

function createBuilder<S extends Status>(
  alg: Algorithm<never, S>,
  options: TraceOptions<S>,
  begin: { state: S; t: Size; start: unknown; key: Key; initMs?: number; resume?: Resume<S> },
): Builder<S> {
  const every = Math.max(1, Math.floor(options.every ?? 1))
  const checkpointEvery = options.checkpointEvery ? Math.max(1, Math.floor(options.checkpointEvery)) : null
  const keep = options.keep ?? 'all'
  const timing = options.timing ?? 'step'
  const record = options.record ?? {}
  const recorders = Object.entries(record) as [string, Recorder<S>][]
  const stopOnNonFinite = options.stopOnNonFinite ?? true
  const key = begin.key

  let state = begin.state
  let t = begin.t
  const r = begin.resume
  const columns =
    r?.columns ??
    new Map<string, Column>(
      recorders.map(([name]) => [
        name,
        { valueShape: null, rows: new Rows<Float64Array>(Float64Array, 1), finiteSeen: false },
      ]),
    )
  const index = r?.index ?? new Rows<Int32Array>(Int32Array, 1)
  const elapsed = r?.elapsed ?? new Rows<Float64Array>(Float64Array, 1)
  const stepMs = r?.stepMs ?? new Rows<Float64Array>(Float64Array, 1)
  const states: S[] = r?.states ?? []
  const checkpoints = r?.checkpoints ?? { index: [], states: [] }
  const phases: Record<string, number> = r?.phases ?? (begin.initMs === undefined ? {} : { init: begin.initMs })
  let totalMs = r?.totalMs ?? 0
  const began = now() - (r?.elapsedOffset ?? 0)
  let stopped: StopReason | null = null

  /**
   * Runs the recorders on a state; returns one row per series, whether any value was not finite, and the columns
   * that saw a finite value (committed by `keepStep` only, so a provisional row shown by `snapshot` changes nothing).
   */
  function recordRow(
    s: S,
    step: Index,
  ): { rows: [Column, ArrayLike<number>][]; nonFinite: boolean; finiteIn: Column[] } {
    const startRecord = timing === 'step' ? now() : 0
    let nonFinite = false
    const rows: [Column, ArrayLike<number>][] = []
    const finiteIn: Column[] = []
    for (const [name, recorder] of recorders) {
      const { shape, values } = flattenRecorded(recorder(s, step) as Recorded, name)
      const column = columns.get(name)!
      if (column.valueShape) {
        if (column.valueShape.length !== shape.length || column.valueShape.some((d, i) => d !== shape[i]))
          throw new ShapeError(
            'trace',
            `trace: recorder "${name}" changed shape from [${column.valueShape}] to [${shape}] at step ${step}`,
          )
      } else {
        column.valueShape = shape
        column.rows.width = values.length
        column.rows.data = new Float64Array(16 * Math.max(1, values.length))
      }
      // Divergence: an infinity anywhere, or a NaN in a series that has already been finite. A NaN before a series has
      // had any finite value means "not defined yet" (a step size or trial before the first step), not divergence.
      let finite = false
      for (let i = 0; i < values.length; i++) {
        const v = values[i]
        if (v === Infinity || v === -Infinity || (Number.isNaN(v) && column.finiteSeen)) nonFinite = true
        else if (Number.isFinite(v)) finite = true
      }
      if (finite) finiteIn.push(column)
      rows.push([column, values])
    }
    if (timing === 'step') phases.record = (phases.record ?? 0) + (now() - startRecord)
    return { rows, nonFinite, finiteIn }
  }

  function keepStep(s: S, step: Index): boolean {
    const { rows, nonFinite, finiteIn } = recordRow(s, step)
    for (const column of finiteIn) column.finiteSeen = true
    if (keep === 'all') states.push(s)
    index.push([step])
    if (timing === 'step') elapsed.push([now() - began])
    for (const [column, values] of rows) column.rows.push(values)
    return nonFinite && stopOnNonFinite
  }

  function checkpoint(s: S, step: Index): void {
    if (keep === 'none') return
    checkpoints.index.push(step)
    checkpoints.states.push(s)
  }

  // A fresh start keeps step 0 and checkpoints it.
  if (!r) {
    checkpoint(state, t)
    if (keepStep(state, t)) stopped = 'diverged'
  }
  stopped ??= stopReason(alg, state)

  function stepOnce(): void {
    const ctx = stepContext(key, t)
    if (timing === 'step') {
      const previous = currentPhases
      currentPhases = phases
      const startStep = now()
      try {
        state = alg.step(state, ctx)
      } finally {
        currentPhases = previous
      }
      const ms = now() - startStep
      stepMs.push([ms])
      totalMs += ms
    } else if (timing === 'total') {
      const previous = currentPhases
      currentPhases = phases
      try {
        state = alg.step(state, ctx)
      } finally {
        currentPhases = previous
      }
    } else state = alg.step(state, ctx)
  }

  return {
    get t() {
      return t
    },
    get stopped() {
      return stopped
    },
    advance(target, deadline) {
      let first = true
      const startAdvance = timing === 'total' ? now() : 0
      while (t < target && !stopped) {
        if (!first && deadline !== undefined && now() >= deadline) break
        first = false
        stepOnce()
        t++
        if (checkpointEvery && t % checkpointEvery === 0) checkpoint(state, t)
        if (t % every === 0 && keepStep(state, t)) stopped = 'diverged'
        stopped ??= stopReason(alg, state)
      }
      if (timing === 'total') totalMs += now() - startAdvance
    },
    snapshot() {
      // The final state is always kept; when it is off the `every` grid it is a provisional row, shown here without
      // committing it, so extending a trace later keeps exactly the rows a longer run would keep.
      const rowsBefore = index.rows
      const committedLast = rowsBefore > 0 ? index.data[rowsBefore - 1] : -1
      let extra: { rows: [Column, ArrayLike<number>][] } | null = null
      let reason: StopReason = stopped ?? 'limit'
      if (committedLast !== t) {
        const { rows, nonFinite } = recordRow(state, t)
        extra = { rows }
        if (nonFinite && stopOnNonFinite) reason = 'diverged'
      }
      const kept = rowsBefore + (extra ? 1 : 0)
      const series: Record<string, Tensor> = {}
      for (const [name, column] of columns) {
        const extraValues = extra?.rows.find(([c]) => c === column)?.[1]
        series[name] = makeSeries([kept, ...(column.valueShape ?? [])], column.rows.view(extraValues))
      }
      const finalState = state
      const keptStates = states.length
      const stateCount = checkpoints.index.length
      const trace = {
        kind: 'trace' as const,
        index: index.view(extra ? [t] : undefined),
        series,
        final: finalState,
        timing: {
          stepMs: stepMs.view(),
          elapsedMs: timing === 'step' ? elapsed.view(extra ? [now() - began] : undefined) : new Float64Array(0),
          totalMs,
          perSecond: timing === false ? NaN : t === 0 ? 0 : totalMs > 0 ? t / (totalMs / 1000) : Infinity,
          phases: { ...phases },
        },
        meta: {
          algorithm: alg.name,
          stopped: reason,
          steps: t,
          every,
          checkpointEvery,
          keep,
          timing,
          start: begin.start,
          key,
          recorders: recorders.map(([name]) => name),
        },
      }
      lazy(trace, 'steps', () =>
        keep === 'all' ? (extra ? [...states.slice(0, keptStates), finalState] : states.slice(0, keptStates)) : [],
      )
      lazy(trace, 'checkpoints', () => ({
        index: checkpoints.index.slice(0, stateCount),
        states: checkpoints.states.slice(0, stateCount),
      }))
      const out = trace as unknown as Trace<S>
      madeWith.set(out, record as Record<string, Recorder<never>>)
      return out
    },
  }
}

function freshBuilder<Start, S extends Status>(
  alg: Algorithm<Start, S>,
  start: Start,
  options: TraceOptions<S>,
): Builder<S> {
  const key = rootKey(options.stream)
  const timed = (options.timing ?? 'step') === 'step'
  const startInit = timed ? now() : 0
  const state = alg.init(start, initStream(key))
  const initMs = timed ? now() - startInit : undefined
  return createBuilder(alg as Algorithm<never, S>, options, { state, t: 0, start, key, initMs })
}

/**
 * Runs `alg` for at most `n` steps and returns its trace: each recorder's values stacked over the kept steps (steps
 * divisible by `every`, and always the final step), the stored states (`keep`), timing (`timing`) and why it stopped:
 * `done` (a `converged` or `terminated` flag, or `done`), `limit`, or `diverged` (the flag, or a non-finite recording).
 */
export function trace<Start, S extends Status>(
  alg: Algorithm<Start, S>,
  start: Start,
  n: Size,
  options: TraceOptions<S> = {},
): Trace<S> {
  const builder = freshBuilder(alg, start, options)
  builder.advance(n)
  return builder.snapshot()
}

/**
 * Continues a trace by `m` more steps from its final state, as though it had been traced for `meta.steps + m` steps
 * in the first place (same kept steps, series, states and checkpoints; the same root key, so the same draws). A trace
 * that stopped `done` or `diverged` is returned unchanged. The recorders are `options.record`, or those the trace was
 * made with in this thread; either way they must be the ones named in `meta.recorders`. `every`, `checkpointEvery`,
 * `keep`, `timing` and the stream always come from the trace.
 */
export function extend<S extends Status>(
  previous: Trace<S>,
  alg: Algorithm<never, S>,
  m: Size,
  options: Pick<TraceOptions<S>, 'record' | 'stopOnNonFinite'> = {},
): Trace<S> {
  const meta = previous.meta
  if (meta.stopped !== 'limit' || m <= 0) return previous
  const record = options.record ?? (madeWith.get(previous) as Record<string, Recorder<S>> | undefined) ?? {}
  const names = Object.keys(record)
  if (names.length !== meta.recorders.length || names.some((n) => !meta.recorders.includes(n)))
    throw new AifnError(
      'trace',
      `trace: extend needs the same recorders as the trace (${meta.recorders.join(', ') || 'none'})`,
    )

  // Drop the provisional final row (the final state kept off the `every` grid); a longer run would not keep it.
  const last = previous.index.length - 1
  const keptRows = previous.index.length - (previous.index[last] % meta.every !== 0 ? 1 : 0)
  const columns = new Map<string, Column>()
  for (const name of meta.recorders) {
    const s = previous.series[name]
    const valueShape = s.shape.slice(1)
    const width = valueShape.reduce((a, b) => a * b, 1)
    const rows = new Rows<Float64Array>(Float64Array, width)
    const data = seriesData(s)
    for (let k = 0; k < keptRows; k++) rows.push(data.subarray(k * width, (k + 1) * width))
    let finiteSeen = false
    for (let i = 0; i < keptRows * width && !finiteSeen; i++) finiteSeen = Number.isFinite(data[i])
    columns.set(name, { valueShape, rows, finiteSeen })
  }
  const column = (values: ArrayLike<number>, count: number, make: ArrayCtor) => {
    const rows = new Rows<Float64Array | Int32Array>(make, 1)
    for (let k = 0; k < count; k++) rows.push([values[k]])
    return rows
  }
  const timed = meta.timing === 'step'
  const builder = createBuilder(
    alg,
    {
      every: meta.every,
      checkpointEvery: meta.checkpointEvery ?? undefined,
      keep: meta.keep,
      timing: meta.timing,
      record,
      stopOnNonFinite: options.stopOnNonFinite,
    },
    {
      state: previous.final,
      t: meta.steps,
      start: meta.start,
      key: meta.key,
      resume: {
        columns,
        index: column(previous.index, keptRows, Int32Array) as Rows<Int32Array>,
        elapsed: column(previous.timing.elapsedMs, timed ? keptRows : 0, Float64Array) as Rows<Float64Array>,
        stepMs: column(previous.timing.stepMs, previous.timing.stepMs.length, Float64Array) as Rows<Float64Array>,
        states: meta.keep === 'all' ? previous.steps.slice(0, keptRows) : [],
        checkpoints: { index: [...previous.checkpoints.index], states: [...previous.checkpoints.states] },
        phases: { ...previous.timing.phases },
        totalMs: previous.timing.totalMs,
        elapsedOffset: timed ? (previous.timing.elapsedMs[previous.timing.elapsedMs.length - 1] ?? 0) : 0,
      },
    },
  )
  builder.advance(meta.steps + m)
  return builder.snapshot()
}

/**
 * Runs a trace in slices of about `budgetMs` of work each, yielding the partial trace after every slice and the full
 * trace last. Between slices it waits on `schedule(resume)`, which the caller supplies (for example
 * `requestAnimationFrame` in a page); the default is `setTimeout(resume, 0)`. Stop early by breaking out of the loop.
 * Each partial trace is a valid trace of the steps so far (`meta.stopped` is `limit` until the run ends), taken in O(1).
 */
export async function* timeSliced<Start, S extends Status>(
  alg: Algorithm<Start, S>,
  start: Start,
  n: Size,
  budgetMs: number,
  options: TraceOptions<S> & { schedule?: (resume: () => void) => void } = {},
): AsyncGenerator<Trace<S>, void, unknown> {
  const schedule = options.schedule ?? ((resume: () => void) => void setTimeout(resume, 0))
  const builder = freshBuilder(alg, start, options)
  for (;;) {
    builder.advance(n, now() + budgetMs)
    yield builder.snapshot()
    if (builder.t >= n || builder.stopped) return
    await new Promise<void>((resume) => schedule(resume))
  }
}

/**
 * A copy of a trace thinned to at most `maxPoints` kept steps, evenly spaced over the kept steps and always keeping
 * the first and last, for drawing. Series, `index`, stored states and `elapsedMs` are thinned together; `stepMs` and
 * the checkpoints are unchanged. Do not `extend` a decimated trace.
 */
export function decimate<S>(t: Trace<S>, maxPoints: Size): Trace<S> {
  const kept = t.index.length
  if (kept <= maxPoints || maxPoints < 2) return t
  const picks: number[] = []
  for (let k = 0; k < maxPoints; k++) {
    const p = Math.round((k * (kept - 1)) / (maxPoints - 1))
    if (p !== picks[picks.length - 1]) picks.push(p)
  }
  const series: Record<string, Tensor> = {}
  for (const [name, s] of Object.entries(t.series)) {
    const valueShape = s.shape.slice(1)
    const width = valueShape.reduce((a, b) => a * b, 1)
    const source = seriesData(s)
    const data = new Float64Array(picks.length * width)
    picks.forEach((p, r) => data.set(source.subarray(p * width, (p + 1) * width), r * width))
    series[name] = makeSeries([picks.length, ...valueShape], data)
  }
  const elapsed = t.timing.elapsedMs
  return {
    ...t,
    steps: t.steps.length === kept ? picks.map((p) => t.steps[p]) : t.steps,
    index: Int32Array.from(picks, (p) => t.index[p]),
    series,
    timing: {
      ...t.timing,
      elapsedMs: elapsed.length === kept ? Float64Array.from(picks, (p) => elapsed[p]) : elapsed,
    },
  }
}
