/**
 * `useComputed(fn, inputs, { mode })`: the one scheduler for derived computation (DESIGN.md §8a), in place of
 * per-widget debouncing.
 *
 * - **Latest wins, once per frame.** Input changes are coalesced to at most one run per animation frame, always on
 *   the latest inputs; a run never paints over a newer one. Nothing waits on a timer.
 * - **Measured.** Each run is timed. A run whose last time was under `inline` (default 2 ms) is computed in the
 *   render that sees the new inputs, so it costs no second render. Under the frame budget (default 8 ms) it is
 *   computed in the next frame and rendered before that frame paints. Over budget during a drag, the run moves after
 *   the paint (the handle and other `live` layers move first) and repeats as often as it can; meanwhile the last result
 *   is returned with `stale: true`, which derived layers show dimmed (`stale` prop on any layer). On release it always
 *   runs on the final inputs.
 * - **`mode: 'release'`** defers to the release outright while a pointer is held (a keyboard or button change still
 *   runs at once).
 * - **`mode: 'worker'`** runs an aifn computation in a Web Worker: `fn` returns a task (`call(address, ...args)` from
 *   `state/task.ts`: aifn addresses and plain inputs), `then` maps the worker's result on the page (cheap work, such as
 *   mapping points to plot coordinates), and `initial` is the value until the first answer. Latest wins: a newer task
 *   replaces a queued one, and a superseded job that runs too long is cancelled by terminating its worker
 *   (`ComputeWorker`). While a task is pending, `stale` is true.
 *
 * Outside worker mode the first value is computed synchronously on mount, so the first paint (and the server render
 * check) has a result.
 *
 *   const path = useComputed(() => gradientPath(start, lr), [start, lr])
 *   <Curve x={path.value.x} y={path.value.y} stale={path.stale} />
 *
 *   const run = useComputed(() => call<Trace<HmcState>>('foundation/trace/trace', …), [start, eps], {
 *     mode: 'worker', initial: [], then: (tr) => tr.steps.map(…) })
 */
import { useLayoutEffect, useReducer, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { isPointerHeld, onceReleased } from './pointer'
import type { Task } from './task'
import { ComputeWorker } from './worker'

export type ComputeMode = 'frame' | 'release' | 'worker'

export type ComputeOptions = {
  /** `frame` (default): every frame while fast, thinned while slow; `release`: on release during a drag. */
  mode?: 'frame' | 'release'
  /** The frame budget in ms: runs slower than this are treated as slow (default 8). */
  budget?: number
  /** Runs whose last time was under this (ms, default 2) are computed in the render itself. 0 turns it off. */
  inline?: number
}

export type WorkerComputeOptions<R, T> = {
  mode: 'worker'
  /** The value until the worker's first answer. */
  initial: T
  /** Maps the worker's result on the page (default: the result itself). */
  then?: (result: R) => T
  /** A superseded job running longer than this (ms, default 200) is cancelled by terminating the worker. */
  cancelAfter?: number
}

export type Computed<T> = {
  value: T
  /** True while `value` was computed from older inputs than the current ones and a newer run is on its way. */
  stale: boolean
  /** Duration of the last run, ms (in worker mode, measured in the worker). */
  ms: number
  /** Runs so far (for profiling readouts). */
  runs: number
  /** Whether the last run exceeded the budget. */
  slow: boolean
  /** The last worker job's error, if it failed (worker mode only; the value stays the last good one). */
  error?: string
}

type Result<T> = { value: T; inputs: readonly unknown[]; ms: number }

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now())
const same = (a: readonly unknown[], b: readonly unknown[]) =>
  a.length === b.length && a.every((v, i) => Object.is(v, b[i]))
/** Inputs no real inputs equal: worker mode's `initial` was computed from nothing. */
const NONE: readonly unknown[] = [Symbol('none')]

/** After the next paint: a frame, then a task (a MessageChannel post runs before timers clamp). */
function afterPaint(f: () => void): () => void {
  let cancelled = false
  const frame = requestAnimationFrame(() => {
    const channel = new MessageChannel()
    channel.port1.onmessage = () => {
      channel.port1.close()
      if (!cancelled) f()
    }
    channel.port2.postMessage(null)
  })
  return () => {
    cancelled = true
    cancelAnimationFrame(frame)
  }
}

/**
 * The scheduler state of one `useComputed`, kept across renders. `current` is the newest result; it is replaced in
 * render by an inline run, or by a scheduled run, which then asks the component to render (`commit`).
 */
class Job<T> {
  current: Result<T>
  runs = 1
  fn: () => unknown = () => undefined
  then: (r: unknown) => T = (r) => r as T
  inputs: readonly unknown[] = []
  mode: ComputeMode = 'frame'
  budget = 8
  slow = false
  lastMs = 0
  error: string | undefined
  /** Waiting for a release, a slow run's slot or the worker rather than for the next frame. */
  lagging = false
  render: (sync: boolean) => void
  worker: ComputeWorker | null = null
  cancelAfter = 200
  private cancel: (() => void) | null = null
  private releaseHook: (() => void) | null = null

  constructor(first: Result<T>, budget: number, render: (sync: boolean) => void) {
    this.current = first
    this.render = render
    this.lastMs = first.ms
    this.slow = first.ms > budget
  }

  private record(value: T, inputs: readonly unknown[], ms: number) {
    this.current = { value, inputs, ms }
    this.runs++
    this.slow = ms > this.budget
    this.lastMs = ms
  }

  /** Compute now, in the render that sees `inputs` (a fast run: no second render). */
  inline(fn: () => T, inputs: readonly unknown[]) {
    const t0 = now()
    const value = fn()
    this.record(value, inputs, now() - t0)
  }

  run(sync: boolean) {
    this.cancel = null
    this.lagging = false
    // An inline run in a later render may have answered these inputs already.
    if (same(this.current.inputs, this.inputs)) return
    const inputs = this.inputs
    const t0 = now()
    const value = this.fn() as T
    this.record(value, inputs, now() - t0)
    this.render(sync)
  }

  /** The newest computation, its inputs and its scheduling options. */
  configure(
    fn: () => unknown,
    then: (r: unknown) => T,
    inputs: readonly unknown[],
    options: { mode: ComputeMode; budget: number; cancelAfter: number | undefined },
  ) {
    this.fn = fn
    this.then = then
    this.inputs = inputs
    this.mode = options.mode
    this.budget = options.budget
    this.cancelAfter = options.cancelAfter ?? 200
  }

  /** The inputs changed: schedule a run according to the mode, the last run's cost and the pointer. */
  request() {
    if (this.mode === 'worker') return this.toWorker()
    if (this.cancel) return // a run is already scheduled; it will read the latest inputs
    const held = isPointerHeld()
    if (held && this.mode === 'release') {
      this.lagging = true
      this.whenReleased()
      return
    }
    if (held && this.slow) {
      // Slow during a drag: after the paint, so the handle moves first; repeated as often as the work allows.
      this.lagging = true
      // As many frames are left free as the last run took, so pointer moves and live patches paint between runs
      // (the work takes at most about half the main thread).
      let frames = Math.min(8, Math.ceil(this.lastMs / 16))
      let next: (() => void) | null = null
      let frame = 0
      const wait = () => {
        if (frames-- > 0) frame = requestAnimationFrame(wait)
        else next = afterPaint(() => this.run(false))
      }
      wait()
      this.cancel = () => {
        cancelAnimationFrame(frame)
        next?.()
      }
      this.whenReleased()
      return
    }
    const frame = requestAnimationFrame(() => this.run(true))
    this.cancel = () => cancelAnimationFrame(frame)
  }

  /** Send the newest task to the worker; its answer, mapped by the `then` of its time, renders when it lands. */
  private toWorker() {
    const inputs = this.inputs
    const then = this.then
    this.lagging = true
    this.worker ??= new ComputeWorker()
    this.worker.cancelAfter = this.cancelAfter
    this.worker.submit(this.fn(), (r) => {
      if (r.ok) {
        this.error = undefined
        this.record(then(r.value), inputs, r.ms)
      } else {
        this.error = r.error
        console.error(`useComputed (worker): ${r.error}`)
      }
      this.lagging = this.worker?.busy ?? false
      this.render(false)
    })
  }

  /** Always run on the final inputs once the pointer is released. */
  private whenReleased() {
    if (this.releaseHook) return
    this.releaseHook = onceReleased(() => {
      this.releaseHook = null
      this.cancel?.()
      this.run(false)
    })
  }

  dispose() {
    this.cancel?.()
    this.cancel = null
    this.releaseHook?.()
    this.releaseHook = null
    this.worker?.dispose()
    this.worker = null
  }
}

export function useComputed<T>(fn: () => T, inputs: readonly unknown[], options?: ComputeOptions): Computed<T>
export function useComputed<R, T = R>(
  task: () => Task<R>,
  inputs: readonly unknown[],
  options: WorkerComputeOptions<R, T>,
): Computed<T>
export function useComputed<T>(
  fn: () => unknown,
  inputs: readonly unknown[],
  options: ComputeOptions | WorkerComputeOptions<unknown, T> = {},
): Computed<T> {
  const mode: ComputeMode = options.mode ?? 'frame'
  const budget = ('budget' in options ? options.budget : undefined) ?? 8
  const inlineUnder = ('inline' in options ? options.inline : undefined) ?? 2
  const [, rerender] = useReducer((n: number) => n + 1, 0)
  const [job] = useState(() => {
    // A scheduled run renders before the paint when it ran in the frame (sync), else when it lands.
    const render = (sync: boolean) => (sync ? flushSync(rerender) : rerender())
    if (options.mode === 'worker') return new Job<T>({ value: options.initial, inputs: NONE, ms: 0 }, budget, render)
    const t0 = now()
    const value = fn() as T
    return new Job<T>({ value, inputs, ms: now() - t0 }, budget, render)
  })

  let changed = !same(job.current.inputs, inputs)
  // Fast enough to compute here and now: no frame's wait and no second render. While a pointer is held, release mode
  // still waits for the release.
  if (changed && mode !== 'worker' && job.lastMs < inlineUnder && !(mode === 'release' && isPointerHeld())) {
    job.inline(fn as () => T, inputs)
    changed = false
  }

  // The newest fn, for whenever the job runs; a worker answer is mapped by the `then` of its request's render.
  const live = useRef(fn)
  const then = (options as Partial<WorkerComputeOptions<unknown, T>>).then
  const cancelAfter = (options as Partial<WorkerComputeOptions<unknown, T>>).cancelAfter
  useLayoutEffect(() => {
    live.current = fn
    job.configure(() => live.current(), then ?? ((r) => r as T), inputs, { mode, budget, cancelAfter })
    if (changed) job.request()
  })
  useLayoutEffect(() => () => job.dispose(), [job])
  return {
    value: job.current.value,
    // Stale only when the answer lags beyond this frame (slow, waiting for release, or any pending worker job): no
    // flicker when fast.
    stale: changed && (job.lagging || mode === 'worker'),
    ms: job.current.ms,
    runs: job.runs,
    slow: job.slow,
    error: job.error,
  }
}
