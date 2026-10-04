/**
 * The page side of the compute worker (`compute.worker.ts`): one worker per `useComputed(…, { mode: 'worker' })`,
 * latest wins. One job is in flight at a time; a newer task replaces the one queued behind it (that one is cancelled
 * before it starts). A job in flight finishes, so a drag keeps showing answers that are at most one job old, unless it
 * has been superseded and has run longer than `cancelAfter` ms: then the worker is terminated and a fresh one takes the
 * newest task.
 */
import { fromMessage, toMessage, type WorkerRequest, type WorkerResponse } from './task'

export type WorkerResult = { ok: true; value: unknown; ms: number } | { ok: false; error: string; ms: number }
type Job = { task: unknown; done: (r: WorkerResult) => void; partial?: (value: unknown, ms: number) => void }

let nextId = 1

export class ComputeWorker {
  private worker: Worker | null = null
  private running: (Job & { id: number; started: number }) | null = null
  private queued: Job | null = null
  private timer: ReturnType<typeof setTimeout> | null = null
  /** Workers started (1 + cancellations by termination), for profiling. */
  spawned = 0
  cancelAfter: number

  constructor(cancelAfter = 200) {
    this.cancelAfter = cancelAfter
  }

  /**
   * Run `task`, after or instead of what is pending; `done` gets its answer, and `partial` every value a streaming task
   * (a generator) yields before it.
   */
  submit(task: unknown, done: (r: WorkerResult) => void, partial?: (value: unknown, ms: number) => void) {
    this.queued = { task, done, partial }
    if (!this.running) return this.next()
    const left = this.cancelAfter - (performance.now() - this.running.started)
    if (left <= 0) this.restart()
    else if (!this.timer) this.timer = setTimeout(() => this.restart(), left)
  }

  /** True while a job is in flight or queued. */
  get busy() {
    return this.running !== null || this.queued !== null
  }

  /** Drop the job in flight and any queued one, terminating the worker (a fresh one starts with the next submit). */
  cancel() {
    this.dispose()
  }

  dispose() {
    this.clearTimer()
    this.worker?.terminate()
    this.worker = null
    this.running = this.queued = null
  }

  private clearTimer() {
    if (this.timer) clearTimeout(this.timer)
    this.timer = null
  }

  /** The job in flight is stale and slow: drop it with its worker, and start the newest task on a fresh one. */
  private restart() {
    this.clearTimer()
    if (!this.running || !this.queued) return
    this.worker?.terminate()
    this.worker = null
    this.running = null
    this.next()
  }

  private spawn(): Worker {
    if (this.worker) return this.worker
    const w = new Worker(new URL('./compute.worker.ts', import.meta.url), { type: 'module', name: 'aifn compute' })
    this.spawned++
    w.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const r = e.data
      const job = this.running
      if (!job || job.id !== r.id) return
      if (r.ok && r.partial) return job.partial?.(fromMessage(r.value), r.ms)
      this.running = null
      this.clearTimer()
      job.done(r.ok ? { ok: true, value: fromMessage(r.value), ms: r.ms } : { ok: false, error: r.error, ms: r.ms })
      this.next()
    }
    w.onerror = (e) => {
      const job = this.running
      this.running = null
      this.worker?.terminate()
      this.worker = null
      job?.done({ ok: false, error: e.message || 'worker failed to load', ms: 0 })
      this.next()
    }
    this.worker = w
    return w
  }

  private next() {
    const job = this.queued
    if (!job) return
    this.queued = null
    const id = nextId++
    this.running = { ...job, id, started: performance.now() }
    const request: WorkerRequest = { id, task: toMessage(job.task) }
    this.spawn().postMessage(request)
  }
}
