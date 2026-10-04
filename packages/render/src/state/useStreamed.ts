/**
 * `useStreamed(task)`: run a streaming worker task (a generator, such as `aifn-methods/gym` `training`) and follow its
 * partial answers. Each new task (by identity) cancels the one in flight; `null` runs nothing. The value is the latest
 * partial or final answer, so a figure fills in while the worker computes. `stop()` cancels the task in flight and keeps
 * the last partial answer as the value, with `stopped` set.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { Task } from './task'
import { ComputeWorker } from './worker'

export type Streamed<T> = {
  /** The latest answer (partial while `running`), or null before the first. */
  value: T | null
  /** True while the task is still yielding. */
  running: boolean
  /** Milliseconds in the worker so far. */
  ms: number
  error?: string
  /** True when `stop` ended the task: `value` is the last partial answer. */
  stopped?: boolean
}

export function useStreamed<T>(task: Task<T> | null): Streamed<T> & { stop: () => void } {
  const worker = useRef<ComputeWorker | null>(null)
  // The answers so far, tagged with the task they belong to: a newer task is running until its first answer.
  // The partial answers of a stopped task are ignored from then on.
  const live = useRef<{ live: boolean } | null>(null)
  const [state, setState] = useState<Streamed<T> & { task: Task<T> | null }>({
    task: null,
    value: null,
    running: false,
    ms: 0,
  })
  useEffect(() => {
    // A superseded job is cancelled at once: training runs are long and only the newest matters.
    worker.current ??= new ComputeWorker(0)
    return () => {
      worker.current?.dispose()
      worker.current = null
    }
  }, [])
  useEffect(() => {
    if (!task) return
    worker.current ??= new ComputeWorker(0)
    const flag = { live: true }
    live.current = flag
    worker.current.submit(
      task,
      (r) => {
        if (!flag.live) return
        setState((s) =>
          r.ok
            ? { task, value: r.value as T, running: false, ms: r.ms }
            : { ...s, task, running: false, ms: r.ms, error: r.error },
        )
      },
      (value, ms) => flag.live && setState({ task, value: value as T, running: true, ms }),
    )
    return () => {
      flag.live = false
    }
  }, [task])
  const stop = useCallback(() => {
    if (live.current) live.current.live = false
    worker.current?.cancel()
    setState((s) =>
      s.task !== task
        ? { task, value: null, running: false, ms: 0, stopped: true }
        : s.running
          ? { ...s, running: false, stopped: true }
          : s,
    )
  }, [task])
  const { task: of, ...rest } = state
  return of === task ? { ...rest, stop } : { ...rest, running: task !== null, error: undefined, stopped: false, stop }
}
