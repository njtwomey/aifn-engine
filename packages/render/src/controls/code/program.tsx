/**
 * A program from `aifn-compute/interpreter` on a page: its entry function's parameters as controls (`useEntryArgs`,
 * `EntryControls`), its run in the compute worker (`useProgram`), and a status line (`ProgramStatus`). Shared by the
 * lab's interpreter page, the examples and notes; what a program's value means (a curve, a table, a dataset) stays
 * with the page.
 *
 *   const entry = useEntryArgs(code)
 *   const run = useProgram(code, { args: entry.args, runKey })
 *   <CodeEditor value={code} onChange={setCode} errors={run.errors} onRun={…} />
 *   <EntryControls key={entry.controlsKey} {...entry.controls} />
 *   <ProgramStatus run={run} />
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import type { Space } from 'aifn-compute/foundation/space'
import { entrySignature, type EntrySignature, type RunResult } from 'aifn-compute/interpreter'
import { fromSpace } from '../../state/schema'
import { call, type Task } from '../../state/task'
import { useComputed } from '../../state/useComputed'
import { StatusText } from '../base/StatusText'
import { ParamControls } from '../schema/ParamControls'
import { useParams } from '../schema/variants'
import type { CodeError } from './CodeEditor'

export type ArgValues = Readonly<Record<string, unknown>>

/** Controls for an entry function's parameters, from its signature's `Space`; reports their values. */
export function EntryControls({
  space,
  initial,
  onChange,
}: {
  space: Space
  initial: ArgValues
  onChange: (values: ArgValues) => void
}) {
  const defs = useMemo(() => fromSpace(space), [space])
  const p = useParams(defs, initial)
  useEffect(() => onChange(p.values as ArgValues), [p.values, onChange])
  return <ParamControls {...p} />
}

/**
 * The entry function's signature and the arguments its controls give. The controls rebuild (a new `key`) when the
 * signature changes (names, types, ranges, defaults), keeping the values of parameters whose dimension is unchanged;
 * editing the body keeps every value. A parameter without a control is passed as undefined, so its default applies.
 */
export function useEntryArgs(source: string): {
  signature: EntrySignature
  args: unknown[]
  /** Props for `EntryControls`; empty `space.dims` when there are no parameters. */
  controls: { space: Space; initial: ArgValues; onChange: (values: ArgValues) => void }
  /** The `key` for `EntryControls`: it changes with the signature, so the controls rebuild. */
  controlsKey: string
} {
  const signature = useMemo(() => entrySignature(source), [source])
  const key = JSON.stringify(signature.space)
  const [shape, setShape] = useState({ key, space: signature.space, initial: {} as ArgValues })
  const [values, setValues] = useState<ArgValues>({})
  if (shape.key !== key) {
    const same = (k: string) => JSON.stringify(shape.space.dims[k]) === JSON.stringify(signature.space.dims[k])
    const kept = Object.fromEntries(Object.entries(values).filter(([k]) => k in signature.space.dims && same(k)))
    setShape({ key, space: signature.space, initial: kept })
  }
  const onChange = useCallback((v: ArgValues) => setValues(v), [])
  const args = signature.params.map((p) => (p.name in shape.space.dims ? values[p.name] : undefined))
  return { signature, args, controls: { space: shape.space, initial: shape.initial, onChange }, controlsKey: shape.key }
}

export type ProgramRun = {
  /** The last run's result (null before the first answer). */
  result: RunResult | null
  /** A newer run is on its way. */
  running: boolean
  /** The run's error as editor marks (with line and column when known). */
  errors: CodeError[]
  /** A worker failure (not the program's own error), e.g. a runaway run cancelled. */
  failure?: string
}

/**
 * Runs `source` with `runProgram` in the compute worker whenever it, `seed`, `args` or `runKey` changes; latest wins,
 * and a superseded run longer than `cancelAfter` ms (a runaway loop) is stopped. `prelude` is a task giving a larger
 * prelude than compute's (e.g. `call('applied/interpreter/prelude')`).
 */
export function useProgram(
  source: string,
  options: { seed?: number; args?: readonly unknown[]; prelude?: Task; runKey?: unknown; cancelAfter?: number } = {},
): ProgramRun {
  const { seed = 0, args = [], prelude, runKey, cancelAfter = 400 } = options
  const run = useComputed(
    () => call<RunResult>('interpreter/runProgram', source, { seed, args, ...(prelude && { prelude }) }),
    [source, seed, JSON.stringify(args), runKey],
    { mode: 'worker', initial: null as RunResult | null, cancelAfter },
  )
  const result = run.value
  const error = result && !result.ok ? result.error : null
  const errors = useMemo<CodeError[]>(() => (error ? [error] : []), [error])
  return { result, running: run.stale, errors, failure: run.error }
}

/**
 * One status line for a run: its error in the destructive tone with the line and column, else its time; `problem`
 * reports the page's own complaint about the value (a wrong shape). Printed output follows.
 */
export function ProgramStatus({ run, problem }: { run: ProgramRun; problem?: string | null }) {
  const { result } = run
  const error = result && !result.ok ? result.error : null
  const where = error?.line !== undefined ? ` (line ${error.line}, column ${error.column})` : ''
  const message = run.failure ?? (error ? `${error.name}: ${error.message}${where}` : problem)
  return (
    <div className="flex min-h-6 flex-col gap-1 font-mono text-xs" aria-live="polite">
      {message ? (
        <StatusText tone="error">{message}</StatusText>
      ) : (
        <StatusText>
          {result?.ok ? `ran in ${result.ms.toFixed(1)} ms` : 'running…'}
          {run.running && result ? ' · running…' : ''}
        </StatusText>
      )}
      {result && result.output.length > 0 && (
        <pre className="max-h-24 overflow-auto whitespace-pre-wrap text-muted-foreground">
          {result.output.join('\n')}
        </pre>
      )}
    </div>
  )
}
