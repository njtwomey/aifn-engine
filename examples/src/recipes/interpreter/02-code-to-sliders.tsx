import { corePrelude } from 'aifn-compute/interpreter'
import {
  CodeEditor,
  Curve,
  EntryControls,
  Figure,
  Plot,
  ProgramStatus,
  useAxis,
  useEntryArgs,
  useProgram,
} from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Code to sliders',
  question: "How do I turn a function's parameters into sliders?",
  explain:
    "`useEntryArgs` reads the entry function's signature (`make` by default): literal defaults give the types, and a JSDoc `@param {real} name [lo, hi]` line gives the range. `EntryControls` draws the controls; their values are the run's `args`, so the plot follows every move.",
}

const START = `/**
 * @param {real} freq [0.5, 5] frequency
 * @param {real} damping [0, 1] decay rate
 * @param {int} n [10, 500] number of points
 */
function make(freq = 2, damping = 0.25, n = 300) {
  const x = array.linspace(0, 10, n)
  return [x, x.map((v) => math.sin(freq * v) * math.exp(-damping * v))]
}
`

// region (the program itself is in the editor above)
export default function CodeToSliders() {
  const [code, setCode] = useState(START)
  const entry = useEntryArgs(code)
  const run = useProgram(code, { args: entry.args })
  const value = run.result?.ok ? (run.result.value as [number[], number[]]) : null
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'make(…)[1]', hold: 'union' })
  return (
    <Figure
      title="Sliders from a signature"
      purpose="The controls come from make()'s parameters."
      defaultSize="L"
      equation={
        <div className="flex flex-col gap-2 text-left">
          <CodeEditor value={code} onChange={setCode} prelude={corePrelude} errors={run.errors} label="Program" />
          <div className="grid grid-cols-[repeat(auto-fill,minmax(14rem,1fr))] gap-x-6 gap-y-3">
            <EntryControls key={entry.controlsKey} {...entry.controls} />
          </div>
          <ProgramStatus run={run} />
        </div>
      }
    >
      <Plot x={x} y={y}>
        {value && <Curve name="make" x={value[0]} y={value[1]} />}
      </Plot>
    </Figure>
  )
}
// endregion
