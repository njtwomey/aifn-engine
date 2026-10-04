import { corePrelude } from 'aifn-compute/interpreter'
import { Button, CodeEditor, Curve, Figure, Plot, ProgramStatus, useAxis, useProgram } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Code to plot',
  question: 'How do I let the reader type a program and plot what it returns?',
  explain:
    '`CodeEditor` edits the source (with completion from the prelude); `useProgram` runs the submitted source in the compute worker, so a runaway loop is stopped. The page decides what the value means: here `[x, y]` becomes a `Curve`. `ProgramStatus` shows errors in red with their line.',
}

const START = `// Return [x, y]: math, array, random, stats, linalg and signal are in scope
const x = array.linspace(0, 10, 300)
return [x, x.map((v) => math.sin(2 * v) * math.exp(-v / 4))]
`
const isPair = (v: unknown): v is [number[], number[]] =>
  Array.isArray(v) && v.length === 2 && v.every((a) => Array.isArray(a) && a.every((n) => typeof n === 'number'))

// region (the program itself is in the editor above)
export default function CodeToPlot() {
  const [code, setCode] = useState(START)
  const [submitted, setSubmitted] = useState(START)
  const run = useProgram(submitted)
  const value = run.result?.ok ? run.result.value : null
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return (
    <Figure
      title="A plot from code"
      purpose="Edit the program and press Run (or ⌘↵)."
      defaultSize="L"
      equation={
        <div className="flex flex-col gap-2 text-left">
          <CodeEditor
            value={code}
            onChange={setCode}
            prelude={corePrelude}
            errors={run.errors}
            onRun={() => setSubmitted(code)}
            label="Program"
          />
          <div className="flex items-center gap-3">
            <Button size="sm" variant="outline" onClick={() => setSubmitted(code)}>
              Run (⌘↵)
            </Button>
            <ProgramStatus
              run={run}
              problem={value !== null && !isPair(value) ? 'return [x, y]: two arrays of numbers' : null}
            />
          </div>
        </div>
      }
    >
      <Plot x={x} y={y}>
        {isPair(value) && <Curve name="y" x={value[0]} y={value[1]} />}
      </Plot>
    </Figure>
  )
}
// endregion
