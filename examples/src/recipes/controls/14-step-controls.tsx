import { Curve, Figure, Plot, Points, Readout, StepControls, useAxis } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Step, run, reset',
  question: 'How do I let the reader run an iteration live, one step at a time?',
  explain:
    '`StepControls` gives Step, Run and Reset buttons for an algorithm held in React state; Step and Run disable once `done`. Use a `Player` instead when the whole run can be computed up front.',
}

const xs = grid(0.5, 3, 200)
const f = (v: number) => v * v - 2

export default function StepControlsRecipe() {
  // Newton's method for √2: x ← x − f(x) / f′(x)
  const [iterates, setIterates] = useState([3])
  const last = iterates[iterates.length - 1]
  const next = (v: number) => v - f(v) / (2 * v)
  const done = Math.abs(f(last)) < 1e-12 || iterates.length > 8
  const x = useAxis({ label: 'x', range: [0.5, 3] })
  const y = useAxis({ label: 'f(x) = x² − 2', range: [-2, 7] })
  return (
    <Figure
      title="Newton's method"
      purpose="Each step follows the tangent to its zero."
      controls={
        <StepControls
          done={done}
          onStep={() => setIterates([...iterates, next(last)])}
          onRun={() => {
            const out = [...iterates]
            while (out.length < 9) out.push(next(out[out.length - 1]))
            setIterates(out)
          }}
          onReset={() => setIterates([3])}
        />
      }
      readouts={<Readout label={`x${iterates.length - 1}`} value={last.toPrecision(12)} />}
    >
      <Plot x={x} y={y}>
        <Curve name="f" x={xs} y={xs.map(f)} />
        <Curve name="tangent" x={[last, next(last)]} y={[f(last), 0]} dashed slot={1} />
        <Points name="iterates" x={iterates} y={iterates.map(f)} emphasis />
      </Plot>
    </Figure>
  )
}
