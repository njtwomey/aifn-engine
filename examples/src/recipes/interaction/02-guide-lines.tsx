import { Curve, Figure, formatNumber, Handle, Plot, Readout, useAxis } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Draggable guide lines',
  question: 'How do I drag a threshold or a level?',
  explain:
    '`kind="x"` is a vertical guide that sets an x value (a threshold, a rank); `kind="y"` a horizontal one. With several handles, the nearest to the pointer is grabbed.',
}

const xs = grid(-4, 4, 200)
const cdf = (v: number) => 1 / (1 + Math.exp(-1.7 * v))

export default function GuideLines() {
  const [t, setT] = useState(1)
  const [level, setLevel] = useState(0.25)
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'F(x)', range: [0, 1] })
  return (
    <Figure
      title="A threshold and a level"
      purpose="The x guide reads F at a threshold; the y guide marks a level."
      readouts={
        <>
          <Readout label="F(t)" value={formatNumber(cdf(t))} />
          <Readout label="level" value={formatNumber(level)} />
        </>
      }
      caption="Drag either guide line."
    >
      <Plot x={x} y={y}>
        <Curve name="F" x={xs} y={xs.map(cdf)} />
        <Handle kind="x" at={t} onDrag={setT} label="t" />
        <Handle kind="y" at={level} onDrag={(v) => setLevel(Math.min(1, Math.max(0, v)))} label="level" />
      </Plot>
    </Figure>
  )
}
