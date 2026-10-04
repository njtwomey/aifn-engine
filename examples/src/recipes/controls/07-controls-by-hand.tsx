import { ControlRow, Curve, Figure, NumberField, Plot, Select, Slider, Switch, useAxis } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Controls by hand',
  question: 'How do I place controls myself, without a state schema?',
  explain:
    "`Slider`, `NumberField`, `Select` and `Switch` take a `value` and an `onChange` (or a `param`); pass them as the figure's `controls`. `ControlRow` groups several under one label.",
}

const xs = grid(0, 1, 200)

export default function HandPlacedControls() {
  const [p, setP] = useState(2)
  const [n, setN] = useState(3)
  const [kind, setKind] = useState<'power' | 'root'>('power')
  const [flip, setFlip] = useState(false)
  const f = (v: number) => (flip ? 1 - v : v) ** (kind === 'power' ? p : 1 / p)
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'f(x)', range: [0, 1] })
  return (
    <Figure
      title="Controls placed by hand"
      purpose="Four kinds of control, each with its own state."
      controls={
        <>
          <ControlRow label="curve">
            <Select label="kind" value={kind} onChange={setKind} options={['power', 'root']} />
            <Slider label="p" value={p} min={1} max={6} step={0.5} onChange={setP} />
          </ControlRow>
          <NumberField label="copies" value={n} onChange={setN} type="int" ge={1} le={8} />
          <Switch label="mirror" checked={flip} onChange={setFlip} />
        </>
      }
    >
      <Plot x={x} y={y}>
        {Array.from({ length: n }, (_, i) => (
          <Curve key={i} name={`copy ${i + 1}`} x={xs} y={xs.map((v) => f(v) * (1 - i / (n + 1)))} slot={i % 4} />
        ))}
      </Plot>
    </Figure>
  )
}
