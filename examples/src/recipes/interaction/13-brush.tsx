import { Figure, Plot, Points, Readout, useAxis, type Range } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'
import { rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Brush to select',
  question: 'How do I select points by dragging a rectangle?',
  explain:
    '`onBrush` reports the dragged rectangle in data coordinates on release, or null on a plain click; filter the data by it and draw the selection on top.',
}

const r = rng(17)
const px = Array.from({ length: 300 }, () => r.normal())
const py = px.map((v) => 0.5 * v + r.normal())

export default function Brush() {
  const [box, setBox] = useState<{ x: Range; y: Range } | null>(null)
  const inside = px.map(
    (v, i) => box !== null && v >= box.x[0] && v <= box.x[1] && py[i] >= box.y[0] && py[i] <= box.y[1],
  )
  const x = useAxis({ label: 'x', hold: 'initial' })
  const y = useAxis({ label: 'y', hold: 'initial' })
  return (
    <Figure
      title="A brushed selection"
      purpose="Drag a rectangle to select points."
      readouts={<Readout label="selected" value={inside.filter(Boolean).length} />}
    >
      <Plot x={x} y={y} onBrush={setBox}>
        <Points name="all" x={px} y={py} muted />
        <Points name="selected" x={px.filter((_, i) => inside[i])} y={py.filter((_, i) => inside[i])} slot={0} />
      </Plot>
    </Figure>
  )
}
