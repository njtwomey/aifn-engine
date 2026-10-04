import { Button, Curve, Figure, Plot, Points, useAxis } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Click to add points',
  question: 'How do I add a point where the reader clicks?',
  explain:
    '`onPlotClick` gives the clicked position in data coordinates. Prefer a `Handle` when a value has a place on the chart; a click suits adding things. Fix the axes so they do not move under the pointer.',
}

export default function ClickToAdd() {
  const [pts, setPts] = useState<[number, number][]>([
    [1, 1],
    [3, 2.5],
  ])
  const sorted = [...pts].sort((a, b) => a[0] - b[0])
  const x = useAxis({ label: 'x', range: [0, 5] })
  const y = useAxis({ label: 'y', range: [0, 4] })
  return (
    <Figure
      title="Click to add"
      purpose="Each click adds a point; the line joins them in x order."
      controls={
        <Button size="sm" variant="outline" onClick={() => setPts([])}>
          Clear
        </Button>
      }
    >
      <Plot x={x} y={y} onPlotClick={(p) => setPts((ps) => [...ps, p])}>
        <Curve name="path" x={sorted.map((p) => p[0])} y={sorted.map((p) => p[1])} muted />
        <Points name="points" x={pts.map((p) => p[0])} y={pts.map((p) => p[1])} />
      </Plot>
    </Figure>
  )
}
