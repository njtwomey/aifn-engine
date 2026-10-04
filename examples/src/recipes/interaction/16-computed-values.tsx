import { Curve, Figure, Handle, Plot, Readout, useAxis, useComputed, type Vec2 } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Slow derived values',
  question: 'How do I recompute something slow while the reader drags?',
  explain:
    '`useComputed(fn, inputs)` runs at most once a frame on the latest inputs; when a run is over budget it returns the last value with `stale: true` (pass `stale` to the layer to fade it) and always finishes on release.',
}

// A deliberately slow orbit: many small steps of a pendulum from a start point
function orbit([q, p]: Vec2) {
  const xs = [q],
    ys = [p]
  for (let i = 0; i < 200_000; i++) {
    p -= 0.0005 * Math.sin(q)
    q += 0.0005 * p
    if (i % 200 === 0) {
      xs.push(q)
      ys.push(p)
    }
  }
  return { xs, ys }
}

export default function ComputedValues() {
  const [start, setStart] = useState<Vec2>([1, 0.5])
  const path = useComputed(() => orbit(start), [start[0], start[1]])
  const x = useAxis({ label: 'angle q', range: [-4, 4] })
  const y = useAxis({ label: 'momentum p', range: [-3, 3] })
  return (
    <Figure
      title="A computed orbit"
      purpose="The orbit is recomputed as the start moves."
      readouts={<Readout label="last run" value={`${path.ms.toFixed(1)} ms`} />}
    >
      <Plot x={x} y={y}>
        <Curve name="orbit" x={path.value.xs} y={path.value.ys} stale={path.stale} />
        <Handle kind="point" at={start} onDrag={setStart} />
      </Plot>
    </Figure>
  )
}
