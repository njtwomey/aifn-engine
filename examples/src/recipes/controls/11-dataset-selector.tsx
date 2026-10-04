import { choice, Figure, float, int, Plot, Points, row, useAxis, useFigureState } from 'aifn-render'
import { useMemo } from 'react'
import type { Recipe } from '@examples/recipe'
import { rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Dataset selector',
  question: 'How do I let the reader pick a dataset and its size?',
  explain:
    'One `row` holds the dataset `choice` and its typed size and noise; the data are generated in a `useMemo` keyed on those values, so they change only when the dataset does. Switching datasets keeps the size and noise the reader set.',
}

const makers = {
  circles: (t: number, k: number) => [(1 + k) * Math.cos(t), (1 + k) * Math.sin(t)],
  moons: (t: number, k: number) =>
    k ? [1 - Math.cos(t / 2), 0.5 - Math.sin(t / 2)] : [Math.cos(t / 2), Math.sin(t / 2)],
  lines: (t: number, k: number) => [t / Math.PI - 1, k ? 0.5 : -0.5],
}

export default function DatasetSelector() {
  const s = useFigureState({
    data: row('data', {
      name: choice(['circles', 'moons', 'lines'], 'moons', { label: 'dataset' }),
      n: int(200, { ge: 10, le: 2000, label: 'points' }),
      noise: float(0.1, { ge: 0, le: 1, label: 'noise' }),
    }),
  })
  const { name, n, noise } = s.data
  const pts = useMemo(() => {
    const r = rng(1)
    const k = Array.from({ length: n }, (_, i) => i % 2)
    const xy = k.map((g) => makers[name](2 * Math.PI * r.uniform(), g))
    return { k, x: xy.map((p) => p[0] + noise * r.normal()), y: xy.map((p) => p[1] + noise * r.normal()) }
  }, [name, n, noise])
  const x = useAxis({ label: 'x₁' })
  const y = useAxis({ label: 'x₂', equal: x })
  return (
    <Figure title="Pick a dataset" purpose="Two classes from a chosen generator." state={s}>
      <Plot x={x} y={y}>
        <Points x={pts.x} y={pts.y} group={pts.k} groupNames={['class 0', 'class 1']} />
      </Plot>
    </Figure>
  )
}
