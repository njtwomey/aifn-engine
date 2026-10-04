import { Curve, Figure, Plot, Plots, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Grid of plots',
  question: 'How do I lay out a grid of subplots?',
  explain:
    '`Plots` arranges one `Plot` per cell in row-major order, with aligned plot areas; `heights` and `widths` set relative sizes. Each `Plot` can carry a small `title`.',
}

const xs = grid(-2, 2, 200)
const fns: [string, (v: number) => number][] = [
  ['x', (v) => v],
  ['x²', (v) => v * v],
  ['x³', (v) => v ** 3],
  ['sin 3x', (v) => Math.sin(3 * v)],
]

export default function SubplotsGrid() {
  const x = useAxis({ label: 'x' })
  const ys = [useAxis({}), useAxis({}), useAxis({}), useAxis({})]
  return (
    <Figure title="A 2 × 2 grid" purpose="Four functions, one per panel, sharing x." defaultSize="L">
      <Plots rows={2} cols={2}>
        {fns.map(([name, f], i) => (
          <Plot key={name} title={name} x={x} y={ys[i]}>
            <Curve name={name} x={xs} y={xs.map(f)} slot={i} />
          </Plot>
        ))}
      </Plots>
    </Figure>
  )
}
