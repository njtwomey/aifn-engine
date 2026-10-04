import { Figure, Plot, Raster, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Heatmap',
  question: 'How do I draw a heatmap of a function on a grid?',
  explain:
    '`Raster` takes cell centres `x` and `y` and a row-major `z` (`z[i][j]` at `x[j]`, `y[i]`); it draws one cached image with a colour bar, and hover reads the cell.',
}

const xs = grid(-3, 3, 80)
const ys = grid(-2, 2, 60)
const z = ys.map((b) =>
  xs.map((a) => Math.exp(-(a * a + b * b) / 2) + 0.5 * Math.exp(-((a - 1.5) ** 2 + (b + 1) ** 2))),
)

export default function Heatmap() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return (
    <Figure title="A heatmap" purpose="Two Gaussian bumps on a grid.">
      <Plot x={x} y={y}>
        <Raster x={xs} y={ys} z={z} valueLabel="f(x, y)" />
      </Plot>
    </Figure>
  )
}
