import { Contours, Figure, Plot, Raster, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Contour lines',
  question: 'How do I draw contour lines over a heatmap?',
  explain:
    '`Contours` traces level sets of the same grid by marching squares, at the `levels` you give. A `fillOpacity` below one mutes the raster so the lines stand out.',
}

const xs = grid(-2, 2, 90)
const ys = grid(-1, 3, 90)
// Rosenbrock's banana, on a log scale
const z = ys.map((b) => xs.map((a) => Math.log10(1 + (1 - a) ** 2 + 100 * (b - a * a) ** 2)))

export default function ContoursRecipe() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  return (
    <Figure title="Contours" purpose="Level sets of log₁₀ of Rosenbrock's function.">
      <Plot x={x} y={y}>
        <Raster x={xs} y={ys} z={z} fillOpacity={0.6} valueLabel="log₁₀ f" />
        <Contours x={xs} y={ys} z={z} levels={[0.5, 1, 1.5, 2, 2.5, 3]} />
      </Plot>
    </Figure>
  )
}
