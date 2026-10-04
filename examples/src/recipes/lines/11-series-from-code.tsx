import { Figure, Plot, seriesLayers, useAxis, type SeriesSpec } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Series built by code',
  question: 'How do I draw a list of series built in a loop?',
  explain:
    'When the series come from data (one per k, optional extras), build `SeriesSpec` objects and let `seriesLayers` turn them into layers. A fixed set reads better written out as layer elements.',
}

const xs = grid(-1, 1, 101)
// Chebyshev polynomials T_k(x) = cos(k arccos x)
const series: SeriesSpec[] = [1, 2, 3, 4].map((k) => ({
  name: `T${k}`,
  type: 'line',
  x: xs,
  y: xs.map((v) => Math.cos(k * Math.acos(v))),
  slot: k - 1,
}))

export default function SeriesFromCode() {
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'Tₖ(x)' })
  return (
    <Figure title="Series from a list" purpose="The first four Chebyshev polynomials, one spec each.">
      <Plot x={x} y={y}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
