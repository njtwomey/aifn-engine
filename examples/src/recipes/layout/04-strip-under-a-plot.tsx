import { Bars, Figure, Plot, Plots, Points, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Strip under a plot',
  question: 'How do I put a short strip under a plot, edge to edge?',
  explain:
    '`heights={[3, 1]}` gives the main plot three quarters of the height; `tight` closes the gap between panels that share x, for a strip that belongs to the panel above.',
}

const r = rng(4)
const px = Array.from({ length: 200 }, () => r.normal())
const py = px.map((v) => v + 0.5 * r.normal())
const edges = Array.from({ length: 13 }, (_, i) => -3 + i * 0.5)
const counts = edges.slice(1).map((hi, i) => px.filter((v) => v >= edges[i] && v < hi).length)

export default function StripUnderAPlot() {
  const x = useAxis({ label: 'x', range: [-3, 3] })
  const y = useAxis({ label: 'y' })
  const n = useAxis({ label: 'count', range: [0, undefined] })
  return (
    <Figure title="A scatter with a count strip" purpose="The strip counts points per bin of x.">
      <Plots rows={2} heights={[3, 1]} tight>
        <Plot x={x} y={y}>
          <Points name="data" x={px} y={py} />
        </Plot>
        <Plot x={x} y={n}>
          <Bars name="count" x={edges.slice(0, -1).map((e) => e + 0.25)} y={counts} width={0.45} muted />
        </Plot>
      </Plots>
    </Figure>
  )
}
