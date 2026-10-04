import { Curve, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid, rng } from '@examples/data'

export const recipe: Recipe = {
  title: 'Line styles',
  question: 'How do I draw dashed, thin and marked lines?',
  explain:
    '`dashed` for a reference, `thin` for one of many random draws (with `muted` they sit behind), `showPoints` to mark every vertex, and `emphasis` for the line that matters, drawn in ink.',
}

const xs = grid(0, 10, 41)
const r = rng(3)
const walks = Array.from({ length: 12 }, () => {
  let v = 0
  return xs.map(() => (v += 0.3 * r.normal()))
})

export default function LineStyles() {
  const x = useAxis({ label: 't' })
  const y = useAxis({ label: 'value' })
  return (
    <Figure title="Line styles" purpose="Draws as thin muted lines, their mean in ink, a dashed zero line.">
      <Plot x={x} y={y}>
        {walks.map((w, i) => (
          <Curve key={i} name="draws" x={xs} y={w} thin muted />
        ))}
        <Curve
          name="mean"
          x={xs}
          y={xs.map((_, j) => walks.reduce((s, w) => s + w[j], 0) / walks.length)}
          emphasis
          showPoints
        />
        <Curve name="zero" x={[0, 10]} y={[0, 0]} dashed slot={1} />
      </Plot>
    </Figure>
  )
}
