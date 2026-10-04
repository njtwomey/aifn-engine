import { Bars, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Outcome colours',
  question: 'How do I colour a success or a failure?',
  explain:
    '`tone="success"` and `tone="destructive"` take the theme\'s green and red in place of a palette slot, for outcomes rather than categories.',
}

const runs = [1, 2, 3, 4, 5, 6, 7, 8]
const ms = [120, 95, 300, 88, 410, 101, 99, 97]
const ok = ms.map((t) => t < 200)

export default function OutcomeTones() {
  const x = useAxis({ label: 'run', integer: true })
  const y = useAxis({ label: 'ms', range: [0, undefined] })
  return (
    <Figure title="Passed and failed runs" purpose="Runs over the 200 ms budget fail.">
      <Plot x={x} y={y}>
        <Bars name="passed" x={runs.filter((_, i) => ok[i])} y={ms.filter((_, i) => ok[i])} tone="success" />
        <Bars name="failed" x={runs.filter((_, i) => !ok[i])} y={ms.filter((_, i) => !ok[i])} tone="destructive" />
      </Plot>
    </Figure>
  )
}
