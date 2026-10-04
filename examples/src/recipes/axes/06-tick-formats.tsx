import { Bars, Figure, Plot, useAxis } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Tick formats',
  question: 'How do I format tick labels, or keep them on whole numbers?',
  explain:
    '`format` labels ticks, tooltips and readouts alike; `integer` keeps ticks on whole numbers for steps and counts.',
}

const steps = Array.from({ length: 8 }, (_, i) => i + 1)

export default function TickFormats() {
  const x = useAxis({ label: 'epoch', integer: true })
  const y = useAxis({ label: 'accuracy', range: [0, 1], format: (v) => `${Math.round(v * 100)}%` })
  return (
    <Figure title="Formatted ticks" purpose="Accuracy as a percentage over whole epochs.">
      <Plot x={x} y={y}>
        <Bars name="accuracy" x={steps} y={steps.map((s) => 1 - 0.6 * Math.exp(-s / 2.5))} />
      </Plot>
    </Figure>
  )
}
