import { Curve, Figure, MathText, Plot, Readout, Slider, Tex, useAxis, useParam } from 'aifn-render'
import type { Recipe } from '@examples/recipe'
import { grid } from '@examples/data'

export const recipe: Recipe = {
  title: 'Maths in labels',
  question: 'How do I put maths in control labels, readouts and captions?',
  explain:
    'Labels take React nodes: pass a `Tex` element. `MathText` sets a string with inline `$…$` maths, for captions and descriptions. Axis labels are plain strings, so use Unicode there.',
}

const xs = grid(-4, 4, 200)

export default function MathsInLabels() {
  const sigma = useParam(1, { min: 0.3, max: 2 })
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'φ(x)' })
  return (
    <Figure
      title="Maths in labels"
      purpose={<MathText text="The density of $\Gauss(0, \sigma^2)$ for a chosen $\sigma$." />}
      controls={<Slider label={<Tex>{String.raw`\sigma`}</Tex>} param={sigma} />}
      readouts={
        <Readout
          label={<Tex>{String.raw`\varphi(0)`}</Tex>}
          value={(1 / (sigma.value * Math.sqrt(2 * Math.PI))).toFixed(3)}
        />
      }
      caption={<MathText text="Halving $\sigma$ doubles the peak $\varphi(0) = 1/(\sigma\sqrt{2\pi})$." />}
    >
      <Plot x={x} y={y}>
        <Curve
          name="φ"
          x={xs}
          y={xs.map((v) => Math.exp(-0.5 * (v / sigma.value) ** 2) / (sigma.value * Math.sqrt(2 * Math.PI)))}
        />
      </Plot>
    </Figure>
  )
}
