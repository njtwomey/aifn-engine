import { EquationSteps, Figure } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Step-through derivation',
  question: 'How do I walk through a derivation one line at a time?',
  explain:
    '`EquationSteps` shows the steps up to `step`, the current one at full strength and earlier ones faded, each with a `note`. Given `onStep`, it draws its own `Player` above.',
}

const steps = [
  { tex: String.raw`(x + 1)^2 = (x + 1)(x + 1)`, note: 'Write the square as a product.' },
  { tex: String.raw`= x(x + 1) + 1(x + 1)`, note: 'Distribute the first factor.' },
  { tex: String.raw`= x^2 + x + x + 1`, note: 'Multiply out.' },
  { tex: String.raw`= x^2 + 2x + 1`, note: 'Collect like terms.' },
]

export default function EquationStepsRecipe() {
  const [step, setStep] = useState(0)
  return (
    <Figure title="Expanding a square" purpose="Four steps from (x + 1)² to x² + 2x + 1." hoverReadout={false}>
      <EquationSteps steps={steps} step={step} onStep={setStep} />
    </Figure>
  )
}
