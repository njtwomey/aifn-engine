import { Tex } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'TeX',
  question: 'How do I set maths in a label or a paragraph?',
  explain:
    '`Tex` sets its string with KaTeX, inline or `display`. The shared macros (`\\xvec`, `\\Sigmamat`, `\\expect`, `\\Gauss`, …) are available everywhere; pass `macros` to add your own.',
}

export default function TexRecipe() {
  return (
    <div className="space-y-3 rounded-lg border p-4 font-prose">
      <p>
        A Gaussian vector <Tex>{String.raw`\xvec \sim \Gauss(\muvec, \Sigmamat)`}</Tex> has mean{' '}
        <Tex>{String.raw`\expect[\xvec] = \muvec`}</Tex>.
      </p>
      <Tex
        display
      >{String.raw`p(\xvec) = \frac{1}{\sqrt{(2\pi)^d \abs{\Sigmamat}}} \exp\left(-\tfrac12 (\xvec - \muvec)^\top \Sigmamat^{-1} (\xvec - \muvec)\right)`}</Tex>
      <p>
        Own macros: <Tex macros={{ '\\loss': '\\mathcal{L}' }}>{String.raw`\loss(\thetavec)`}</Tex>.
      </p>
    </div>
  )
}
