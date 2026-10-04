import { Diagram, Figure, link, variable, type DiagramSpec } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Plate notation',
  question: 'How do I draw a graphical model with plates?',
  explain:
    'The `variable` and `link` components build nodes and edges; `filled` marks an observed variable. A group drawn `around` nodes and labelled bottom-right is a plate.',
}

const spec: DiagramSpec = {
  nodes: [
    variable('mu', 0, 0, '$\\mu$'),
    variable('sigma', 0, 2, '$\\sigma$'),
    variable('x', 2.5, 1, '$x_n$', { filled: true }),
  ],
  edges: [link('mu', 'x'), link('sigma', 'x')],
  groups: [{ id: 'N', label: '$N$', tone: 'ink', around: ['x'], pad: 0.4, labelAt: 'bottom-right' }],
}

export default function PlateNotation() {
  return (
    <Figure
      title="A plate model"
      purpose="N observations share a mean and a scale."
      defaultSize="S"
      hoverReadout={false}
    >
      <Diagram spec={spec} ariaLabel="mu and sigma point to x_n inside a plate of N" />
    </Figure>
  )
}
