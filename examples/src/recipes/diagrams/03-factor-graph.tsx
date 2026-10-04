import { Diagram, factor, Figure, link, variable, type DiagramSpec } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Factor graph',
  question: 'How do I draw a factor graph and react to clicks on its nodes?',
  explain:
    '`factor` nodes are small squares; `link(a, b, false)` is undirected. `onNodeClick` makes nodes clickable (and focusable); write the choice back into the spec, e.g. as `highlight`.',
}

export default function FactorGraph() {
  const [chosen, setChosen] = useState('x2')
  const spec: DiagramSpec = {
    nodes: [
      variable('x1', 0, 1.5, '$x_1$'),
      factor('f12', 2, 1.5, '$\\psi_{12}$', 's'),
      variable('x2', 4, 1.5, '$x_2$'),
      factor('f23', 6, 1.5, '$\\psi_{23}$', 's'),
      variable('x3', 8, 1.5, '$x_3$'),
      factor('f2', 4, 0, '$\\phi_2$', 'n'),
    ].map((n) => ({ ...n, highlight: n.id === chosen })),
    edges: [
      link('x1', 'f12', false),
      link('f12', 'x2', false),
      link('x2', 'f23', false),
      link('f23', 'x3', false),
      link('f2', 'x2', false),
    ],
  }
  return (
    <Figure
      title="A chain factor graph"
      purpose="Click a node to highlight it."
      defaultSize="S"
      hoverReadout={false}
      caption={`Chosen: ${chosen}`}
    >
      <Diagram spec={spec} ariaLabel="A chain of three variables with pair factors" onNodeClick={setChosen} />
    </Figure>
  )
}
