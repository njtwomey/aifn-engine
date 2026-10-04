import { Diagram, Figure, type DiagramSpec } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Automatic layout',
  question: 'How do I draw a graph given as data, without placing every node?',
  explain:
    '`layout: "layered"` places nodes in columns by topological depth and reduces crossings, so a graph built from data needs no coordinates.',
}

const steps = ['load', 'clean', 'split', 'train', 'tune', 'test', 'report']
const deps: [string, string][] = [
  ['load', 'clean'],
  ['clean', 'split'],
  ['split', 'train'],
  ['split', 'test'],
  ['train', 'tune'],
  ['tune', 'test'],
  ['test', 'report'],
  ['tune', 'report'],
]

const spec: DiagramSpec = {
  layout: 'layered',
  nodes: steps.map((id, i) => ({ id, label: id, shape: 'box', tone: i % 3 })),
  edges: deps.map(([from, to]) => ({ from, to })),
}

export default function LayeredLayout() {
  return (
    <Figure
      title="A pipeline, laid out automatically"
      purpose="Seven steps placed by their dependencies."
      defaultSize="M"
      hoverReadout={false}
    >
      <Diagram spec={spec} ariaLabel="A machine-learning pipeline from load to report" />
    </Figure>
  )
}
