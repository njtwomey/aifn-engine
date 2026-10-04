import { treeFromNested } from 'aifn-compute/graph'
import { Figure, Player, TreeView, usePlayhead } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Tree walk',
  question: 'How do I show the progress of a walk over a tree?',
  explain:
    '`nodeState` marks each node `idle` (dimmed), `active` (emphasised) or `done`; `nodeTone` colours by slot and `nodeNotes` adds a note beside a node. A `Player` walks the order.',
}

const tree = treeFromNested({
  label: 'A',
  children: [
    { label: 'B', children: [{ label: 'D' }, { label: 'E' }] },
    { label: 'C', children: [{ label: 'F' }] },
  ],
})
// Breadth-first order of node ids (ids are in pre-order: A0 B1 D2 E3 C4 F5)
const order = [0, 1, 4, 2, 3, 5]

export default function TreeStates() {
  const [step, setStep] = usePlayhead(order.length)
  const rank = (id: number) => order.indexOf(id)
  return (
    <Figure
      title="Breadth-first order"
      purpose="Nodes visited level by level."
      defaultSize="S"
      hoverReadout={false}
      controls={<Player value={step} onChange={setStep} count={order.length} label="visit" />}
    >
      <TreeView
        tree={tree}
        nodeState={(id) => (rank(id) < step ? 'done' : rank(id) === step ? 'active' : 'idle')}
        nodeTone={(id) => (rank(id) <= step ? 0 : 'neutral')}
        nodeNotes={(id) => (rank(id) <= step ? `#${rank(id) + 1}` : '')}
        ariaLabel="A tree visited breadth first"
      />
    </Figure>
  )
}
