import { treeFromNested } from 'aifn-compute/graph'
import { Figure, TreeView } from 'aifn-render'
import { useState } from 'react'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Tree',
  question: 'How do I draw a tree and highlight a path on hover?',
  explain:
    '`TreeView` lays out an `aifn-compute/graph` tree (here from nested objects) with parents centred over children. `highlight` lights the given nodes and the edges between them; `onNodeHover` reports the node under the pointer.',
}

const tree = treeFromNested({
  label: '$+$',
  children: [
    { label: '$\\times$', children: [{ label: '$2$' }, { label: '$x$' }] },
    { label: '$\\sin$', children: [{ label: '$\\times$', children: [{ label: '$x$' }, { label: '$y$' }] }] },
  ],
})

const pathTo = (id: number) => {
  const out: number[] = []
  for (let v: number | null = id; v !== null; v = tree.nodes[v].parent) out.push(v)
  return out
}

export default function TreeViewRecipe() {
  const [hovered, setHovered] = useState<number | null>(null)
  return (
    <Figure
      title="An expression tree"
      purpose="2x + sin(xy); hover a node to light its path from the root."
      defaultSize="S"
      hoverReadout={false}
    >
      <TreeView
        tree={tree}
        highlight={hovered === null ? [] : pathTo(hovered)}
        onNodeHover={setHovered}
        ariaLabel="Expression tree of 2x + sin(xy)"
      />
    </Figure>
  )
}
