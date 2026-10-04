import { Diagram, Figure, type DiagramSpec } from 'aifn-render'
import type { Recipe } from '@examples/recipe'

export const recipe: Recipe = {
  title: 'Flow diagram',
  question: 'How do I draw a block diagram with arrows?',
  explain:
    'A `Diagram` is hand-placed: nodes have centres in grid units, edges join ports (`id:n|s|e|w`) at right angles, through `via` waypoints if needed. Labels are KaTeX in `$…$`; `tone` picks a palette slot.',
}

const spec: DiagramSpec = {
  nodes: [
    { id: 'x', x: 0, y: 1, shape: 'pill', label: 'input $\\xvec$' },
    { id: 'enc', x: 2.6, y: 1, shape: 'encoder', dir: 'right', label: 'encode', tone: 0 },
    { id: 'z', x: 5, y: 1, shape: 'circle', label: '$\\zvec$' },
    { id: 'dec', x: 7.4, y: 1, shape: 'decoder', dir: 'right', label: 'decode', tone: 1 },
    { id: 'y', x: 10, y: 1, shape: 'pill', label: '$\\hat{\\xvec}$' },
    { id: 'loss', x: 5, y: 3, shape: 'box', label: 'loss $\\norm{\\xvec - \\hat{\\xvec}}^2$', tone: 'neutral' },
  ],
  edges: [
    { from: 'x:e', to: 'enc:w' },
    { from: 'enc:e', to: 'z:w' },
    { from: 'z:e', to: 'dec:w' },
    { from: 'dec:e', to: 'y:w' },
    { from: 'y:s', to: 'loss:e', via: [[10, 3]], dashed: true },
    { from: 'x:s', to: 'loss:w', via: [[0, 3]], dashed: true },
  ],
}

export default function FlowDiagram() {
  return (
    <Figure title="An autoencoder" purpose="Encode, decode and compare." defaultSize="S" hoverReadout={false}>
      <Diagram spec={spec} ariaLabel="An autoencoder: input, encoder, code, decoder, reconstruction and loss" />
    </Figure>
  )
}
