/** Recurrent cells, shared by the diagram lab and the LSTM and GRU notes. */
import { gate, op } from '../components'
import type { DiagramSpec } from '../types'

/** One LSTM step: the cell state runs along the top; the four gates read h(t−1) and x(t) from the bus below. */
export const lstm: DiagramSpec = {
  nodes: [
    { id: 'cprev', x: 0, y: 1, shape: 'text', label: '$\\cvec_{t-1}$' },
    { id: 'cout', x: 12.8, y: 1, shape: 'text', label: '$\\cvec_t$' },
    { id: 'hprev', x: 0, y: 5.4, shape: 'text', label: '$\\hvec_{t-1}$' },
    { id: 'hout', x: 12.8, y: 4, shape: 'text', label: '$\\hvec_t$' },
    { id: 'xt', x: 1.6, y: 6.9, shape: 'text', label: '$\\xvec_t$' },
    { id: 'bus', x: 1.6, y: 5.4, shape: 'dot' },
    op('fmul', 3, 1, '$\\otimes$'),
    op('add', 6.6, 1, '$\\oplus$'),
    gate('f', 3, 3.8, '$\\sigma$', 0),
    gate('i', 4.8, 3.8, '$\\sigma$', 0),
    gate('g', 6.6, 3.8, '$\\tanh$', 1),
    gate('o', 8.6, 3.8, '$\\sigma$', 0),
    { id: 'fl', x: 2.2, y: 4.5, shape: 'text', small: true, label: 'forget', tone: 'neutral' },
    { id: 'il', x: 4.1, y: 4.5, shape: 'text', small: true, label: 'input', tone: 'neutral' },
    { id: 'gl', x: 5.65, y: 4.5, shape: 'text', small: true, label: 'candidate', tone: 'neutral' },
    { id: 'ol', x: 7.85, y: 4.5, shape: 'text', small: true, label: 'output', tone: 'neutral' },
    op('imul', 6.6, 2.4, '$\\otimes$'),
    { id: 'ctanh', x: 10.4, y: 2.4, shape: 'pill', w: 0.9, h: 0.5, label: '$\\tanh$', tone: 1, small: true },
    op('omul', 10.4, 3.8, '$\\otimes$'),
  ],
  edges: [
    { from: 'cprev', to: 'fmul' },
    { from: 'fmul', to: 'add' },
    { from: 'add', to: 'cout' },
    { from: 'f:n', to: 'fmul:s' },
    { from: 'i:n', to: 'imul:w', via: [[4.8, 2.4]] },
    { from: 'g:n', to: 'imul:s' },
    { from: 'imul:n', to: 'add:s' },
    { from: 'add:e', to: 'ctanh:n', via: [[10.4, 1]] },
    { from: 'ctanh:s', to: 'omul:n' },
    { from: 'o:e', to: 'omul:w' },
    { from: 'omul:e', to: 'hout:w' },
    { from: 'hprev', to: 'bus', arrow: 'none' },
    { from: 'xt', to: 'bus', arrow: 'none' },
    { from: 'bus', to: 'f:s', via: [[3, 5.4]] },
    { from: 'bus', to: 'i:s', via: [[4.8, 5.4]] },
    { from: 'bus', to: 'g:s', via: [[6.6, 5.4]] },
    { from: 'bus', to: 'o:s', via: [[8.6, 5.4]] },
  ],
  groups: [
    {
      id: 'cell',
      label: 'LSTM cell',
      around: ['fmul', 'add', 'f', 'o', 'imul', 'ctanh', 'omul', 'bus', 'fl', 'ol'],
      pad: 0.45,
    },
  ],
}

/**
 * One GRU step in Cho et al.'s convention, h(t) = z ⊙ h(t−1) + (1 − z) ⊙ h̃(t): the update gate multiplies the old state
 * directly and its complement scales the candidate.
 */
export const gru: DiagramSpec = {
  nodes: [
    { id: 'hprev', x: 0, y: 1, shape: 'text', label: '$\\hvec_{t-1}$' },
    { id: 'hout', x: 12.6, y: 1, shape: 'text', label: '$\\hvec_t$' },
    { id: 'xt', x: 1.6, y: 6.9, shape: 'text', label: '$\\xvec_t$' },
    { id: 'top', x: 1.6, y: 1, shape: 'dot' },
    { id: 'bus', x: 1.6, y: 5.4, shape: 'dot' },
    gate('r', 3.2, 3.8, '$\\sigma$', 0),
    gate('z', 5.2, 3.8, '$\\sigma$', 0),
    gate('c', 8, 3.8, '$\\tanh$', 1),
    { id: 'rl', x: 2.5, y: 4.5, shape: 'text', small: true, label: 'reset', tone: 'neutral' },
    { id: 'zl', x: 4.4, y: 4.5, shape: 'text', small: true, label: 'update', tone: 'neutral' },
    { id: 'cl', x: 6.8, y: 4.5, shape: 'text', small: true, label: 'candidate $\\tilde\\hvec_t$', tone: 'neutral' },
    op('rmul', 3.2, 2.4, '$\\otimes$'),
    { id: 'jz', x: 5.2, y: 2, shape: 'dot' },
    { id: 'one', x: 6.6, y: 2, shape: 'pill', w: 0.8, h: 0.5, label: '$1-$', small: true },
    op('amul', 5.2, 1, '$\\otimes$'),
    op('bmul', 8, 2, '$\\otimes$'),
    op('add', 10.2, 1, '$\\oplus$'),
  ],
  edges: [
    { from: 'hprev', to: 'top', arrow: 'none' },
    { from: 'top', to: 'amul' },
    { from: 'amul', to: 'add' },
    { from: 'add', to: 'hout' },
    { from: 'top', to: 'bus', arrow: 'none' },
    { from: 'xt', to: 'bus', arrow: 'none' },
    { from: 'top:e', to: 'rmul:n', via: [[3.2, 1]] },
    { from: 'r:n', to: 'rmul:s' },
    {
      from: 'rmul:e',
      to: 'c:w',
      via: [
        [4.2, 2.4],
        [4.2, 3.1],
        [7.1, 3.1],
        [7.1, 3.8],
      ],
    },
    { from: 'z:n', to: 'jz', arrow: 'none' },
    { from: 'jz', to: 'amul:s' },
    { from: 'jz', to: 'one:w' },
    { from: 'one:e', to: 'bmul:w' },
    { from: 'c:n', to: 'bmul:s' },
    { from: 'bmul:e', to: 'add:s', via: [[10.2, 2]] },
    { from: 'bus', to: 'r:s', via: [[3.2, 5.4]] },
    { from: 'bus', to: 'z:s', via: [[5.2, 5.4]] },
    { from: 'bus', to: 'c:s', via: [[8, 5.4]] },
  ],
  groups: [
    { id: 'cell', label: 'GRU cell', around: ['top', 'bus', 'r', 'rmul', 'amul', 'add', 'bmul', 'c', 'rl'], pad: 0.45 },
  ],
}
