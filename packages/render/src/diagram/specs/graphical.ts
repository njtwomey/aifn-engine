/** Graphical-model diagrams shared by the diagram lab and the notes. */
import { link, variable } from '../components'
import type { DiagramSpec } from '../types'

/** Latent Dirichlet allocation in plate notation: documents d hold word positions n; topics k have their own plate. */
export const lda: DiagramSpec = {
  nodes: [
    variable('alpha', 0, 2.4, '$\\alpha$', { w: 0.7, h: 0.7, small: true }),
    variable('theta', 1.7, 2.4, '$\\thetavec_d$'),
    variable('z', 3.5, 2.4, '$z_{dn}$'),
    variable('w', 5.3, 2.4, '$w_{dn}$', { filled: true }),
    variable('phi', 5.3, -0.7, '$\\phivec_k$'),
    variable('beta', 7.3, -0.7, '$\\beta$', { w: 0.7, h: 0.7, small: true }),
  ],
  edges: [link('alpha', 'theta'), link('theta', 'z'), link('z', 'w'), link('phi', 'w'), link('beta', 'phi')],
  groups: [
    { id: 'N', label: '$N_d$', tone: 'ink', around: ['z', 'w'], pad: 0.3, labelAt: 'bottom-right' },
    { id: 'D', label: '$D$', tone: 'ink', around: ['theta', 'z', 'w'], pad: 0.75, labelAt: 'bottom-right' },
    { id: 'K', label: '$K$', tone: 'ink', around: ['phi'], pad: 0.3, labelAt: 'bottom-right' },
  ],
}
