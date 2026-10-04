/** Autoencoder, VAE and latent-diffusion diagrams, shared by the diagram lab and the generative-model notes. */
import { merge, projector, reparam } from '../components'
import type { DiagramSpec } from '../types'

export const autoencoder: DiagramSpec = {
  nodes: [
    { id: 'x', x: 0, y: 2, w: 0.9, h: 0.9, label: '$\\xvec$' },
    projector('encoder', 'enc', 2.2, 2, 'encoder\n$f_{\\phivec}$'),
    { id: 'z', x: 4.4, y: 2, shape: 'latent', label: '$\\zvec$', tone: 2 },
    projector('decoder', 'dec', 6.6, 2, 'decoder\n$g_{\\thetavec}$', { tone: 1 }),
    { id: 'xh', x: 8.8, y: 2, w: 0.9, h: 0.9, label: '$\\hat\\xvec$' },
  ],
  edges: [
    { from: 'x', to: 'enc' },
    { from: 'enc', to: 'z' },
    { from: 'z', to: 'dec' },
    { from: 'dec', to: 'xh' },
    {
      from: 'xh:s',
      to: 'x:s',
      via: [
        [8.8, 3.9],
        [0, 3.9],
      ],
      dashed: true,
      arrow: 'none',
      label: 'loss $\\norm{\\xvec - \\hat\\xvec}^2$',
    },
  ],
  groups: [{ id: 'bneck', label: 'bottleneck', tone: 2, dashed: true, around: ['z'], pad: 0.3 }],
}

export const vae: DiagramSpec = merge(
  {
    nodes: [
      { id: 'x', x: 0, y: 3, w: 0.9, h: 0.9, label: '$\\xvec$' },
      projector('encoder', 'enc', 2.2, 3, 'encoder\n$q_{\\phivec}(\\zvec \\mid \\xvec)$', { w: 2.2, h: 3 }),
      { id: 'z', x: 9.2, y: 3, shape: 'latent', label: '$\\zvec$', tone: 2 },
      projector('decoder', 'dec', 11.5, 3, 'decoder\n$p_{\\thetavec}(\\xvec \\mid \\zvec)$', { w: 2.2, h: 3, tone: 1 }),
      { id: 'xh', x: 13.8, y: 3, w: 0.9, h: 0.9, label: '$\\hat\\xvec$' },
      { id: 'kl', x: 4.8, y: 0.5, shape: 'text', small: true, label: 'KL term $\\KL(q_{\\phivec} \\,\\|\\, p)$' },
    ],
    edges: [
      { from: 'x', to: 'enc' },
      { from: 'rpadd', to: 'z' },
      { from: 'z', to: 'dec' },
      { from: 'dec', to: 'xh' },
      { from: 'rpmu:n', to: 'kl:s', dashed: true, arrow: 'none' },
    ],
  },
  reparam('rp', 4.8, 3, 'enc'),
)

export const latentDiffusion: DiagramSpec = {
  unit: 38,
  nodes: [
    { id: 'x', x: 0, y: 1.5, w: 0.9, h: 0.9, label: '$\\xvec$' },
    projector('encoder', 'E', 2.1, 1.5, '$\\Ecal$', { w: 1.4, h: 1.9 }),
    { id: 'z0', x: 4.3, y: 1.5, shape: 'latent', label: '$\\zvec_0$', tone: 0 },
    { id: 'dots', x: 6.6, y: 1.5, shape: 'text', label: '$\\cdots$' },
    { id: 'zt', x: 8.9, y: 1.5, shape: 'latent', label: '$\\zvec_t$', tone: 0 },
    { id: 'zT', x: 11.2, y: 1.5, shape: 'noise', label: '$\\zvec_T$', tone: 0 },
    {
      id: 'unet',
      x: 7.8,
      y: 4.1,
      w: 4,
      h: 0.9,
      label: 'denoiser $\\epsilonvec_{\\thetavec}(\\zvec_t, t, \\cvec)$',
      tone: 1,
    },
    { id: 'z0h', x: 4.3, y: 4.1, shape: 'latent', label: '$\\hat\\zvec_0$', tone: 0 },
    projector('decoder', 'D', 2.1, 4.1, '$\\Dcal$', { w: 1.4, h: 1.9, dir: 'left', tone: 1 }),
    { id: 'xh', x: 0, y: 4.1, w: 0.9, h: 0.9, label: '$\\hat\\xvec$' },
    { id: 'cond', x: 7.8, y: 6.1, w: 2.8, h: 0.7, label: 'text encoder $\\tau_{\\thetavec}$', tone: 3 },
    { id: 'prompt', x: 7.8, y: 7.1, shape: 'text', label: 'prompt $\\yvec$' },
    {
      id: 'fwd',
      x: 9.2,
      y: 0.75,
      w: 6,
      shape: 'text',
      small: true,
      tone: 'neutral',
      label: 'forward: add noise, $q(\\zvec_t \\mid \\zvec_{t-1})$',
    },
    {
      id: 'rev',
      x: 7.8,
      y: 3.3,
      w: 6,
      shape: 'text',
      small: true,
      tone: 'neutral',
      label: 'reverse: $T$ denoising steps',
    },
  ],
  edges: [
    { from: 'x', to: 'E' },
    { from: 'E', to: 'z0' },
    { from: 'z0', to: 'dots', dashed: true },
    { from: 'dots', to: 'zt', dashed: true },
    { from: 'zt', to: 'zT', dashed: true },
    { from: 'zT:s', to: 'unet:e', via: [[11.2, 4.1]] },
    { from: 'unet', to: 'z0h' },
    { from: 'z0h', to: 'D' },
    { from: 'D', to: 'xh' },
    { from: 'prompt', to: 'cond' },
    { from: 'cond', to: 'unet', label: 'cross-attention' },
  ],
  groups: [
    { id: 'px', label: 'pixels', tone: 'neutral', around: ['x', 'xh'], pad: 0.4 },
    { id: 'lat', label: 'latent space', tone: 0, around: ['z0', 'zT', 'unet', 'z0h'], pad: 0.45 },
    { id: 'cg', label: 'conditioning', tone: 3, around: ['cond', 'prompt'], pad: 0.3, labelAt: 'bottom-right' },
  ],
}
