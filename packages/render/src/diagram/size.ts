import type { DiagramNode, Shape } from './types'

/** Default width and height of each shape, in grid units. */
export const DEFAULT_SIZE: Record<Shape, [number, number]> = {
  box: [2.4, 0.8],
  pill: [2.4, 0.8],
  circle: [0.9, 0.9],
  latent: [0.9, 0.9],
  noise: [0.9, 0.9],
  op: [0.5, 0.5],
  factor: [0.26, 0.26],
  encoder: [1.6, 2],
  decoder: [1.6, 2],
  stack: [2.4, 0.8],
  dot: [0.14, 0.14],
  text: [1.8, 0.6],
}

/** A node's width and height in grid units: its own, else its shape's default. */
export function nodeSize(n: Pick<DiagramNode, 'shape' | 'w' | 'h'>): [number, number] {
  const [w, h] = DEFAULT_SIZE[n.shape ?? 'box']
  return [n.w ?? w, n.h ?? h]
}
