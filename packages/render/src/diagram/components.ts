/**
 * Reusable pieces for diagrams. Each returns part of a spec with ids under a prefix, so several copies can sit in one
 * diagram; `merge` joins parts into one spec.
 */
import type { DiagramEdge, DiagramGroup, DiagramNode, DiagramSpec, Direction, Side, Tone } from './types'

type Part = { nodes?: DiagramNode[]; edges?: DiagramEdge[]; groups?: DiagramGroup[] }

export function merge(...parts: Part[]): DiagramSpec {
  return {
    nodes: parts.flatMap((p) => p.nodes ?? []),
    edges: parts.flatMap((p) => p.edges ?? []),
    groups: parts.flatMap((p) => p.groups ?? []),
  }
}

/** A random variable in a graphical model: an ink circle, shaded with `filled: true` when observed. */
export function variable(
  id: string,
  x: number,
  y: number,
  label: string,
  extra: Partial<DiagramNode> = {},
): DiagramNode {
  return { id, x, y, shape: 'circle', tone: 'ink', label, ...extra }
}

/** A factor of a factor graph: a small filled square, labelled outside on `side`. */
export function factor(id: string, x: number, y: number, label: string, side: Side = 'n'): DiagramNode {
  return { id, x, y, shape: 'factor', label, labelSide: side }
}

/** A graphical-model edge: a straight arrow, or a plain line when `directed` is false. */
export function link(from: string, to: string, directed = true, extra: Partial<DiagramEdge> = {}): DiagramEdge {
  return { from, to, route: 'straight', arrow: directed ? 'end' : 'none', ...extra }
}

/** A small operator circle: ⊕, ⊗, ⊙, σ or tanh. */
export function op(id: string, x: number, y: number, symbol: string, extra: Partial<DiagramNode> = {}): DiagramNode {
  return { id, x, y, shape: 'op', label: symbol, ...extra }
}

/** A gate or activation box, e.g. σ or tanh inside an LSTM cell. */
export function gate(id: string, x: number, y: number, label: string, tone: Tone = 0): DiagramNode {
  return { id, x, y, shape: 'box', w: 0.9, h: 0.6, label, tone }
}

/** A trapezoid projecting into a smaller space (`encoder`) or back out of it (`decoder`). */
export function projector(
  kind: 'encoder' | 'decoder',
  id: string,
  x: number,
  y: number,
  label: string,
  opts: { dir?: Direction; w?: number; h?: number; tone?: Tone } = {},
): DiagramNode {
  return {
    id,
    x,
    y,
    shape: kind,
    label,
    dir: opts.dir ?? 'right',
    w: opts.w ?? 1.8,
    h: opts.h ?? 2.4,
    tone: opts.tone ?? 0,
  }
}

/**
 * The reparameterisation block z = μ + σ ⊙ ε. `from` feeds μ and σ; the block ends at the sum node `${p}add`, which the
 * caller connects onwards. Placed with μ at (x, y − 1) and σ at (x, y + 1); the output sits at (x + 3, y).
 */
export function reparam(p: string, x: number, y: number, from: string, tone: Tone = 2): Part {
  const mu = `${p}mu`
  const sig = `${p}sig`
  const eps = `${p}eps`
  const mul = `${p}mul`
  const add = `${p}add`
  return {
    nodes: [
      { id: mu, x, y: y - 1, w: 1, h: 0.6, label: '$\\boldsymbol{\\mu}$', tone },
      { id: sig, x, y: y + 1, w: 1, h: 0.6, label: '$\\boldsymbol{\\sigma}$', tone },
      {
        id: eps,
        x: x + 1.7,
        y: y + 2.3,
        shape: 'noise',
        w: 0.7,
        h: 0.7,
        label: '$\\boldsymbol{\\epsilon}$',
        tone: 'neutral',
      },
      {
        id: `${p}epsdist`,
        x: x + 3.4,
        y: y + 2.3,
        shape: 'text',
        small: true,
        label: '$\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$',
      },
      op(mul, x + 1.7, y + 1, '$\\odot$'),
      op(add, x + 3, y, '$+$'),
    ],
    edges: [
      { from, to: `${mu}:w` },
      { from, to: `${sig}:w` },
      { from: sig, to: mul },
      { from: `${eps}:n`, to: `${mul}:s` },
      { from: `${mu}:e`, to: `${add}:n`, via: [[x + 3, y - 1]] },
      { from: `${mul}:e`, to: `${add}:s`, via: [[x + 3, y + 1]] },
    ],
    groups: [{ id: `${p}group`, label: 'reparameterisation', tone, dashed: true, around: [mu, sig, eps, mul, add] }],
  }
}
