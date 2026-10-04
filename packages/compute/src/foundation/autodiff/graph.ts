/**
 * Inspectable computation graphs: `traceGraph(f, x)` records f at x, runs the reverse sweep and returns every node
 * with its value and adjoint, in evaluation (topological) order, so a view can step through the forward pass and then
 * the backward pass node by node.
 */

import { ones, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { ReverseInterpreter } from './reverse'
import { treeFlatten, treeUnflatten, zerosLike } from 'aifn-compute/foundation/pytree'

/**
 * An input of a graph node: another node, or a constant (a number or tensor that does not depend on x). A node input
 * carries what travels along its edge in the backward pass: `partial`, the local derivative ∂(this node)/∂(input) (for
 * a scalar node; null for a tensor node or a primitive without a rule), and `message`, the adjoint contribution
 * pulled back to the input, adjoint · ∂(this node)/∂(input) (null when this node has no adjoint). An input's adjoint is
 * the sum of the messages on its outgoing edges.
 */
export type GraphInput =
  { node: number; partial: number | Tensor | null; message: number | Tensor | null } | { constant: number | Tensor }

/** One node of a recorded computation graph. */
export type GraphNode = {
  /** Position in evaluation order; inputs always have smaller ids. */
  id: number
  /** The primitive's name, or `input` for an input leaf. */
  op: string
  /** For inputs, the leaf's path in x (`x`, `x[1]`, `w.bias`). */
  label?: string
  inputs: GraphInput[]
  /** The forward value. */
  value: number | Tensor
  /** ∂(output)/∂(this node), the cotangent the reverse sweep assigns it; null when the output does not depend on it. */
  adjoint: number | Tensor | null
  /** False for a primitive without a derivative rule. */
  differentiable: boolean
}

/** A recorded computation graph (see `traceGraph`). */
export type Graph = {
  /** Every recorded node in evaluation order: the forward pass visits them by increasing id, the backward by decreasing. */
  nodes: GraphNode[]
  /** Ids of the input leaves, in the order of x's leaves. */
  inputs: number[]
  /** Id of the output node, or −1 when the output does not depend on x. */
  output: number
  /** f(x). */
  value: number | Tensor
  /** The gradient, with the structure of x. */
  grad: unknown
}

/**
 * Record the computation graph of a scalar function f at x (a number, tensor or pytree) and run its reverse sweep.
 * Every primitive application becomes a node, with its inputs, forward value and adjoint. Throws if a primitive on a
 * path to the output has no derivative rule.
 */
export function traceGraph<T>(f: (x: T) => Value, x: T): Graph {
  const rev = new ReverseInterpreter()
  const flat = treeFlatten(x, 'x')
  const inputs = flat.leaves.map((leaf, i) => rev.input(unwrap(leaf), flat.paths[i]))
  const y = f(treeUnflatten(flat.treedef, inputs) as T)
  const raw = unwrap(y)
  if (typeof raw !== 'number' && raw.shape.length !== 0)
    throw new ShapeError('traceGraph', 'traceGraph: f must return a scalar')
  const seed = typeof raw === 'number' ? 1 : ones([])
  const { cotangents, all } = rev.backward([y], [seed], inputs, true)
  const raws = (vs: (Value | null)[]) => vs.map((v) => (v === null ? null : unwrap(v)))
  const nodes: GraphNode[] = rev.records.map((record, id) => {
    const adjoint = all.get(id)
    const adj = adjoint === undefined ? null : unwrap(adjoint)
    const output = unwrap(record.output)
    const scalar = typeof output === 'number' || output.shape.length === 0
    const rule = record.primitive?.vjp ?? null
    // Inspection only: each rule runs again, once for the local partials and once for the messages.
    const partials = scalar && rule !== null ? raws(rev.pullback(id, typeof output === 'number' ? 1 : ones([]))) : null
    const messages = adj !== null && rule !== null ? raws(rev.pullback(id, adj)) : null
    return {
      id,
      op: record.primitive?.name ?? 'input',
      ...(record.label === undefined ? {} : { label: record.label }),
      inputs: record.inputs.map((v, i) =>
        record.sources[i] >= 0
          ? { node: record.sources[i], partial: partials?.[i] ?? null, message: messages?.[i] ?? null }
          : { constant: unwrap(v) },
      ),
      value: output,
      adjoint: adj,
      differentiable: record.primitive === null || rule !== null,
    }
  })
  const grads = cotangents.map((g, i) => (g === null ? zerosLike(unwrap(flat.leaves[i])) : unwrap(g)))
  return {
    nodes,
    inputs: inputs.map((t) => t.record),
    output: rev.owns(y) ? y.record : -1,
    value: raw,
    grad: treeUnflatten(flat.treedef, grads),
  }
}
