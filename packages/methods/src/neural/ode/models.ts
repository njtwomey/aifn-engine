/**
 * The neural ODE family on small problems: a neural ODE classifier or regressor (Chen, Rubanova, Bettencourt &
 * Duvenaud, 2018), the augmented neural ODE (Dupont, Doucet & Teh, 2019: the state padded with zeros so trajectories
 * can pass each other), the second-order neural ODE (Norcliffe et al., 2020: position and velocity,
 * $\xvec'' = f(\xvec, \xvec')$), time-dependent and autonomous fields, and the residual network the ODE discretises
 * (He et al., 2016; a ResNet of $N$ blocks with step $1/N$ is Euler's method with untied weights). Each is the compute
 * `OdeBlock` around an `Mlp` field and a readout.
 *
 * The state $\zvec(t)$ starts from the data, $\zvec(0) = \xvec$ padded as the kind needs, and flows over $[0, 1]$
 * under $\zvec' = f(t, \zvec)$; the readout reads $\zvec(1)$.
 */

import type { Stream } from 'aifn-compute/foundation/random'
import { child } from 'aifn-compute/foundation/random'
import type { Params } from 'aifn-compute/foundation/pytree'
import { add, concat, mul, ones, shapeOfValue, slice, zeros, type Value } from 'aifn-compute/foundation/tensor'
import type { Scalar } from 'aifn-compute/foundation/contracts'
import type { OdeFlowOptions } from 'aifn-compute/dynamics/ode'
import {
  Linear,
  Mlp,
  OdeBlock,
  linear,
  type Layer,
  type LinearParams,
  type OdeBlockLayer,
} from 'aifn-compute/nn/layers'
import { xavierUniform } from 'aifn-compute/nn/init'

/**
 * The members of the family: `'node'` (a neural ODE), `'anode'` (augmented with zeros), `'sonode'` (second order) and
 * `'resnet'` (the residual network of `depth` Euler steps with untied weights).
 */
export type OdeModelKind = 'node' | 'anode' | 'sonode' | 'resnet'

/** Options of {@link odeModel}; plain data. */
export type OdeModelOptions = {
  kind: OdeModelKind
  /** The dimension $d$ of the data $\xvec$. */
  dim: number
  /**
   * Classes of a classifier (a linear readout to logits), or 0 for a regressor (the first `dim` state coordinates).
   */
  classes: number
  /** Hidden width of the field's two-layer MLP. Default 32. */
  hidden?: number
  /** Zero-padded extra dimensions of an augmented NODE. Default 1 (`anode` only). */
  augment?: number
  /** The field takes the time, $f(t, \zvec)$ rather than $f(\zvec)$. Default false. */
  timeDependent?: boolean
  /** Residual blocks of a ResNet. Default 10. */
  depth?: number
  /** The field's activation. Default tanh. */
  activation?: 'tanh' | 'softplus'
  /** The solver and gradient method of the ODE kinds (the ResNet ignores it). Default that of `OdeBlock`. */
  solver?: OdeFlowOptions
}

/**
 * Parameters: the `field` (the MLP's layer parameters, or for a ResNet one such array per block) and the `readout`
 * (the classifier's linear map, empty for a regressor).
 */
export type OdeModelParams = { field: Params; readout: LinearParams | Record<string, never> }

/** A member of the family, ready to train. */
export type OdeModel = {
  /** The options with their defaults filled in; `augment` is 0 except for `'anode'`. */
  options: Required<Omit<OdeModelOptions, 'solver'>> & { solver: OdeFlowOptions }
  /** The dimension of the ODE state (data, then padding or velocity). */
  stateDim: number
  /** Fresh parameters drawn from a stream (Xavier-uniform weights). */
  init(s: Stream): OdeModelParams
  /**
   * The initial state $\zvec(0)$ from data $\xvec$ $[B, d]$: $\xvec$ padded with zeros (ANODE) or with zero velocity
   * (SONODE), unchanged otherwise.
   */
  lift(x: Value): Value
  /** The field at time $t$ on states $\zvec$ $[B, S]$, $S$ = `stateDim` (for a ResNet, the block covering $t$). */
  field(params: OdeModelParams, t: Scalar, z: Value): Value
  /**
   * The states at `times` (each in $[0, 1]$, increasing from 0) from data $\xvec$ $[B, d]$, lifted first; a ResNet
   * returns its block output at the $k/N$ nearest each time.
   */
  flow(params: OdeModelParams, x: Value, times: readonly Scalar[], options?: OdeFlowOptions): Value[]
  /** The readout of a final state: logits $[B, \text{classes}]$ or predictions $[B, d]$. */
  readout(params: OdeModelParams, z: Value): Value
  /** The ODE block (absent for a ResNet). */
  block: OdeBlockLayer<Params> | null
}

/**
 * The second-order field $\xvec'' = f(\xvec, \xvec')$ as a first-order system on $\zvec = [\xvec, \vvec]$:
 * $\zvec' = [\vvec, f(\zvec)]$ (Norcliffe et al., 2020).
 *
 * @param inner The layer $f$, from the $2d$ state coordinates (and the time, when the field takes it) to $d$ outputs,
 *   the acceleration. Its parameters are the layer's.
 * @param d The dimension of the position $\xvec$: the velocity $\vvec$ is state coordinates $d$ to $2d - 1$.
 * @returns A layer mapping states $[B, 2d]$ (or $[B, 2d + 1]$ with the time) to $\zvec'$ $[B, 2d]$.
 */
function secondOrderField(inner: Layer<Params>, d: number): Layer<Params> {
  return {
    kind: 'SecondOrderField',
    label: `SecondOrder(${inner.label})`,
    init: (s) => inner.init(s),
    apply: (p, z, ctx) => {
      const v = slice(z, null, [d, 2 * d])
      return concat([v, inner.apply(p, z, ctx)], 1)
    },
  }
}

/**
 * Build a member of the neural ODE family: the field is a two-hidden-layer `Mlp` (Xavier-uniform), wrapped in the
 * compute `OdeBlock` for the ODE kinds, or `depth` such MLPs applied as the Euler steps
 * $\zvec_{k+1} = \zvec_k + \frac{1}{N} f_k(\zvec_k)$ for the ResNet; a classifier adds a linear readout to logits,
 * and a regressor reads the first $d$ state coordinates. The flow is differentiable, by backprop through the solver or
 * by the adjoint as `solver.gradient` selects.
 *
 * @param options The kind, the data's dimension and classes, and the field's width, padding, time dependence, depth,
 *   activation and solver; see `OdeModelOptions`.
 * @returns The model: its initialiser, lift, field, flow and readout, and the `OdeBlock` (null for a ResNet).
 *
 * @example The flow of a point under a known linear field matches $e^{\Amat t}\xvec$
 * // With tiny first weights e, tanh(e z) / e is close to z: the field is z -> A z, A = [[0, -1], [1, 0]] a rotation
 * const model = odeModel({ kind: 'node', dim: 2, classes: 0, hidden: 2 })
 * const e = 1e-3
 * const layer = (w) => ({ weight: tensor(w), bias: tensor([0, 0]) })
 * const field = [layer([[e, 0], [0, e]]), {}, layer([[1, 0], [0, 1]]), {}, layer([[0, 1 / e], [-1 / e, 0]])]
 * const [, z1] = model.flow({ field, readout: {} }, tensor([[1, 0]]), [0, 1])
 * print('flow to t = 1:', z1)
 * print('exp(A) x:', [Math.cos(1), Math.sin(1)])
 *
 * @example The augmented and second-order states, and a classifier's logits
 * const anode = odeModel({ kind: 'anode', dim: 1, classes: 0, augment: 1 })
 * const sonode = odeModel({ kind: 'sonode', dim: 1, classes: 2 })
 * const x = tensor([[0.5], [-1]])
 * print('ANODE state:', anode.lift(x), ' SONODE state (position, velocity):', sonode.lift(x))
 * const p = sonode.init(stream(0))
 * print('SONODE logits at t = 1:', sonode.readout(p, sonode.flow(p, x, [0, 1])[1]))
 */
export function odeModel(options: OdeModelOptions): OdeModel {
  const {
    kind,
    dim,
    classes,
    hidden = 32,
    augment = kind === 'anode' ? 1 : 0,
    timeDependent = false,
    depth = 10,
    activation = 'tanh',
    solver = {},
  } = options
  const pad = kind === 'anode' ? augment : 0
  const stateDim = kind === 'sonode' ? 2 * dim : dim + pad
  const tIn = timeDependent ? 1 : 0
  const init = xavierUniform()
  const mlp = (inF: number, out: number) => Mlp([inF, hidden, hidden, out], { activation, init })
  const fieldLayer: Layer<Params> =
    kind === 'sonode' ? secondOrderField(mlp(stateDim + tIn, dim), dim) : mlp(stateDim + tIn, stateDim)
  const block = kind === 'resnet' ? null : OdeBlock(fieldLayer, { ...solver, timeDependent })
  const blocks = Array.from({ length: depth }, () => mlp(stateDim + tIn, stateDim))
  const head = classes > 0 ? Linear(stateDim, classes, { init }) : null

  const lift = (x: Value): Value => {
    const extra = stateDim - dim
    return extra === 0 ? x : concat([x, zeros([shapeOfValue(x)[0], extra])], 1)
  }
  const withTime = (t: Scalar, z: Value) => (timeDependent ? concat([z, mul(t, ones([shapeOfValue(z)[0], 1]))], 1) : z)
  const blockAt = (t: Scalar) => Math.min(depth - 1, Math.max(0, Math.floor(t * depth + 1e-9)))
  const field = (params: OdeModelParams, t: Scalar, z: Value): Value =>
    block
      ? block.field(params.field, t, z)
      : blocks[blockAt(t)].apply((params.field as Params[][])[blockAt(t)], withTime(t, z))
  const flow = (params: OdeModelParams, x: Value, times: readonly Scalar[], override?: OdeFlowOptions): Value[] => {
    const z0 = lift(x)
    if (block) return block.flow(params.field, z0, times, override)
    // A ResNet: z_{k+1} = z_k + (1/N) f_k(z_k), its states at k/N read at the nearest requested times.
    const states: Value[] = [z0]
    let z = z0
    for (let k = 0; k < depth; k++) {
      z = add(z, mul(1 / depth, blocks[k].apply((params.field as Params[][])[k], withTime(k / depth, z))))
      states.push(z)
    }
    return times.map((t) => states[Math.round(Math.min(1, Math.max(0, t)) * depth)])
  }
  const readout = (params: OdeModelParams, z: Value): Value =>
    head
      ? linear(z, (params.readout as LinearParams).weight, (params.readout as LinearParams).bias)
      : slice(z, null, [0, dim])
  return {
    options: { kind, dim, classes, hidden, augment: pad, timeDependent, depth, activation, solver },
    stateDim,
    init: (s) => ({
      field: block ? fieldLayer.init(child(s, 'field')) : blocks.map((b, k) => b.init(child(child(s, 'blocks'), k))),
      readout: head ? head.init(child(s, 'readout')) : {},
    }),
    lift,
    field,
    flow,
    readout,
    block,
  }
}
