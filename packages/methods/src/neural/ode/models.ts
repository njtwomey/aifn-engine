/**
 * The neural ODE family on small problems: a neural ODE classifier or regressor (Chen, Rubanova, Bettencourt &
 * Duvenaud, 2018), the augmented neural ODE (Dupont, Doucet & Teh, 2019: the state padded with zeros so trajectories
 * can pass each other), the second-order neural ODE (Norcliffe et al., 2020: position and velocity, x″ = f(x, x′)),
 * time-dependent and autonomous fields, and the residual network the ODE discretises (He et al., 2016; a ResNet of N
 * blocks with step 1/N is Euler's method with untied weights). Each is the compute `OdeBlock` around an `Mlp` field and a
 * readout.
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

/** The members of the family. */
export type OdeModelKind = 'node' | 'anode' | 'sonode' | 'resnet'

/** Options of {@link odeModel}; plain data. */
export type OdeModelOptions = {
  kind: OdeModelKind
  /** Dimension of the data x. */
  dim: number
  /** Classes of a classifier (a linear readout to logits), or 0 for a regressor (the first `dim` state coordinates). */
  classes: number
  /** Hidden width of the field's two-layer MLP. Default 32. */
  hidden?: number
  /** Zero-padded extra dimensions of an augmented NODE. Default 1 (`anode` only). */
  augment?: number
  /** f(t, x) rather than f(x). Default false. */
  timeDependent?: boolean
  /** Residual blocks of a ResNet. Default 10. */
  depth?: number
  /** Field activation. Default tanh. */
  activation?: 'tanh' | 'softplus'
  /** The solver and gradient method of the ODE kinds. */
  solver?: OdeFlowOptions
}

/** Parameters: the field (an array of block fields for a ResNet) and the readout. */
export type OdeModelParams = { field: Params; readout: LinearParams | Record<string, never> }

/** A member of the family, ready to train. */
export type OdeModel = {
  options: Required<Omit<OdeModelOptions, 'solver'>> & { solver: OdeFlowOptions }
  /** Dimension of the ODE state (data, padding, velocity). */
  stateDim: number
  init(s: Stream): OdeModelParams
  /** The initial state z(0) from data x [B, dim]: x padded with zeros (ANODE) or with zero velocity (SONODE). */
  lift(x: Value): Value
  /** The field at time t on states z [B, stateDim] (for a ResNet, the block covering t). */
  field(params: OdeModelParams, t: Scalar, z: Value): Value
  /** States at `times` (each in [0, 1], increasing from 0); a ResNet returns its block outputs at k/N. */
  flow(params: OdeModelParams, x: Value, times: readonly Scalar[], options?: OdeFlowOptions): Value[]
  /** The readout of a final state: logits [B, classes] or predictions [B, dim]. */
  readout(params: OdeModelParams, z: Value): Value
  /** The ODE block (absent for a ResNet). */
  block: OdeBlockLayer<Params> | null
}

/**
 * x″ = f(x, x′) as a first-order system on z = [x, v]: z′ = [v, f(z)] (Norcliffe et al., 2020), with f a layer from
 * 2d (+1 with time) to d.
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

/** A member of the neural ODE family (see the module comment). */
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
