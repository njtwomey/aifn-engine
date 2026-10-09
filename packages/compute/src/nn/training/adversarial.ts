/**
 * Two-player training as a traceable algorithm: alternating gradient steps on a critic (a discriminator) and a
 * generator, each with its own pytree update rule, $k$ critic steps per generator step (Goodfellow et al., 2014,
 * Algorithm 1; Gulrajani et al., 2017, Algorithm 1 with $k = 5$).
 *
 * The losses are the caller's, so the same loop trains a GAN under any of the games of `aifn-compute/learning/losses`
 * (`discriminatorLoss`, `generatorLoss`) or any other two-player objective (domain-adversarial training). Each loss
 * sees the other player's parameters as constants.
 */

import type { Scalar, Size, Status, StepContext } from 'aifn-compute/foundation/contracts'
import { valueAndGrad, type ValueAndGrad } from 'aifn-compute/foundation/autodiff'
import type { Params } from 'aifn-compute/foundation/pytree'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, unwrap, type Value } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { adamRule, applyUpdates, globalNorm, type UpdateRule } from 'aifn-compute/optim/first-order'

/** Options of `adversarialTraining`. */
export type AdversarialTrainingOptions<G extends Params, C extends Params> = {
  /**
   * The critic's loss of its parameters, given the generator's (constants) and a stream for this evaluation's draws.
   */
  criticLoss: (critic: C, generator: G, s: Stream) => Value
  /** The generator's loss of its parameters, given the critic's (constants) and a stream. */
  generatorLoss: (generator: G, critic: C, s: Stream) => Value
  /** The critic's update rule. Default `adamRule({ stepSize: 1e-3 })`. */
  criticOptimizer?: UpdateRule<unknown>
  /** The generator's update rule. Default `adamRule({ stepSize: 1e-3 })`. */
  generatorOptimizer?: UpdateRule<unknown>
  /** Critic updates per generator update, $k$. Default 1. */
  criticSteps?: Size
  /** Flag divergence when a loss is not finite or exceeds this in size. Default 1e8. */
  divergeAbove?: Scalar
}

/** The state of `adversarialTraining` after $t$ generator updates. */
export interface AdversarialTrainingState<G extends Params, C extends Params> extends Status {
  /** Generator updates so far (each preceded by $k$ critic updates). */
  readonly t: Size
  /** The generator's parameters. */
  readonly generator: G
  /** The critic's parameters. */
  readonly critic: C
  /** The generator's update rule's state. */
  readonly generatorOptimizer: unknown
  /** The critic's update rule's state. */
  readonly criticOptimizer: unknown
  /** The critic's loss before its last update (NaN at $t = 0$). */
  readonly criticLoss: Scalar
  /** The generator's loss before its last update (NaN at $t = 0$). */
  readonly generatorLoss: Scalar
  /** Global gradient norm of the last critic update (NaN at $t = 0$). */
  readonly criticGradNorm: Scalar
  /** Global gradient norm of the last generator update (NaN at $t = 0$). */
  readonly generatorGradNorm: Scalar
  /** Whether either loss is not finite or exceeds `divergeAbove` in absolute value; a run stops here. */
  readonly diverged: boolean
}

/**
 * A loss as a number.
 *
 * @param v A number, or a rank-0 (or traced) value.
 * @returns Its value; for a tensor, its first entry.
 */
const scalarOf = (v: Value): number => {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : toFlat(raw)[0]
}

/**
 * Alternating two-player training. `init` takes `{ generator, critic }` (initial parameter trees). Step $t$ makes $k$
 * critic updates, the $j$-th on `criticLoss` with the stream `child(s, 'critic', j)`, then one generator update on
 * `generatorLoss` against the updated critic with `child(s, 'generator')`, where `s` is the step's stream; so a step
 * is a pure function of its state and context.
 *
 * @param options The two losses, their update rules, $k$ and the divergence threshold.
 * @returns The algorithm, to run with `run` or `trace` from `{ generator, critic }`.
 *
 * @example A min-max game whose equilibrium is $g = 2$, $c = 0$: the critic plays $c = g - 2$, the generator follows
 * const criticLoss = (c, g) => sub(mul(0.5, square(c.c)), mul(c.c, sub(g.g, 2)))
 * const generatorLoss = (g, c) => mul(c.c, sub(g.g, 2))
 * const alg = adversarialTraining({ criticLoss, generatorLoss })
 * const tr = trace(alg, { generator: { g: 2.5 }, critic: { c: 0 } }, 1500, {
 *   every: 300,
 *   record: { g: (s) => s.generator.g, c: (s) => s.critic.c },
 * })
 * print('steps:', tr.index)
 * print('g:', tr.series.g)
 * print('c:', tr.series.c)
 */
export function adversarialTraining<G extends Params, C extends Params>(
  options: AdversarialTrainingOptions<G, C>,
): Algorithm<{ generator: G; critic: C }, AdversarialTrainingState<G, C>> {
  const { criticSteps = 1, divergeAbove = 1e8 } = options
  const criticRule = options.criticOptimizer ?? (adamRule({ stepSize: 1e-3 }) as UpdateRule<unknown>)
  const generatorRule = options.generatorOptimizer ?? (adamRule({ stepSize: 1e-3 }) as UpdateRule<unknown>)
  const critic: (c: C, g: G, s: Stream) => ValueAndGrad<Value, unknown> = valueAndGrad(
    (c: C, g: G, s: Stream) => options.criticLoss(c, g, s),
    {},
  )
  const generator: (g: G, c: C, s: Stream) => ValueAndGrad<Value, unknown> = valueAndGrad(
    (g: G, c: C, s: Stream) => options.generatorLoss(g, c, s),
    {},
  )
  const bad = (v: number) => !Number.isFinite(v) || Math.abs(v) > divergeAbove
  return {
    name: `adversarial-${generatorRule.name}`,
    init: (start) => ({
      t: 0,
      generator: start.generator,
      critic: start.critic,
      generatorOptimizer: generatorRule.init(start.generator),
      criticOptimizer: criticRule.init(start.critic),
      criticLoss: NaN,
      generatorLoss: NaN,
      criticGradNorm: NaN,
      generatorGradNorm: NaN,
      diverged: false,
    }),
    step: (state, ctx: StepContext) => {
      let c = state.critic
      let cOpt = state.criticOptimizer
      let criticLoss = NaN
      let criticGradNorm = NaN
      for (let j = 0; j < criticSteps; j++) {
        const out = critic(c, state.generator, child(ctx.stream, 'critic', j))
        const g = out.grad as unknown as Params
        criticLoss = scalarOf(out.value as Value)
        criticGradNorm = globalNorm(g)
        const { updates, state: next } = criticRule.update(g, cOpt, c)
        c = applyUpdates(c, updates)
        cOpt = next
      }
      const out = generator(state.generator, c, child(ctx.stream, 'generator'))
      const g = out.grad as unknown as Params
      const generatorLoss = scalarOf(out.value as Value)
      const { updates, state: gOpt } = generatorRule.update(g, state.generatorOptimizer, state.generator)
      return {
        t: state.t + 1,
        generator: applyUpdates(state.generator, updates),
        critic: c,
        generatorOptimizer: gOpt,
        criticOptimizer: cOpt,
        criticLoss,
        generatorLoss,
        criticGradNorm,
        generatorGradNorm: globalNorm(g),
        diverged: bad(criticLoss) || bad(generatorLoss),
      }
    },
  }
}
