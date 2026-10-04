/** The `GymTrainer` renderer per `render.kind` (`renderers.tsx` holds the components). */
import type { Environment } from 'aifn-compute/foundation/contracts'
import { BanditRenderer, CartPoleRenderer, GridRenderer, PendulumRenderer, type GymRenderer } from './renderers'

/** The renderer per `render.kind` (`bandit` for an environment without a render). Extend it for new kinds. */
export const GYM_RENDERERS: Record<string, GymRenderer> = {
  grid: GridRenderer,
  bandit: BanditRenderer,
  pendulum: PendulumRenderer,
  cartpole: CartPoleRenderer,
}

/** The renderer key of an environment: its `render.kind`, or `bandit` when it has none. */
export const renderKind = (env: Environment<unknown, unknown, unknown>): string => env.render?.kind ?? 'bandit'

/** Render kinds drawn wide and short (a track): the scene takes a full-width row above the learning curves. */
export const WIDE_KINDS: ReadonlySet<string> = new Set(['cartpole'])
