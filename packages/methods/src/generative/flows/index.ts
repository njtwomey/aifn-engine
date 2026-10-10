/**
 * `aifn-methods/generative/flows`: normalising flows on 2-d data, RealNVP's affine coupling layers and their training.
 *
 * - The flow: `realNvp` (alternating masks, MLP conditioners, bounded log-scales on `aifn-compute`'s
 *   `affineCouplingBijector`), `initRealNvp` (every layer starts as the identity), `couplingLayer` (one layer as a
 *   bijector).
 * - Densities and samples: `flowLogDensity` (differentiable, the training objective) and `flowLogDensityValues`
 *   (plain numbers), `flowForward` (the data after each layer), `flowSample` (base points through the inverses).
 * - Training: `realNvpRun`, a generator of plain-data snapshots of Adam on the negative log-likelihood, with the
 *   density on a grid, samples and the layer-by-layer picture at checkpoints.
 * - The registry: `flowFunctions`.
 *
 * The flow maps data $\xvec$ to base $\zvec \sim \Gauss(\zeros, \Imat)$; parameters are one conditioner's per layer.
 */

export {
  couplingLayer,
  flowForward,
  flowLogDensity,
  flowLogDensityValues,
  flowSample,
  initRealNvp,
  realNvp,
  type RealNvp,
  type RealNvpOptions,
} from './realnvp'
export { realNvpRun, type RealNvpCheckpoint, type RealNvpRun, type RealNvpRunOptions } from './run'
export { flowFunctions } from './registry'
