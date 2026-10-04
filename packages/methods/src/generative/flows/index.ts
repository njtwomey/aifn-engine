/**
 * `aifn-methods/generative/flows`: RealNVP, a normalising flow of affine coupling layers (`realnvp.ts`, on compute's
 * `affineCouplingBijector`), and a streamed training run (`run.ts`).
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
