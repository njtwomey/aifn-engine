/**
 * `aifn-methods/generative/autoencoders`: the autoencoder, the variational autoencoder (with β), the conditional VAE
 * and the VQ-VAE on small data (`models.ts`), and a streamed training run for a worker (`run.ts`).
 */

export {
  autoencoder,
  autoencoderLoss,
  autoencoderDecode,
  autoencoderEncode,
  initAutoencoder,
  type Autoencoder,
  type AutoencoderKind,
  type AutoencoderOptions,
  type AutoencoderParams,
  type AutoencoderSpec,
} from './models'
export { autoencoderRun, type AutoencoderCheckpoint, type AutoencoderRun, type AutoencoderRunOptions } from './run'
export { autoencoderFunctions } from './registry'
