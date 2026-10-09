/**
 * `aifn-methods/generative/autoencoders`: the autoencoder, the variational autoencoder (with $\beta$), the conditional
 * VAE, the VQ-VAE and the RQ-VAE on small data (`models.ts`), and a streamed training run for a worker (`run.ts`).
 *
 * - Models: `autoencoder` builds any kind, `initAutoencoder` its parameters, `autoencoderLoss` its training loss;
 *   `autoencoderEncode` and `autoencoderDecode` map between inputs and codes.
 * - Discrete codes (VQ-VAE and RQ-VAE): `autoencoderQuantise` gives each input's code, one index per stage, with its
 *   coarse-to-fine latents; `latentOfCodes` turns codes back into latents; `initQuantiserCodebook` starts the
 *   codebooks from $k$-means of the encoder's outputs.
 * - Training: `autoencoderRun` streams a run's losses and checkpoints; for an RQ-VAE they include each stage's error,
 *   every row's code and the tree of code prefixes.
 */

export {
  autoencoder,
  autoencoderLoss,
  autoencoderDecode,
  autoencoderEncode,
  autoencoderQuantise,
  initAutoencoder,
  initQuantiserCodebook,
  latentOfCodes,
  type Autoencoder,
  type AutoencoderKind,
  type AutoencoderOptions,
  type AutoencoderParams,
  type AutoencoderSpec,
} from './models'
export { autoencoderRun, type AutoencoderCheckpoint, type AutoencoderRun, type AutoencoderRunOptions } from './run'
export { autoencoderFunctions } from './registry'
