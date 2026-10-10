/**
 * `aifn-methods/generative`: generative models on toy data, with the densities and grids their figures share.
 *
 * - `generative/diffusion`: diffusion models, with noise schedules and forward SDEs, the forward process, noise
 *   predictors (exact for Gaussian mixtures, or a learned MLP) and the DDPM, DDIM, reverse-SDE and probability-flow
 *   samplers.
 * - `generative/gan`: generative adversarial networks under the minimax, non-saturating, Wasserstein and hinge games,
 *   with mode coverage and the optimal discriminator against a known density.
 * - `generative/flows`: RealNVP, a normalising flow of affine coupling layers, for exact likelihoods and sampling.
 * - `generative/autoencoders`: the autoencoder, the variational autoencoder (with $\beta$), the conditional VAE,
 *   and the VQ-VAE and RQ-VAE, whose codes are discrete.
 * - `generative/energy`: energy-based models, centred on JEM, a classifier whose logits are read as an energy.
 * - `generative/boltzmann`: networks of binary units: the restricted Boltzmann machine, the deep belief network, and
 *   the classical and modern Hopfield networks.
 *
 * The shared layer, used by the modules' diagnostics and figures: densities known as labelled mixtures
 * $p(\xvec) = \sum_j \pi_j \, p(\xvec \mid j)$ (`knownDensity` reads one from a dataset's truth;
 * `mixtureLogDensityOf` and `modeOf` evaluate it), and the square grid of 2-d fields (`squareGrid`).
 */

export { knownDensity, mixtureLogDensityOf, modeOf, squareGrid, type Grid2d, type LabelledDensity } from './densities'
