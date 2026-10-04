/**
 * `aifn-methods/generative`: generative models on toy data: diffusion models (`generative/diffusion`), generative
 * adversarial networks (`generative/gan`) and energy-based models, including a classifier read as one (JEM,
 * `generative/energy`). The shared layer: densities known as labelled mixtures (`mixtureLogDensityOf`, `modeOf`) and
 * the square grid of 2-d fields (`squareGrid`).
 */

export { knownDensity, mixtureLogDensityOf, modeOf, squareGrid, type Grid2d, type LabelledDensity } from './densities'
