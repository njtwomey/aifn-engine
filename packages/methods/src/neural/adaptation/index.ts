/**
 * `aifn-methods/neural/adaptation`: low-rank adaptation of a weight change, the toy problem behind LoRA's rank.
 *
 * - Targets: `lowRankTarget`, a square change $\Delta\Wmat$ with random singular vectors and a power-law
 *   ($\sigma_i = i^{-d}$) or low-rank-plus-noise spectrum.
 * - Fits: `lowRankFit`, the adapter $s\,\Bmat\Amat$ of rank $r$ trained on
 *   $\frac{1}{2}\lVert s\,\Bmat\Amat - \Delta\Wmat \rVert_F^2$ by SGD or Adam, from LoRA's start ($\Bmat = \mathbf{0}$)
 *   or PiSSA's (the target's leading singular directions), with the scale $s = \alpha / r$ (LoRA) or
 *   $\alpha / \sqrt{r}$ (rank-stabilised LoRA); it reports the loss per step, the adapter's spectrum at checkpoints and
 *   the Eckart–Young floor $\frac{1}{2}\sum_{i > r} \sigma_i^2$; `lowRankLoss` gives one step's loss and gradients.
 *
 * Everything is deterministic from its seed, and small enough for a browser: a $64 \times 64$ fit of a few hundred
 * steps takes a fraction of a second.
 */

export {
  lowRankFit,
  lowRankLoss,
  lowRankTarget,
  type LowRankCheckpoint,
  type LowRankLoss,
  type LowRankFitOptions,
  type LowRankFitResult,
  type LowRankTargetOptions,
} from './low-rank'
export { adaptationFunctions } from './registry'
