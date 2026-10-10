/**
 * `aifn-methods/neural/contrastive`: a tiny CLIP, two MLP encoders aligned by the symmetric InfoNCE loss with a
 * learnable temperature, and the measures of what it learns.
 *
 * - The model: `TwoTower` (an MLP encoder for each view into one shared space) and `temperatureOf` (a fixed
 *   $\tau$, or CLIP's learned $\log(1/\tau)$ with its scale capped).
 * - Training: `contrastiveLoss` (the symmetric InfoNCE over in-batch negatives), `contrastiveTraining` (a traceable
 *   Adam loop), `contrastiveTrainingRun` (a generator of scored checkpoints for a worker) and `contrastiveAblation`
 *   (one run per batch size and temperature).
 * - Using the embedding: `embed` (unit-norm embeddings of one view), `similarities`, `retrieve` (the nearest keys)
 *   and `zeroShot` (the class whose prototype is nearest).
 * - Scoring: `scoreEmbedding` gives the InfoNCE, Wang & Isola's alignment and uniformity, zero-shot accuracy on seen
 *   and held-out classes, and top-1 retrieval.
 *
 * The data are paired views row by row (`ContrastivePairs`, the shape of `aifn-methods/data`'s `pairedShapes`).
 * Embeddings are normalised onto the unit sphere, so similarities are cosines. Runs are deterministic from their seed.
 */

export {
  contrastiveAblation,
  contrastiveLoss,
  contrastiveTraining,
  contrastiveTrainingRun,
  embed,
  retrieve,
  scoreEmbedding,
  similarities,
  temperatureOf,
  TwoTower,
  zeroShot,
  type ContrastiveAblationOptions,
  type ContrastiveAblationRun,
  type ContrastiveCheckpoint,
  type ContrastiveLossOptions,
  type ContrastivePairs,
  type ContrastiveRunOptions,
  type ContrastiveScores,
  type ContrastiveSnapshot,
  type ContrastiveTrainingOptions,
  type TemperatureSetting,
  type TwoTowerConfig,
  type TwoTowerParams,
} from './clip'
