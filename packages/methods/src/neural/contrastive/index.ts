/**
 * `aifn-methods/neural/contrastive`: a tiny CLIP: two MLP encoders aligned by the symmetric InfoNCE loss with a
 * learnable temperature, trained on paired views (`aifn-methods/data`'s `pairedShapes`), with zero-shot
 * classification, retrieval, Wang & Isola's alignment and uniformity, and an ablation over batch size and temperature.
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
