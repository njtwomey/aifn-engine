/**
 * Tiny fits for the model registry's protocol test: for every registered estimator factory, how to make an estimator
 * (hyperparameters at their `hyper` defaults where the factory needs more than those) and the data to fit it on.
 */
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset, type Estimator } from 'aifn-compute/learning/estimators'
import { rbf } from 'aifn-compute/learning/kernels'
import { family } from 'aifn-compute/probability/likelihoods'
import * as gam from 'aifn-methods/learning/generalised/gam'
import * as glm from 'aifn-methods/learning/generalised/glm'
import * as ordinal from 'aifn-methods/learning/generalised/ordinal'
import * as gp from 'aifn-methods/learning/gaussian-processes'
import * as bayes from 'aifn-methods/learning/generative-classifiers'
import * as kernels from 'aifn-methods/learning/kernel-methods'
import * as linear from 'aifn-methods/learning/linear'
import * as mdn from 'aifn-methods/learning/mixture-density'
import * as neighbours from 'aifn-methods/learning/neighbours'
import * as pre from 'aifn-methods/learning/preprocessing'
import * as reductions from 'aifn-methods/learning/reductions'
import * as trees from 'aifn-methods/learning/trees-and-ensembles'
import * as clustering from 'aifn-methods/unsupervised/clustering'
import * as embedLinear from 'aifn-methods/unsupervised/embedding/linear'
import * as manifold from 'aifn-methods/unsupervised/embedding/manifold'
import * as neighbour from 'aifn-methods/unsupervised/embedding/neighbour'
import * as languageModels from 'aifn-methods/neural/language-models'

/** 24 points in two noisy classes: x [24, 2], y in {0, 1}. */
function twoClasses(): { x: Tensor; y: Tensor } {
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i < 24; i++) {
    const c = i % 2
    const a = (i * 2.399) % (2 * Math.PI)
    xs.push(c * 2 + 0.6 * Math.cos(a), c * 1.5 + 0.6 * Math.sin(a))
    ys.push(c)
  }
  return { x: fromData(Float64Array.from(xs), [24, 2]), y: fromData(Int32Array.from(ys)) }
}

/** 30 points in three classes. */
function threeClasses(): { x: Tensor; y: Tensor } {
  const xs: number[] = []
  const ys: number[] = []
  const centres = [
    [0, 0],
    [3, 0],
    [0, 3],
  ]
  for (let i = 0; i < 30; i++) {
    const c = i % 3
    const a = (i * 2.399) % (2 * Math.PI)
    xs.push(centres[c][0] + 0.7 * Math.cos(a), centres[c][1] + 0.7 * Math.sin(a))
    ys.push(c)
  }
  return { x: fromData(Float64Array.from(xs), [30, 2]), y: fromData(Int32Array.from(ys)) }
}

/** 30 points of a 1-d regression problem in two features. */
function regression(): { x: Tensor; y: Tensor } {
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i < 30; i++) {
    const u = i / 29
    const v = ((i * 7) % 30) / 29
    xs.push(u, v)
    ys.push(Math.sin(4 * u) + 0.5 * v + 0.05 * Math.cos(13 * i))
  }
  return { x: fromData(Float64Array.from(xs), [30, 2]), y: fromData(Float64Array.from(ys)) }
}

/** Positive counts for count models. */
function counts(): { x: Tensor; y: Tensor } {
  const { x } = regression()
  const y = fromData(Float64Array.from({ length: 30 }, (_, i) => (i * 7) % 5))
  return { x, y }
}

const bin = twoClasses()
const tri = threeClasses()
const reg = regression()
const cnt = counts()
const unlabelled = dataset(tri.x)
const kernel = rbf({ lengthscale: 1 })
const binaryBase = glm.logisticRegression({ l2: 1 })

/** An estimator and the data to fit it on. */
export interface Fixture {
  make: () => Estimator<never, unknown>
  data: unknown
}

const f = (make: () => Estimator<never, unknown>, data: unknown): Fixture => ({ make, data })

/** Every registered model key → its tiny fit. The protocol test fails on a registered key without one. */
/** A short character corpus. */
const corpus = languageModels.charCorpus('the cat sat on the mat. the dog sat on the log.')

export const MODEL_FIXTURES: Record<string, Fixture> = {
  // learning/linear
  linearRegression: f(() => linear.linearRegression(), dataset(reg.x, reg.y)),
  perceptron: f(() => linear.perceptron(), dataset(bin.x, bin.y)),
  // learning/generalised
  logisticRegression: f(() => glm.logisticRegression(), dataset(bin.x, bin.y)),
  multinomialLogisticRegression: f(() => glm.multinomialLogisticRegression(), dataset(tri.x, tri.y)),
  glm: f(() => glm.glm({ family: family('poisson') }), { x: cnt.x, y: cnt.y }),
  negativeBinomialRegression: f(() => glm.negativeBinomialRegression(), { x: cnt.x, y: cnt.y }),
  ordinalRegression: f(() => ordinal.ordinalRegression({ l2: 1 }), dataset(tri.x, tri.y)),
  thresholdOrdinalRegression: f(() => ordinal.thresholdOrdinalRegression(), dataset(tri.x, tri.y)),
  binaryDecomposition: f(() => ordinal.binaryDecomposition(), dataset(tri.x, tri.y)),
  deepOrdinalRegression: f(() => ordinal.deepOrdinalRegression({ hidden: [4], steps: 20 }), dataset(tri.x, tri.y)),
  gam: f(() => gam.gam({ terms: [gam.s(0, { k: 6 }), gam.s(1, { k: 6 })] }), { x: reg.x, y: reg.y }),
  expectileGam: f(() => gam.expectileGam({ terms: [gam.s(0, { k: 6 })], tau: 0.8 }), { x: reg.x, y: reg.y }),
  explainableBoostingMachine: f(() => gam.explainableBoostingMachine({ rounds: 20 }), dataset(reg.x, reg.y)),
  // learning/generative-classifiers
  gaussianNaiveBayes: f(() => bayes.gaussianNaiveBayes(), dataset(tri.x, tri.y)),
  multinomialNaiveBayes: f(
    () => bayes.multinomialNaiveBayes(),
    dataset(
      tensor([
        [2, 0],
        [3, 1],
        [0, 2],
        [1, 3],
      ]),
      fromData(Int32Array.from([0, 0, 1, 1])),
    ),
  ),
  bernoulliNaiveBayes: f(() => bayes.bernoulliNaiveBayes(), dataset(bin.x, bin.y)),
  linearDiscriminant: f(() => bayes.linearDiscriminant(), dataset(tri.x, tri.y)),
  quadraticDiscriminant: f(() => bayes.quadraticDiscriminant(), dataset(tri.x, tri.y)),
  // learning/kernel-methods
  supportVectorMachine: f(() => kernels.supportVectorMachine({ kernel }), dataset(bin.x, bin.y)),
  linearSvm: f(() => kernels.linearSvm(), dataset(bin.x, bin.y)),
  crammerSinger: f(() => kernels.crammerSinger(), dataset(tri.x, tri.y)),
  // learning/mixture-density
  mixtureDensityNetwork: f(
    () => mdn.mixtureDensityNetwork({ components: 2, hidden: 4, steps: 5 }),
    dataset(reg.x, reg.y),
  ),
  // learning/gaussian-processes
  gaussianProcessRegressor: f(
    () => gp.gaussianProcessRegressor({ kernel, noiseVariance: 0.01 }),
    dataset(reg.x, reg.y),
  ),
  sparseGaussianProcessRegressor: f(
    () => gp.sparseGaussianProcessRegressor({ kernel, inducing: 5, noiseVariance: 0.01 }),
    dataset(reg.x, reg.y),
  ),
  relevanceVectorMachine: f(() => gp.relevanceVectorMachine({ kernel }), dataset(reg.x, reg.y)),
  gpClassifier: f(() => gp.gpClassifier({ kernel }), dataset(bin.x, bin.y)),
  gpOrdinalRegression: f(() => gp.gpOrdinalRegression({ kernel, hyperSteps: 10 }), dataset(tri.x, tri.y)),
  // learning/trees-and-ensembles
  decisionTree: f(() => trees.decisionTree(), dataset(tri.x, tri.y)),
  regressionTree: f(() => trees.regressionTree(), dataset(reg.x, reg.y)),
  randomForest: f(() => trees.randomForest({ trees: 3 }), dataset(tri.x, tri.y)),
  adaBoost: f(() => trees.adaBoost({ rounds: 5 }), dataset(bin.x, bin.y)),
  gradientBoosting: f(() => trees.gradientBoosting({ stages: 5 }), dataset(reg.x, reg.y)),
  // learning/neighbours
  kNearestNeighbours: f(() => neighbours.kNearestNeighbours(), dataset(tri.x, tri.y)),
  kNearestNeighboursRegression: f(() => neighbours.kNearestNeighboursRegression(), dataset(reg.x, reg.y)),
  // learning/reductions
  oneVersusRest: f(() => reductions.oneVersusRest(binaryBase), dataset(tri.x, tri.y)),
  oneVersusOne: f(() => reductions.oneVersusOne(binaryBase), dataset(tri.x, tri.y)),
  outputCode: f(() => reductions.outputCode(binaryBase, reductions.exhaustiveCode(3)), dataset(tri.x, tri.y)),
  nestedDichotomies: f(() => reductions.nestedDichotomies(binaryBase), dataset(tri.x, tri.y)),
  // learning/preprocessing
  standardScaler: f(() => pre.standardScaler(), { x: reg.x }),
  minMaxScaler: f(() => pre.minMaxScaler(), { x: reg.x }),
  robustScaler: f(() => pre.robustScaler(), { x: reg.x }),
  maxAbsScaler: f(() => pre.maxAbsScaler(), { x: reg.x }),
  powerTransform: f(() => pre.powerTransform(), { x: reg.x }),
  whitening: f(() => pre.whitening(), { x: reg.x }),
  simpleImputer: f(() => pre.simpleImputer(), { x: reg.x }),
  polynomialFeatures: f(() => pre.polynomialFeatures(), { x: reg.x }),
  splineFeatures: f(() => pre.splineFeatures(), { x: reg.x }),
  randomFourierFeatures: f(() => pre.randomFourierFeatures(), { x: reg.x }),
  oneHotEncoder: f(() => pre.oneHotEncoder(), { x: ['a', 'b', 'c', 'a'] }),
  ordinalEncoder: f(() => pre.ordinalEncoder(), { x: ['a', 'b', 'c', 'a'] }),
  targetEncoder: f(() => pre.targetEncoder(), { x: ['a', 'b', 'c', 'a', 'b', 'c'], y: tensor([1, 2, 3, 1, 2, 4]) }),
  // unsupervised/clustering
  kmeans: f(() => clustering.kmeans({ k: 3 }), unlabelled),
  miniBatchKMeans: f(() => clustering.miniBatchKMeans({ k: 3 }), unlabelled),
  kMedoids: f(() => clustering.kMedoids({ k: 3 }), unlabelled),
  dbscan: f(() => clustering.dbscan({ eps: 1 }), unlabelled),
  optics: f(() => clustering.optics(), unlabelled),
  meanShift: f(() => clustering.meanShift({ bandwidth: 1.5 }), unlabelled),
  agglomerative: f(() => clustering.agglomerative({ clusters: 3 }), unlabelled),
  gaussianMixture: f(() => clustering.gaussianMixture({ k: 3 }), unlabelled),
  spectralClustering: f(() => clustering.spectralClustering({ k: 3 }), unlabelled),
  // unsupervised/embedding
  pca: f(() => embedLinear.pca(), unlabelled),
  kernelPca: f(() => embedLinear.kernelPca({ kernel }), unlabelled),
  metricMds: f(() => embedLinear.metricMds(), unlabelled),
  factorAnalysis: f(() => embedLinear.factorAnalysis({ latent: 1, maxSteps: 50 }), unlabelled),
  probabilisticPca: f(() => embedLinear.probabilisticPca({ latent: 1 }), unlabelled),
  fastIca: f(() => embedLinear.fastIca({ maxSteps: 50 }), unlabelled),
  diffusionMap: f(() => manifold.diffusionMap(), unlabelled),
  selfOrganisingMap: f(() => manifold.selfOrganisingMap({ rows: 3, cols: 3, epochs: 5 }), unlabelled),
  isomap: f(() => manifold.isomap({ neighbours: 6 }), unlabelled),
  laplacianEigenmaps: f(() => manifold.laplacianEigenmaps({ neighbours: 6 }), unlabelled),
  locallyLinearEmbedding: f(() => manifold.locallyLinearEmbedding({ neighbours: 6 }), unlabelled),
  tsne: f(() => neighbour.tsne({ perplexity: 5, iterations: 50 }), unlabelled),
  umap: f(() => neighbour.umap({ neighbours: 6, epochs: 20 }), unlabelled),
  pacmap: f(() => neighbour.pacmap({ neighbours: 5, iterations: 30 }), unlabelled),
  // neural/language-models
  kneserNey: f(() => languageModels.kneserNey(), corpus),
  charGpt: f(() => languageModels.charGpt({ width: 8, layers: 1, heads: 2, context: 8, steps: 3 }), corpus),
}

/** A stream for fits that draw. */
export const fitStream = () => stream(7)
