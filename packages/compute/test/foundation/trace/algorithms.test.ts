/**
 * The generated algorithm suite (design S §7): every algorithm registered in the compute (`kind: 'algorithm'`, found as
 * the catalog finds it) runs on a small problem from `CASES`, keyed by its address `module/key`, and must
 *
 * - pass the trace protocol checks of `protocol.ts` (same key, same trace; `seek` = `run`; `extend`; states survive a
 *   structured clone);
 * - draw from its streams exactly when it declares `random: true` (another key gives another trace, or the same one);
 * - have every state field its `info.state` roles name (iterate, objective, grad, stepSize), in the initial state and
 *   after stepping;
 * - set only the `Status` flags its `info.state.flags` declares.
 *
 * A registered algorithm without a case fails, so registering one means adding its case here.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import {
  tensor,
  toFlat,
  mul,
  sum,
  square,
  get,
  stack,
  sub,
  type Tensor,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import { run, trace, type Algorithm } from 'aifn-compute/foundation/trace'
import type { AlgorithmInfo, Status } from 'aifn-compute/foundation/contracts'
import {
  bdf,
  adaptiveBdf,
  dormandPrince,
  implicitEuler,
  implicitTrapezoid,
  rungeKutta,
  symplectic,
  type Rhs,
} from 'aifn-compute/dynamics/ode'
import { eulerMaruyama, geometricBrownianMotion, milstein, stochasticRungeKutta } from 'aifn-compute/dynamics/sde'
import { fromEdges } from 'aifn-compute/graph'
import { edmondsKarpSteps, minCostFlowSteps } from 'aifn-compute/graph/flows'
import { labelPropagationSteps, labelSpreadingSteps } from 'aifn-compute/graph/propagation'
import { aStarSteps, bellmanFordSteps, dijkstraSteps, floydWarshallSteps } from 'aifn-compute/graph/shortest-paths'
import { kruskalSteps, primSteps } from 'aifn-compute/graph/spanning-trees'
import {
  breadthFirstSteps,
  depthFirstSteps,
  iterativeDeepeningSteps,
  kahnSteps,
  kosarajuSteps,
  tarjanSteps,
} from 'aifn-compute/graph/traversal'
import {
  chainSumProduct,
  enumerationSteps,
  forwardBackwardSteps,
  variableEliminationSteps,
  viterbiSteps,
} from 'aifn-compute/inference/exact'
import {
  assumedDensityFiltering,
  expectationPropagation,
  modelExpectationPropagation,
  multivariateExpectationPropagation,
  probitTilted,
  type EpOptions,
} from 'aifn-compute/inference/expectation-propagation'
import { bocpd, kalmanFilterSteps, normalKnownVariance, rtsSmootherSteps } from 'aifn-compute/inference/filtering'
import { beliefPropagationSteps, gaussianBeliefPropagationSteps } from 'aifn-compute/inference/message-passing'
import { discreteFactor, discreteFactorGraph, dist, model } from 'aifn-compute/inference/model'
import {
  bivariateGaussianConditionals,
  factorGraphGibbs,
  gibbs,
  hmc,
  independenceMetropolis,
  mala,
  metropolisHastings,
  modelGibbs,
  nuts,
  particleFilter,
  randomWalkMetropolis,
  langevinParticles,
  sgld,
  sliceSampler,
  temperedSmc,
  unadjustedLangevin,
} from 'aifn-compute/inference/stochastic'
import { bbvi } from 'aifn-compute/inference/variational'
import { Normal } from 'aifn-compute/probability/distributions'
import {
  confidenceSequence,
  cusum,
  groupSequentialBoundaries,
  groupSequentialTest,
  msprt,
  sprt,
} from 'aifn-compute/probability/tests'
import { normal, normals, uniform } from 'aifn-compute/foundation/random'
import { binaryCrossEntropyWithLogits, discriminatorLoss, generatorLoss } from 'aifn-compute/learning/losses'
import { poolAdjacentViolatorsSteps } from 'aifn-compute/learning/calibration'
import { markovChainSteps } from 'aifn-compute/probability/markov'
import {
  fixedShare,
  followTheRegularisedLeader,
  hedge,
  onlineAdagrad,
  onlineGradientDescent,
  onlineNewtonStep,
  weightedMajority,
  type OnlineLoss,
} from 'aifn-compute/optim/online'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import {
  adversarialTraining,
  contrastiveDivergence,
  fullBatchTraining,
  methodTraining,
  privateTraining,
  trainingLoop,
} from 'aifn-compute/nn/training'
import { flashAttentionSteps } from 'aifn-compute/nn/attention'
import { beamSearch, greedyDecoding, samplingDecoding, speculativeDecoding } from 'aifn-compute/nn/decoding'
import { add as addT, blellochScanSteps, hillisSteeleScanSteps } from 'aifn-compute/foundation/tensor'
import {
  gaussSeidelSteps,
  gramSchmidtSteps,
  householderSteps,
  jacobiSteps,
  kleinmanIteration,
  powerIterationSteps,
  riccatiDoubling,
  riccatiMatrixSign,
  riccatiRecursion,
} from 'aifn-compute/numerics/linalg'
import { nmfSteps } from 'aifn-compute/numerics/factorisation'
import { adaptiveSimpson, gaussKronrod, monteCarlo, romberg } from 'aifn-compute/numerics/quadrature'
import {
  bisection,
  brent,
  broyden,
  continuation,
  fixedPoint,
  newtonHomotopy,
  newtonRoot,
  newtonSystem,
  regulaFalsi,
  secant,
  type SystemWithJacobian,
} from 'aifn-compute/numerics/roots'
import { cmaEs, nelderMead, simulatedAnnealing } from 'aifn-compute/optim/derivative-free'
import { refinementSearchSteps } from 'aifn-compute/optim/search'
import { selectorLanguage, subgroupDiscoverySteps, wraccQuality } from 'aifn-compute/learning/subgroups'
import {
  adagrad,
  adam,
  adamw,
  conjugateGradient,
  coordinateDescent,
  gradientDescent,
  linearConjugateGradient,
  momentum,
  nesterov,
  rmsprop,
} from 'aifn-compute/optim/first-order'
import {
  activeSet,
  boxQuadraticProgram,
  branchAndBound,
  dynamicProgram,
  gomory,
  hungarianSteps,
  linearInteriorPoint,
  quadraticInteriorPoint,
  simplex,
} from 'aifn-compute/optim/programming'
import {
  alternatingProjectionsSteps,
  fista,
  ista,
  projectBox,
  projectedGradient,
  proximalGradient,
  proxL1,
} from 'aifn-compute/optim/proximal'
import {
  bfgs,
  gaussNewton,
  lbfgs,
  levenbergMarquardt,
  newton,
  owlqn,
  trustRegion,
} from 'aifn-compute/optim/second-order'
import { siftSteps, vmdSteps } from 'aifn-compute/signal/decompositions'
import { lms, nlms, rls } from 'aifn-compute/signal/statistical'
import { scrimpSteps } from 'aifn-compute/signal/similarity'
import {
  basisPursuitDenoisingSteps,
  convolutionalDictionaryLearningSteps,
  convolutionalSparseCodeSteps,
  dictionaryLearningSteps,
  iterativeHardThresholdingSteps,
  matchingPursuitSteps,
  orthogonalMatchingPursuitSteps,
} from 'aifn-compute/signal/sparse'
import { stateSpace, simulate, predictionErrorMethod } from 'aifn-compute/systems'
import { lqg, lqgSimulation, mpcController, recedingHorizon } from 'aifn-compute/dynamics/control'
import { ransac } from 'aifn-compute/numerics/robust'
import { costMatrix, gromovWassersteinSteps, sinkhornSteps, uniformWeights } from 'aifn-compute/transport'
import { fixture } from '../../fixtures'
import { randomGraph } from '../../graph/helpers'
import { casinoChain, isingGrid, randomTree, sprinkler } from '../../inference/graphs'
import { banana, gaussianTarget } from '../../inference/stochastic/targets'
import { bowl, rosenbrock } from '../../optim/problems'
import { bpeSteps, unigramLmSteps, wordPieceSteps } from 'aifn-compute/text/subword'
import { trainingSteps, whitespacePreTokeniser } from 'aifn-compute/text/pipeline'
import { hyphenationPatterns, liangSteps, parseHyphenated, patgenSteps } from 'aifn-compute/text/hyphenation'
import { foilProblem, foilSteps } from 'aifn-compute/logic/induction'
import { prologProgram, sldSteps } from 'aifn-compute/logic/resolution'
import { checkProtocol, plainOf } from '../../protocol'
import { address, entriesOf } from '../../registries'

/** A case: the algorithm, its start, and how many steps the checks run. */
type Case = { alg: Algorithm<unknown, Status>; start: unknown; steps?: number }
const at = <S, St extends Status>(alg: Algorithm<S, St>, start: S, steps?: number): Case => ({
  alg: alg as unknown as Algorithm<unknown, Status>,
  start,
  steps,
})

const sparseDictionary = [
  [1, 0, 0, 0.5],
  [0, 1, 0, 0.5],
  [0, 0, 1, 0.7],
]

const diagonallyDominant = [
  [4, 1, 0],
  [1, 5, 2],
  [0, 2, 6],
]

// ── Problems ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const lotkaVolterra: Rhs = (_t, x) => {
  const [a, b] = [get(x, 0), get(x, 1)]
  return stack([sub(mul(1.5, a), mul(a, b)), sub(mul(a, b), mul(3, b))])
}
const lv = { x0: [10, 5] }
const oscillator = { potential: (q: Tensor) => mul(0.5, sum(square(q))) }
const gbm = geometricBrownianMotion({ mu: 0.1, sigma: 0.3 }).sde
const paths = { x0: 1, paths: 8 }
const rosen = rosenbrock()
const quad = bowl({ condition: 5, angle: 0.5 })
const fromRosen = { x0: rosen.start }
const cubic = (x: number) => x * x * x - 2 * x - 5
const F: SystemWithJacobian = (v) => {
  const [x, y] = toFlat(v)
  return {
    value: [x * x + y * y - 4, x * y - 1],
    jacobian: [
      [2 * x, 2 * y],
      [y, x],
    ],
  }
}
const lasso = (x: Vector) => {
  const c = [3, -0.5, 1.2]
  const d = toFlat(x).map((vi, i) => vi - c[i])
  return { value: 0.5 * d.reduce((s, di) => s + di * di, 0), grad: d }
}
const ts = [0, 0.5, 1, 1.5, 2, 3]
const residuals = (x: Vector) => {
  const [a, b] = toFlat(x)
  return {
    residuals: ts.map((t) => a * Math.exp(b * t) - 2 * Math.exp(-0.5 * t)),
    jacobian: ts.map((t) => [Math.exp(b * t), a * t * Math.exp(b * t)]),
  }
}
const wyndor = {
  c: [-3, -5],
  A_ub: [
    [1, 0],
    [0, 2],
    [3, 2],
  ],
  b_ub: [4, 12, 18],
}
const qp = {
  Q: [
    [2, 0],
    [0, 2],
  ],
  c: [-2, -5],
  A: [
    [-1, 2],
    [1, 2],
    [1, -2],
    [-1, 0],
    [0, -1],
  ],
  b: [2, 6, 2, 0, 0],
}
const ip = {
  c: [-1, -1],
  A_ub: [
    [2, -2],
    [-8, 10],
  ],
  b_ub: [-1, 13],
}
const edit = (a: string, b: string) => ({
  shape: [a.length + 1, b.length + 1] as [number, number],
  cell: (i: number, j: number, g: (i: number, j: number) => number) =>
    i === 0
      ? j
      : j === 0
        ? i
        : Math.min(g(i - 1, j) + 1, g(i, j - 1) + 1, g(i - 1, j - 1) + (a[i - 1] === b[j - 1] ? 0 : 1)),
})
const graph = randomGraph(5, 8, 0.35)
const undirected = randomGraph(7, 8, 0.5, false)
const dag = fromEdges(6, [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 3],
  [3, 4],
  [5, 4],
])
const network = fromEdges(4, [
  [0, 1, 3],
  [0, 2, 2],
  [1, 2, 1],
  [1, 3, 2],
  [2, 3, 3],
])
const ising = isingGrid(2, 2, 0.4, 0.1)
const obs = [5, 5, 0, 5, 2, 5, 5]
const chain = casinoChain(obs)
const chainGraph = (() => {
  const cards = new Array<number>(obs.length).fill(2)
  const node = Array.from({ length: obs.length }, (_, n) => toFlat(chain.nodePotentials).slice(2 * n, 2 * n + 2))
  const transition = toFlat(chain.transition)
  return discreteFactorGraph(cards, [
    ...node.map((row, n) => discreteFactor([n], cards, row)),
    ...obs.slice(1).map((_, n) => discreteFactor([n, n + 1], cards, transition)),
  ])
})()
const cuts = [-1, 0.5, 0.2, 1.5, -0.3]
const signs = [1, 1, -1, -1, 1]
const probit: EpOptions = {
  prior: { mean: 0, variance: 2 },
  factors: cuts.length,
  tilted: (i, c) => probitTilted(c.mean, c.variance, signs[i], { offset: cuts[i] }),
}
const target = gaussianTarget(
  [0, 0],
  [
    [1, 0.5],
    [0.5, 1],
  ],
)
const chainStart = { x0: [0.2, 0.1] }
const ys = [0.3, 1.1, -0.2, 0.8, 0.5, 1.4]
const stateSpaceModel = {
  dim: 1,
  sampleInitial: (s: ReturnType<typeof stream>) => normal(s, 0, 1),
  sampleTransition: (v: Vector, _t: number, s: ReturnType<typeof stream>) => normal(s, 0.9 * toFlat(v)[0], 0.5),
  logObservation: (y: number, v: Vector) => -0.5 * ((y - toFlat(v)[0]) / 0.4) ** 2,
}
const mlp = Mlp([2, 4, 1], { activation: 'tanh', init: xavierUniform() })
const data = {
  x: tensor(Array.from({ length: 12 }, (_, i) => [Math.cos(i), Math.sin(i)])),
  y: tensor(Array.from({ length: 12 }, (_, i) => [i % 2])),
}
const systems = fixture<Record<'care' | 'dare', never>>('systems')
const sift = Array.from(
  { length: 128 },
  (_, i) => Math.sin((2 * Math.PI * i) / 16) + 0.8 * Math.sin((2 * Math.PI * i) / 128) + i / 128,
)
// System identification for the adaptive filters: d = (0.5, −0.3, 0.1) ∗ x.
const adaptiveInput = sift.map((v, i) => v * Math.cos(0.7 * i))
const adaptiveDesired = adaptiveInput.map(
  (v, i) => 0.5 * v - 0.3 * (adaptiveInput[i - 1] ?? 0) + 0.1 * (adaptiveInput[i - 2] ?? 0),
)
const points = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
  [2, 0.5],
  [0.5, 2],
]
const cx = costMatrix(points, points, { p: 1 })

// ── Cases, by address ────────────────────────────────────────────────────────────────────────────────────────────────

// A bigram language model over three tokens, for the decoders.
const toyLm = (prefix: readonly number[]): Tensor =>
  tensor(
    [
      [1.0, 0.2, -0.5],
      [0.1, 1.4, 0.8],
      [0.6, -0.3, 1.1],
    ][prefix.length ? prefix[prefix.length - 1] : 0],
  )

// An expert-loss matrix and a linear online loss, for the online learners.
const expertLosses = [
  [0.1, 0.9, 0.5],
  [0.8, 0.2, 0.5],
  [0.3, 0.6, 0.4],
  [0.9, 0.1, 0.5],
  [0.2, 0.7, 0.6],
  [0.4, 0.4, 0.4],
]
const onlineLinear: OnlineLoss = (t, w) => {
  const g = [Math.cos(t), Math.sin(2 * t)]
  const x = toFlat(w)
  return { value: g[0] * x[0] + g[1] * x[1], grad: g }
}

const CASES: Record<string, () => Case> = {
  'dynamics/ode/rungeKutta': () => at(rungeKutta(lotkaVolterra, 'rk4', { stepSize: 0.05 }), lv, 16),
  'dynamics/ode/dormandPrince': () => at(dormandPrince(lotkaVolterra, { tEnd: 50 }), lv, 16),
  'dynamics/ode/adaptiveBdf': () => at(adaptiveBdf(lotkaVolterra, { tEnd: 50 }), lv, 16),
  'dynamics/ode/implicitEuler': () => at(implicitEuler(lotkaVolterra, { stepSize: 0.05 }), lv, 12),
  'dynamics/ode/implicitTrapezoid': () => at(implicitTrapezoid(lotkaVolterra, { stepSize: 0.05 }), lv, 12),
  'dynamics/ode/bdf': () => at(bdf(lotkaVolterra, 3, { stepSize: 0.05 }), lv, 12),
  'dynamics/ode/symplectic': () => at(symplectic(oscillator, 'leapfrog', { stepSize: 0.1 }), { q0: [1], p0: [0] }),
  'dynamics/sde/eulerMaruyama': () => at(eulerMaruyama(gbm, { stepSize: 0.05 }), paths),
  'dynamics/sde/milstein': () => at(milstein(gbm, { stepSize: 0.05 }), paths),
  'dynamics/sde/stochasticRungeKutta': () => at(stochasticRungeKutta(gbm, { stepSize: 0.05 }), paths),
  'graph/flows/edmondsKarpSteps': () => at(edmondsKarpSteps(network, { source: 0, sink: 3 }), undefined, 6),
  'graph/propagation/labelPropagationSteps': () =>
    at(labelPropagationSteps(undirected, [0, -1, -1, -1, -1, -1, -1, 1]), undefined, 6),
  'graph/propagation/labelSpreadingSteps': () =>
    at(labelSpreadingSteps(undirected, [0, -1, -1, -1, -1, -1, -1, 1], { alpha: 0.5 }), undefined, 6),
  'signal/similarity/scrimpSteps': () => at(scrimpSteps(sift, 16, { diagonalsPerStep: 10 }), undefined, 6),
  'signal/sparse/matchingPursuitSteps': () => at(matchingPursuitSteps(sparseDictionary, [0, 2, -3]), undefined, 3),
  'signal/sparse/orthogonalMatchingPursuitSteps': () =>
    at(orthogonalMatchingPursuitSteps(sparseDictionary, [0, 2, -3], { sparsity: 2 }), undefined, 2),
  'signal/sparse/basisPursuitDenoisingSteps': () =>
    at(basisPursuitDenoisingSteps(sparseDictionary, [0, 2, -3], { lambda: 0.1 }), undefined, 6),
  'signal/sparse/iterativeHardThresholdingSteps': () =>
    at(iterativeHardThresholdingSteps(sparseDictionary, [0, 2, -3], { sparsity: 2 }), undefined, 6),
  'signal/sparse/convolutionalSparseCodeSteps': () =>
    at(
      convolutionalSparseCodeSteps([[1, 2, 1]], [0, 0, 1, 2, 1, 0, 0, 0, -1, -2, -1, 0], { lambda: 0.1 }),
      undefined,
      6,
    ),
  'signal/sparse/convolutionalDictionaryLearningSteps': () =>
    at(
      convolutionalDictionaryLearningSteps(
        [0.3, 1, -0.4, 2, 0.1, -1.5, 0.8, 0, 1.2, -0.7, 0.5, 2.2, -0.9, 0.4, 1.7, -0.2, 0.6, -1.1, 0.9, 0.05, -0.6],
        {
          filters: 2,
          length: 4,
          lambda: 0.1,
          codeSteps: 5,
          filterSteps: 5,
        },
      ),
      undefined,
      3,
    ),
  'signal/sparse/dictionaryLearningSteps': () =>
    at(
      dictionaryLearningSteps(
        [
          [1, 0, 0.5, -1, 2, 0],
          [0, 1, 0.5, 1, 0, -1],
          [0.3, 0, 1, 0, 1, 1],
        ],
        { atoms: 4, sparsity: 1 },
      ),
      undefined,
      4,
    ),
  'graph/flows/minCostFlowSteps': () =>
    at(
      minCostFlowSteps({
        nodes: 4,
        arcs: [
          { from: 0, to: 1, capacity: 3, cost: 1 },
          { from: 0, to: 2, capacity: 2, cost: 2 },
          { from: 1, to: 2, capacity: 1, cost: 1 },
          { from: 1, to: 3, capacity: 2, cost: 3 },
          { from: 2, to: 3, capacity: 3, cost: 1 },
        ],
        supply: [4, 0, 0, -4],
      }),
      undefined,
      6,
    ),
  'graph/shortest-paths/dijkstraSteps': () => at(dijkstraSteps(graph, { source: 0 }), undefined, 10),
  'graph/shortest-paths/aStarSteps': () => at(aStarSteps(graph, { source: 0, target: 7 }), undefined, 10),
  'graph/shortest-paths/bellmanFordSteps': () => at(bellmanFordSteps(graph, { source: 0 }), undefined, 10),
  'graph/shortest-paths/floydWarshallSteps': () => at(floydWarshallSteps(graph), undefined, 10),
  'graph/spanning-trees/kruskalSteps': () => at(kruskalSteps(undirected), undefined),
  'graph/spanning-trees/primSteps': () => at(primSteps(undirected), undefined),
  'graph/traversal/breadthFirstSteps': () => at(breadthFirstSteps(graph), undefined),
  'graph/traversal/depthFirstSteps': () => at(depthFirstSteps(graph), undefined),
  'graph/traversal/iterativeDeepeningSteps': () =>
    at(iterativeDeepeningSteps(undirected, { source: 0, target: 5 }), undefined),
  'graph/traversal/kahnSteps': () => at(kahnSteps(dag), undefined, 10),
  'graph/traversal/tarjanSteps': () => at(tarjanSteps(graph), undefined, 20),
  'graph/traversal/kosarajuSteps': () => at(kosarajuSteps(graph), undefined, 20),
  'inference/exact/enumerationSteps': () => at(enumerationSteps(ising, { chunk: 3 }), undefined, 6),
  'inference/exact/variableEliminationSteps': () => at(variableEliminationSteps(ising), undefined, 6),
  'inference/exact/forwardBackwardSteps': () => at(forwardBackwardSteps(chain), undefined, 20),
  'inference/exact/viterbiSteps': () => at(viterbiSteps(chain), undefined, 20),
  'inference/exact/chainSumProduct': () => at(chainSumProduct(chainGraph), undefined, 20),
  'inference/expectation-propagation/expectationPropagation': () => at(expectationPropagation(probit), undefined),
  'inference/expectation-propagation/assumedDensityFiltering': () => at(assumedDensityFiltering(probit), undefined, 5),
  'inference/expectation-propagation/multivariateExpectationPropagation': () =>
    at(
      multivariateExpectationPropagation({
        prior: {
          mean: tensor([0, 0]),
          covariance: tensor([
            [1, 0.5],
            [0.5, 1],
          ]),
        },
        tilted: (i, c) => probitTilted(c.mean, c.variance, signs[i]),
      }),
      undefined,
      6,
    ),
  'inference/expectation-propagation/modelExpectationPropagation': () =>
    at(
      modelExpectationPropagation(
        model('truncated chain', (m) => {
          const a = m.variable('a', dist.Normal(0, 1))
          const b = m.variable('b', dist.Normal(a, 0.5))
          m.observed('y', dist.Bernoulli(m.deterministic('inside', 'interval', [b, 0.5, Infinity])))
        }),
        { data: { y: 1 } },
      ),
      undefined,
      6,
    ),
  'inference/filtering/bocpd': () =>
    at(
      bocpd(normalKnownVariance({ mean: 0, priorSd: 2, sd: 1 }), [0.1, -0.4, 0.3, 3.2, 2.8, 3.1], { hazard: 0.2 }),
      undefined,
      6,
    ),
  'inference/filtering/kalmanFilterSteps': () =>
    at(kalmanFilterSteps({ A: 0.9, C: 1, Q: 0.1, R: 0.5, m0: 0, P0: 1 }, [0.3, 0.1, -0.2, 0.4]), undefined, 4),
  'inference/filtering/rtsSmootherSteps': () =>
    at(rtsSmootherSteps({ A: 0.9, C: 1, Q: 0.1, R: 0.5, m0: 0, P0: 1 }, [0.3, 0.1, -0.2, 0.4]), undefined, 4),
  'inference/message-passing/beliefPropagationSteps': () =>
    at(beliefPropagationSteps(randomTree(3), { mode: 'max' }), undefined),
  'inference/message-passing/gaussianBeliefPropagationSteps': () =>
    at(
      gaussianBeliefPropagationSteps(
        [
          [3, 1, 0],
          [1, 3, 1],
          [0, 1, 3],
        ],
        [1, 0, 1],
      ),
      undefined,
    ),
  'inference/stochastic/metropolisHastings': () =>
    at(
      metropolisHastings(banana(), (x, st) => ({
        proposal: toFlat(x).map((v, i) => v + 0.5 * (toFlat(normals(st, 2))[i] as number)),
        logProposalRatio: 0,
      })),
      chainStart,
      20,
    ),
  'inference/stochastic/randomWalkMetropolis': () => at(randomWalkMetropolis(banana()), chainStart, 20),
  'inference/stochastic/independenceMetropolis': () =>
    at(
      independenceMetropolis(target, { sample: (s) => [normal(s, 0, 2), normal(s, 0, 2)], logDensity: () => 0 }),
      chainStart,
      20,
    ),
  'inference/stochastic/gibbs': () => at(gibbs(bivariateGaussianConditionals(0.8), { scan: 'random' }), chainStart, 20),
  'inference/stochastic/sliceSampler': () => at(sliceSampler(banana()), chainStart, 20),
  'inference/stochastic/hmc': () => at(hmc(banana(), { stepSize: 0.1, steps: 5 }), chainStart, 20),
  'inference/stochastic/nuts': () => at(nuts(banana()), chainStart, 12),
  'inference/stochastic/unadjustedLangevin': () => at(unadjustedLangevin(banana()), chainStart, 20),
  'inference/stochastic/mala': () => at(mala(banana()), chainStart, 20),
  'inference/stochastic/sgld': () =>
    at(
      sgld(
        {
          dim: 1,
          size: ys.length,
          gradLogPrior: (th: Vector) => [-toFlat(th)[0] / 100],
          gradLogLikelihood: (th: Vector, i: number) => [ys[i] - toFlat(th)[0]],
        },
        { batchSize: 2, stepSize: 1e-2 },
      ),
      { x0: [0] },
    ),
  'inference/stochastic/particleFilter': () =>
    at(particleFilter(stateSpaceModel, { particles: 50 }), { observations: ys }, 6),
  'inference/stochastic/temperedSmc': () =>
    at(
      temperedSmc(
        {
          dim: 1,
          samplePrior: (st) => normal(st, 0, 1),
          logPrior: (v) => -0.5 * toFlat(v)[0] ** 2,
          logLikelihood: (v) => -0.5 * ((1.2 - toFlat(v)[0]) / 0.5) ** 2,
        },
        { particles: 100 },
      ),
      undefined,
      6,
    ),
  'inference/stochastic/factorGraphGibbs': () => at(factorGraphGibbs(ising), undefined, 10),
  'inference/stochastic/modelGibbs': () => at(modelGibbs(sprinkler, { data: { wet: 1 } }), undefined, 8),
  'inference/variational/bbvi': () => at(bbvi(target, { stepSize: 0.05 }), {}),
  'foundation/tensor/hillisSteeleScanSteps': () =>
    at(hillisSteeleScanSteps(addT, tensor([1, 2, 3, 4, 5])), undefined, 4),
  'foundation/tensor/blellochScanSteps': () => at(blellochScanSteps(addT, tensor([1, 2, 3, 4, 5]), 0), undefined, 8),
  'nn/attention/flashAttentionSteps': () =>
    at(
      flashAttentionSteps(
        tensor([
          [1, 0],
          [0, 1],
          [1, 1],
        ]),
        tensor([
          [1, 2],
          [0, 1],
          [2, 0],
          [1, 1],
        ]),
        tensor([[1], [2], [3], [4]]),
        { causal: true, queryBlock: 2, keyBlock: 2 },
      ),
      undefined,
      6,
    ),
  'nn/decoding/greedyDecoding': () => at(greedyDecoding(toyLm, { prompt: [0], maxTokens: 5 }), undefined, 6),
  'nn/decoding/samplingDecoding': () =>
    at(samplingDecoding(toyLm, { prompt: [0], maxTokens: 6, topP: 0.9, temperature: 1.2 }), undefined, 6),
  'nn/decoding/beamSearch': () => at(beamSearch(toyLm, { prompt: [1], maxTokens: 4, beams: 2 }), undefined, 4),
  'nn/decoding/speculativeDecoding': () =>
    at(
      speculativeDecoding(toyLm, (p) => mul(0.5, toyLm(p)) as Tensor, { prompt: [0], maxTokens: 8, lookahead: 2 }),
      undefined,
      6,
    ),
  'inference/stochastic/langevinParticles': () =>
    at(
      langevinParticles((x: Tensor) => mul(-1, x) as Tensor, { stepSize: 0.05 }),
      { x: normals(stream('particles'), [16, 2]) },
      10,
    ),
  'nn/training/adversarialTraining': () =>
    at(
      adversarialTraining<Tensor[], Tensor[]>({
        // A point-mass generator θ against a linear critic a·x + b on data drawn near 1.
        criticLoss: ([a, b], [theta], s) =>
          discriminatorLoss(addT(mul(a, normals(s, 2, 1, 0.1)), b), addT(mul(a, mul(theta, tensor([1, 1]))), b)),
        generatorLoss: ([theta], [a, b]) => generatorLoss(addT(mul(a, mul(theta, tensor([1, 1]))), b)),
        criticSteps: 2,
      }),
      { generator: [tensor([0])], critic: [tensor([0.1]), tensor([0])] },
      8,
    ),
  'nn/training/contrastiveDivergence': () =>
    at(
      contrastiveDivergence<Tensor[], { x: Tensor }>({
        energy: ([theta], x) => mul(0.5, sum(square(sub(x, theta)), -1)),
        data: { x: tensor(ys.map((v) => [v])) },
        batchSize: 4,
        sampler: { steps: 3, stepSize: 0.1, fresh: (s, n) => uniform(s, -2, 2, { shape: [n, 1] }) as Tensor },
        bufferSize: 10,
      }),
      { params: [tensor([0])] },
      8,
    ),
  'nn/training/fullBatchTraining': () =>
    at(
      fullBatchTraining({
        loss: (p: ReturnType<typeof mlp.init>, b: typeof data) => binaryCrossEntropyWithLogits(mlp.apply(p, b.x), b.y),
        data,
        options: { memory: 5 },
      }),
      { params: mlp.init(stream('protocol')) },
      8,
    ),
  'nn/training/methodTraining': () =>
    at(
      methodTraining(
        (p: ReturnType<typeof mlp.init>, b: typeof data) => binaryCrossEntropyWithLogits(mlp.apply(p, b.x), b.y),
        data,
        { method: 'adam', stepSize: 0.05, batchSize: 4 },
      ),
      { params: mlp.init(stream('protocol')) },
      8,
    ),
  'nn/training/privateTraining': () =>
    at(
      privateTraining({
        loss: (p: Tensor[], e: typeof data) => sum(square(sub(sum(mul(p[0], e.x)), e.y))),
        data,
        batchSize: 4,
        clipNorm: 1,
        noiseMultiplier: 1,
      }),
      { params: [tensor([0.1, -0.2])] },
      6,
    ),
  'nn/training/trainingLoop': () =>
    at(
      trainingLoop({
        loss: (p: ReturnType<typeof mlp.init>, b: typeof data) => binaryCrossEntropyWithLogits(mlp.apply(p, b.x), b.y),
        data,
        batchSize: 4,
      }),
      { params: mlp.init(stream('protocol')) },
      8,
    ),
  'numerics/factorisation/nmfSteps': () =>
    at(
      nmfSteps(
        [
          [1, 2, 0.5],
          [0.2, 1, 3],
          [2, 0.1, 1],
          [1, 1, 1],
        ],
        { rank: 2, tolerance: 0 },
      ),
      undefined,
      6,
    ),
  'numerics/linalg/kleinmanIteration': () => at(kleinmanIteration(systems.care), undefined, 6),
  'numerics/linalg/riccatiMatrixSign': () => at(riccatiMatrixSign(systems.care), undefined, 6),
  'numerics/linalg/riccatiRecursion': () => at(riccatiRecursion(systems.dare), undefined, 6),
  'numerics/linalg/riccatiDoubling': () => at(riccatiDoubling(systems.dare), undefined, 5),
  'numerics/linalg/gramSchmidtSteps': () =>
    at(
      gramSchmidtSteps(
        [
          [1, 1, 0],
          [1, 0, 1],
          [0, 1, 1],
          [1, 1, 1],
        ],
        { variant: 'classical' },
      ),
      undefined,
      3,
    ),
  'numerics/linalg/householderSteps': () =>
    at(
      householderSteps([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41],
      ]),
      undefined,
      3,
    ),
  'numerics/linalg/jacobiSteps': () => at(jacobiSteps(diagonallyDominant, [1, 2, 3]), undefined, 8),
  'numerics/linalg/gaussSeidelSteps': () =>
    at(gaussSeidelSteps(diagonallyDominant, [1, 2, 3], { omega: 1.1 }), undefined, 8),
  'learning/calibration/poolAdjacentViolatorsSteps': () =>
    at(poolAdjacentViolatorsSteps([3, 1, 4, 1, 5, 9, 2, 6]), undefined, 20),
  'numerics/linalg/powerIterationSteps': () => at(powerIterationSteps(diagonallyDominant), undefined, 10),
  'numerics/quadrature/adaptiveSimpson': () =>
    at(
      adaptiveSimpson((x) => Math.abs(x - 0.3), { tolerance: 1e-12 }),
      { a: 0, b: 1 },
      20,
    ),
  'numerics/quadrature/gaussKronrod': () =>
    at(
      gaussKronrod((x) => 1 / Math.sqrt(x), { atol: 0, rtol: 0 }),
      { a: 0, b: 1 },
      10,
    ),
  'numerics/quadrature/romberg': () =>
    at(
      romberg((x) => Math.exp(x)),
      { a: 0, b: 1 },
      6,
    ),
  'numerics/quadrature/monteCarlo': () =>
    at(
      monteCarlo((x: { data: ArrayLike<number> }) => Math.exp(x.data[0] + x.data[1]), {
        lo: [0, 0],
        hi: [1, 1],
        batch: 50,
      }),
      undefined,
      8,
    ),
  'numerics/roots/bisection': () => at(bisection(cubic), { lo: 0, hi: 5 }),
  'numerics/roots/regulaFalsi': () => at(regulaFalsi(cubic), { lo: 0, hi: 5 }),
  'numerics/roots/brent': () => at(brent(cubic), { lo: 0, hi: 5 }),
  'numerics/roots/secant': () => at(secant(cubic), { x0: 2, x1: 3 }, 6),
  'numerics/roots/newtonRoot': () =>
    at(
      newtonRoot((x: number) => ({ value: cubic(x), derivative: 3 * x * x - 2 }), { damped: true }),
      { x0: 0.5 },
      6,
    ),
  'numerics/roots/newtonSystem': () => at(newtonSystem(F), { x0: [2, 0.3] }, 5),
  'numerics/roots/broyden': () =>
    at(
      broyden((v) => F(v).value),
      { x0: [2, 0.3] },
      6,
    ),
  'numerics/roots/fixedPoint': () =>
    at(
      fixedPoint((x) => toFlat(x).map(Math.cos)),
      { x0: [1] },
      10,
    ),
  'numerics/roots/continuation': () => at(continuation(newtonHomotopy(F, [3, 0.5])), { x0: [3, 0.5] }, 10),
  'probability/tests/sprt': () =>
    at(
      sprt([0.3, 1.2, 0.8, -0.1, 0.9, 1.4], { h0: Normal(0, 1), h1: Normal(1, 1), alpha: 0.1, beta: 0.1 }),
      undefined,
      6,
    ),
  'probability/tests/msprt': () => at(msprt([0.3, 1.2, 0.8, -0.1, 0.9, 1.4], { sigma: 1, tau: 0.5 }), undefined, 6),
  'probability/tests/confidenceSequence': () =>
    at(confidenceSequence([0.3, 1.2, 0.8, -0.1, 0.9, 1.4], { sigma: 1, tau: 0.5 }), undefined, 6),
  'probability/tests/groupSequentialTest': () =>
    at(
      groupSequentialTest([0.3, 1.2, 0.8, -0.1, 0.9, 1.4], {
        looks: [2, 4, 6],
        boundaries: groupSequentialBoundaries(3, { points: 101 }),
        sigma: 1,
      }),
      undefined,
      6,
    ),
  'probability/tests/cusum': () => at(cusum([0.3, 1.2, 2.8, 2.1, 0.9, 3.4], { h: 2 }), undefined, 6),
  'probability/markov/markovChainSteps': () =>
    at(
      markovChainSteps(
        [
          [0.7, 0.2, 0.1],
          [0.3, 0.4, 0.3],
          [0.2, 0.3, 0.5],
        ],
        { start: [0.2, 0.3, 0.5] },
      ),
      undefined,
      8,
    ),
  'optim/online/hedge': () => at(hedge(expertLosses, { eta: 0.5 }), undefined, 6),
  'optim/search/refinementSearchSteps': () =>
    at(
      refinementSearchSteps<readonly number[]>(
        {
          root: [],
          refine: (n) => [0, 1, 2, 3, 4].filter((i) => i > (n.at(-1) ?? -1)).map((i) => [...n, i]),
          quality: (n) => n.reduce((a, i) => a + [2, -1, 1.5, -2, 1][i] - 0.3, 0),
          bound: (n) => n.reduce((a, i) => a + [2, -1, 1.5, -2, 1][i] - 0.3, 0) + 4,
        },
        { strategy: 'best-first', maxDepth: 3, k: 3 },
      ),
      undefined,
      20,
    ),
  'learning/subgroups/subgroupDiscoverySteps': () =>
    at(
      subgroupDiscoverySteps(
        selectorLanguage({ a: ['x', 'y', 'x', 'y', 'x', 'z'], b: [1, 2, 3, 4, 5, 6] }, { bins: 3 }),
        wraccQuality([1, 0, 1, 0, 0, 1]),
        { beamWidth: 2, maxDepth: 2 },
      ),
      undefined,
      4,
    ),
  'optim/online/fixedShare': () => at(fixedShare(expertLosses, { eta: 0.5, alpha: 0.1 }), undefined, 6),
  'optim/online/weightedMajority': () =>
    at(
      weightedMajority(
        expertLosses.map((r) => r.map((v) => (v > 0.5 ? 1 : 0))),
        [1, 0, 1, 1, 0, 1],
      ),
      undefined,
      6,
    ),
  'optim/online/onlineGradientDescent': () =>
    at(
      onlineGradientDescent(onlineLinear, { dim: 2, domain: { kind: 'ball', radius: 1 }, comparator: [0.5, 0] }),
      undefined,
      6,
    ),
  'optim/online/followTheRegularisedLeader': () =>
    at(followTheRegularisedLeader(onlineLinear, { dim: 2, eta: 0.3, l1: 0.1 }), undefined, 6),
  'optim/online/onlineNewtonStep': () =>
    at(onlineNewtonStep(onlineLinear, { dim: 2, domain: { kind: 'ball', radius: 1 } }), undefined, 6),
  'optim/online/onlineAdagrad': () =>
    at(onlineAdagrad(onlineLinear, { dim: 2, domain: { kind: 'box', lower: -1, upper: 1 } }), undefined, 6),
  'optim/derivative-free/nelderMead': () => at(nelderMead(rosen.value), fromRosen),
  'optim/derivative-free/simulatedAnnealing': () => at(simulatedAnnealing(rosen.value), fromRosen),
  'optim/derivative-free/cmaEs': () => at(cmaEs(rosen.value), fromRosen),
  'optim/first-order/gradientDescent': () => at(gradientDescent(rosen.objective, { stepSize: 1e-3 }), fromRosen),
  'optim/first-order/momentum': () => at(momentum(rosen.objective, { stepSize: 1e-3 }), fromRosen),
  'optim/first-order/nesterov': () => at(nesterov(rosen.objective, { stepSize: 1e-3 }), fromRosen),
  'optim/first-order/adagrad': () => at(adagrad(rosen.objective), fromRosen),
  'optim/first-order/rmsprop': () => at(rmsprop(rosen.objective), fromRosen),
  'optim/first-order/adam': () => at(adam(rosen.objective, { stepSize: 0.05 }), fromRosen),
  'optim/first-order/adamw': () => at(adamw(rosen.objective, { stepSize: 0.05 }), fromRosen),
  'optim/first-order/conjugateGradient': () => at(conjugateGradient(rosen.objective), fromRosen),
  'optim/first-order/linearConjugateGradient': () =>
    at(
      linearConjugateGradient(
        [
          [4, 1, 0],
          [1, 3, 1],
          [0, 1, 2],
        ],
        [1, 2, 3],
      ),
      {},
      3,
    ),
  'optim/first-order/coordinateDescent': () => at(coordinateDescent(quad.objective), { x0: quad.start }),
  'optim/programming/simplex': () => at(simplex(wyndor), {}, 6),
  'optim/programming/linearInteriorPoint': () => at(linearInteriorPoint(wyndor), {}, 8),
  'optim/programming/activeSet': () => at(activeSet(qp), { x0: [2, 0] }, 6),
  'optim/programming/quadraticInteriorPoint': () => at(quadraticInteriorPoint(qp), {}, 8),
  'optim/programming/boxQuadraticProgram': () =>
    at(
      boxQuadraticProgram({
        Q: [
          [2, 0.5],
          [0.5, 1],
        ],
        c: [-4, 1],
        lower: [0, 0],
        upper: [1, 1],
      }),
      {},
      5,
    ),
  'optim/programming/branchAndBound': () => at(branchAndBound(ip), {}, 6),
  'optim/programming/gomory': () => at(gomory(ip), {}, 6),
  'optim/programming/hungarianSteps': () =>
    at(
      hungarianSteps([
        [4, 1, 3],
        [2, 0, 5],
        [3, 2, 2],
      ]),
      {},
      8,
    ),
  'optim/programming/dynamicProgram': () => at(dynamicProgram(edit('kitten', 'sitting')), {}, 5),
  'optim/proximal/proximalGradient': () =>
    at(proximalGradient(lasso, proxL1(1), { stepSize: 0.5 }), { x0: [0, 0, 0] }, 8),
  'optim/proximal/ista': () => at(ista(lasso, proxL1(1), { stepSize: 0.5 }), { x0: [0, 0, 0] }, 8),
  'optim/proximal/fista': () =>
    at(fista(lasso, proxL1(1), { stepSize: 0.5, backtracking: true }), { x0: [0, 0, 0] }, 8),
  'optim/proximal/projectedGradient': () =>
    at(projectedGradient(lasso, projectBox(0, 1), { stepSize: 0.5 }), { x0: [0.5, 0.5, 0.5] }, 8),
  'optim/proximal/alternatingProjectionsSteps': () =>
    at(alternatingProjectionsSteps([projectBox(0, 1)]), { x0: [2, -1] }, 4),
  'optim/second-order/newton': () => at(newton(rosen.objective, { hessian: rosen.hessian }), fromRosen),
  'optim/second-order/trustRegion': () => at(trustRegion(rosen.objective, { hessian: rosen.hessian }), fromRosen),
  'optim/second-order/bfgs': () => at(bfgs(rosen.objective), fromRosen),
  'optim/second-order/lbfgs': () => at(lbfgs(rosen.objective, { memory: 3 }), fromRosen),
  'optim/second-order/owlqn': () => at(owlqn(rosen.objective, { memory: 3, l1: 0.1 }), fromRosen),
  'optim/second-order/gaussNewton': () => at(gaussNewton(residuals), { x0: [1, 0] }, 6),
  'optim/second-order/levenbergMarquardt': () => at(levenbergMarquardt(residuals), { x0: [1, 0] }, 6),
  'signal/decompositions/siftSteps': () => at(siftSteps(sift), undefined, 4),
  'signal/decompositions/vmdSteps': () => at(vmdSteps(sift, { modes: 2 }), undefined, 4),
  'signal/statistical/lms': () => at(lms(adaptiveInput, adaptiveDesired, { order: 3, stepSize: 0.1 }), undefined),
  'signal/statistical/nlms': () => at(nlms(adaptiveInput, adaptiveDesired, { order: 3, stepSize: 0.5 }), undefined),
  'signal/statistical/rls': () => at(rls(adaptiveInput, adaptiveDesired, { order: 3 }), undefined),
  'systems/predictionErrorMethod': () => {
    const u = Array.from({ length: 120 }, (_, t) => Math.sin(0.7 * t) + Math.cos(1.9 * t))
    const y = u.map((_, t) => (t > 0 ? 0.8 * u[t - 1] : 0) + 0.05 * Math.sin(3.1 * t))
    return at(predictionErrorMethod(y, u, { na: 1, nb: 1, nc: 1 }), {}, 4)
  },
  'dynamics/control/recedingHorizon': () => {
    const plant = {
      A: [
        [1, 0.1],
        [0, 1],
      ],
      B: [[0.005], [0.1]],
    }
    const c = mpcController({
      ...plant,
      Q: [
        [1, 0],
        [0, 0.1],
      ],
      R: [[0.1]],
      horizon: 5,
      uMin: -1,
      uMax: 1,
    })
    return at(recedingHorizon(c), { x0: [2, 0] }, 5)
  },
  'dynamics/control/lqgSimulation': () => {
    const plant = {
      A: [
        [0.9, 0.1],
        [0, 0.95],
      ],
      B: [[0], [0.1]],
      C: [[1, 0]],
    }
    const w = {
      Q: [
        [1, 0],
        [0, 1],
      ],
      R: [[1]],
      W: [
        [0.01, 0],
        [0, 0.01],
      ],
      V: [[0.1]],
    }
    return at(lqgSimulation(plant, w, lqg(plant, w, { discrete: true })), { x0: [1, 0] }, 5)
  },
  'numerics/robust/ransac': () => {
    const xs = Array.from({ length: 30 }, (_, i) => i / 3)
    const ys = xs.map((x, i) => (i % 5 === 0 ? 20 - x : 2 * x + 1))
    return at(
      ransac(
        {
          count: 30,
          sampleSize: 2,
          fit: ([i, j]) => (xs[i] === xs[j] ? null : (ys[j] - ys[i]) / (xs[j] - xs[i])),
          residuals: (a) => xs.map((x, k) => Math.abs(ys[k] - ys[0] - a * (x - xs[0]))),
        },
        { threshold: 0.1 },
      ),
      undefined,
      5,
    )
  },
  'systems/simulate': () =>
    at(
      simulate(
        stateSpace({
          A: [
            [0, 1],
            [0, 0],
          ],
          B: [0, 1],
          C: [1, 0],
        }),
        1,
        { dt: 0.1, tEnd: 0.5 },
      ),
      {},
      10,
    ),
  'transport/sinkhornSteps': () =>
    at(sinkhornSteps(uniformWeights(6), uniformWeights(6), cx, { epsilon: 0.5, tolerance: 0 }), {}),
  'transport/gromovWassersteinSteps': () =>
    at(
      gromovWassersteinSteps({ cx, cy: cx, a: uniformWeights(6), b: uniformWeights(6) }, { epsilon: 0.05 }),
      undefined,
      6,
    ),
  'text/hyphenation/liangSteps': () =>
    at(
      liangSteps(hyphenationPatterns(['hy3ph', 'he2n', 'hena4', 'hen5at', '1na', 'n2at']), 'hyphenation'),
      undefined,
      13,
    ),
  'text/hyphenation/patgenSteps': () =>
    at(
      patgenSteps(
        ['hy-phen-a-tion', 'ta-ble', 'peo-ple', 'win-ter', 'bet-ter'].map((w) => parseHyphenated(w)),
        {
          leftMin: 1,
          rightMin: 1,
        },
      ),
      undefined,
      6,
    ),
  'logic/resolution/sldSteps': () =>
    at(
      sldSteps(prologProgram('t(a). t(b). t(c). f(X) :- t(X), !. g(X) :- \\+ t(X).'), 'f(X), g(d), member(Y, [1, 2])'),
      undefined,
      30,
    ),
  'logic/induction/foilSteps': () =>
    at(
      foilSteps(
        foilProblem(
          'parent(ann, mary). parent(ann, tom). parent(tom, eve). parent(tom, ian). female(ann). female(mary). female(eve).',
          'daughter(mary, ann). daughter(eve, tom).',
        ),
      ),
      undefined,
      4,
    ),
  'text/subword/bpeSteps': () => at(bpeSteps({ low: 5, lower: 2, newest: 6, widest: 3 }), undefined, 12),
  'text/subword/wordPieceSteps': () => at(wordPieceSteps({ hug: 10, pug: 5, pun: 12, bun: 4, hugs: 5 }), undefined, 8),
  'text/subword/unigramLmSteps': () =>
    at(unigramLmSteps({ hug: 10, pug: 5, pun: 12, bun: 4, hugs: 5 }, { vocabularySize: 12 }), undefined, 4),
  'text/pipeline/trainingSteps': () =>
    at(
      trainingSteps({ preTokeniser: whitespacePreTokeniser() }, ['low lower newest widest', 'newest low'], {
        type: 'bpe',
        vocabularySize: 20,
        minCount: 1,
      }),
      undefined,
      6,
    ),
}

const FLAGS = ['converged', 'diverged', 'stalled', 'terminated'] as const

const algorithms = await entriesOf<AlgorithmInfo>('algorithm')

describe('the algorithm registry', () => {
  it('has algorithms, each with a case', () => {
    expect(algorithms.length).toBeGreaterThan(90)
    expect(algorithms.map(address).filter((a) => !(a in CASES))).toEqual([])
  })
  it('has no case for an unregistered algorithm', () => {
    const registered = new Set(algorithms.map(address))
    expect(Object.keys(CASES).filter((a) => !registered.has(a))).toEqual([])
  })
})

describe.each(algorithms.map((a) => [address(a), a] as const))('%s', (key, entry) => {
  const info = entry.info
  const make = CASES[key]
  it.runIf(make !== undefined)('satisfies the trace protocol', () => {
    const { alg, start, steps = 12 } = make()
    checkProtocol(alg, start, { steps, random: info.random === true })
    if (info.random !== true) {
      // A deterministic algorithm gives the same trace under any key.
      const a = trace(alg, start, steps, { stream: stream('a') })
      const b = trace(alg, start, steps, { stream: stream('b') })
      expect(b.steps.map(plainOf)).toEqual(a.steps.map(plainOf))
    }
  })
  it.runIf(make !== undefined)('has the fields its state roles name and sets only its declared flags', () => {
    const { alg, start, steps = 12 } = make()
    const states = trace(alg, start, steps, { stream: stream(1) }).steps
    const roles = Object.entries(info.state).filter(([role]) => role !== 'flags') as [string, string][]
    for (const s of [states[0], states.at(-1)!]) {
      for (const [role, field] of roles) expect(field in s, `${role} → ${field}`).toBe(true)
    }
    const set = new Set<string>()
    for (const s of states)
      for (const flag of FLAGS) if ((s as unknown as Record<string, unknown>)[flag] !== undefined) set.add(flag)
    expect([...set].filter((f) => !info.state.flags.includes(f as (typeof FLAGS)[number]))).toEqual([])
    expect(run(alg, start, 0).t).toBe(0)
  })
})
