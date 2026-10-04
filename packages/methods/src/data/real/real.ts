/**
 * Small real datasets, embedded: Iris, Old Faithful, Anscombe's quartet, the coal-mining disasters and Zachary's karate
 * club.
 */

import { dense, fromData } from 'aifn-compute/foundation/tensor'
import { fromEdges, type Graph } from 'aifn-compute/graph'
import { ANSCOMBE_DATA, COAL_MINING_DATA, FAITHFUL_DATA, IRIS_DATA, KARATE_CLUBS, KARATE_EDGES } from './embedded'
import { labels, matrix, vector, type Dataset } from '../types'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { space } from 'aifn-compute/foundation/space'

/**
 * Fisher's Iris data: 150 flowers (50 each of setosa, versicolor and virginica) with four measurements in cm. x is
 * 150 × 4, y the species (0, 1, 2).
 */
export function iris(): Dataset {
  return {
    kind: 'dataset',
    x: matrix(Float64Array.from(IRIS_DATA), 150, 4),
    y: labels(Array.from({ length: 150 }, (_, i) => Math.floor(i / 50))),
    meta: {
      name: 'Iris',
      description: 'Fisher’s Iris data: sepal and petal length and width (cm) of 150 flowers of three species.',
      task: 'classification',
      featureNames: ['sepal length', 'sepal width', 'petal length', 'petal width'],
      labelNames: ['setosa', 'versicolor', 'virginica'],
      source:
        'Fisher (1936), "The use of multiple measurements in taxonomic problems", Annals of Eugenics 7(2); UCI copy',
      url: 'https://archive.ics.uci.edu/dataset/53/iris',
    },
  }
}

/**
 * Old Faithful geyser data: 272 eruptions with duration and waiting time to the next eruption (minutes). x is 272 × 2
 * (eruption, waiting); there are no labels, but the data form two clear clusters.
 */
export function oldFaithful(): Dataset {
  return {
    kind: 'dataset',
    x: matrix(Float64Array.from(FAITHFUL_DATA), 272, 2),
    meta: {
      name: 'Old Faithful',
      description:
        'Eruption duration and waiting time to the next eruption (minutes) for 272 eruptions of Old Faithful.',
      task: 'clustering',
      featureNames: ['eruption duration', 'waiting time'],
      source: 'Härdle (1991), Smoothing Techniques with Implementation in S; R datasets::faithful',
      url: 'https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html',
    },
  }
}

/**
 * Anscombe's quartet: four sets of eleven (x, y) points with the same means, variances, correlation (0.816) and
 * least-squares line (y = 3 + 0.5x), yet very different shapes. Returns the four sets, each with x 11 × 1 and y.
 */
export function anscombe(): Dataset[] {
  const numerals = ['I', 'II', 'III', 'IV']
  return numerals.map((numeral, k) => ({
    kind: 'dataset',
    x: matrix(Float64Array.from(ANSCOMBE_DATA.x[k]), 11, 1),
    y: vector(ANSCOMBE_DATA.y[k]),
    meta: {
      name: `Anscombe ${numeral}`,
      description: `Set ${numeral} of Anscombe's quartet: eleven points sharing the summary statistics of the other three sets.`,
      task: 'regression',
      featureNames: ['x'],
      targetName: 'y',
      source: 'Anscombe (1973), "Graphs in statistical analysis", The American Statistician 27(1)',
    },
  }))
}

/**
 * British coal-mining disasters per year, 1851–1962 (Jarrett, 1979): x is 112 × 1 counts, t the calendar years. The
 * classic Poisson changepoint series: the rate drops from about 3 to about 1 per year around 1890, and Adams and MacKay
 * (2007) ran BOCPD on it with a Poisson–gamma segment model. No truth: the changepoint is what is inferred.
 */
export function coalMining(): Dataset {
  const n = COAL_MINING_DATA.length
  return {
    kind: 'dataset',
    x: matrix(Float64Array.from(COAL_MINING_DATA), n, 1),
    t: fromData(Float64Array.from({ length: n }, (_, i) => 1851 + i)),
    meta: {
      name: 'coal-mining disasters',
      description: 'Explosions in British coal mines that killed ten or more men, counted per year from 1851 to 1962.',
      task: 'sequence',
      featureNames: ['disasters'],
      source:
        'Jarrett (1979), "A note on the intervals between coal-mining disasters", Biometrika 66(1); yearly counts as in Carlin, Gelfand and Smith (1992), Applied Statistics 41(2)',
      url: 'https://doi.org/10.1093/biomet/66.1.191',
    },
  }
}

/** A dataset whose rows are the nodes of a graph. */
export interface GraphDataset extends Dataset {
  readonly graph: Graph
}

/**
 * Zachary's karate club: 34 members joined by 78 friendships, labelled by the club each joined when the club split (0:
 * Mr. Hi, 1: the Officer). x is the 34 × 34 identity (one feature per member, as in Kipf and Welling's experiment), y
 * the clubs, and `graph` the undirected friendship graph.
 */
export function karateClub(): GraphDataset {
  const n = KARATE_CLUBS.length
  const pairs = Array.from(
    { length: KARATE_EDGES.length / 2 },
    (_, k) => [KARATE_EDGES[2 * k], KARATE_EDGES[2 * k + 1]] as const,
  )
  return {
    kind: 'dataset',
    x: matrix(dense.identity(n), n, n),
    y: labels(KARATE_CLUBS),
    graph: fromEdges(n, pairs, { directed: false }),
    meta: {
      name: "Zachary's karate club",
      description:
        'Friendships among 34 members of a university karate club, and the club each joined when it split in two.',
      task: 'classification',
      featureNames: Array.from({ length: n }, (_, i) => `member ${i}`),
      labelNames: ['Mr. Hi', 'Officer'],
      source:
        'Zachary (1977), "An information flow model for conflict and fission in small groups", Journal of Anthropological Research 33(4); networkx karate_club_graph',
      url: 'https://doi.org/10.1086/jar.33.4.3629752',
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/real')

dataset(
  {
    key: 'iris',
    name: 'Iris',
    summary: "Fisher's iris measurements: 150 flowers, four features, three species.",
    task: 'classification',
    output: 'dataset',
    knobs: space({}),
    truth: false,
    random: false,
  },
  iris,
)

dataset(
  {
    key: 'oldFaithful',
    name: 'Old Faithful',
    summary: 'Eruption durations and waiting times of the Old Faithful geyser.',
    task: 'clustering',
    output: 'dataset',
    knobs: space({}),
    truth: false,
    random: false,
    notes: ['gaussian-mixture-model'],
  },
  oldFaithful,
)

dataset(
  {
    key: 'anscombe',
    name: "Anscombe's quartet",
    summary: 'Four small datasets with the same summary statistics and different shapes.',
    task: 'regression',
    output: 'datasets',
    knobs: space({}),
    truth: false,
    random: false,
    notes: ['linear-regression'],
  },
  anscombe,
)

dataset(
  {
    key: 'coalMining',
    name: 'Coal-mining disasters',
    summary: 'Yearly counts of British coal-mining disasters, 1851–1962.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({}),
    truth: false,
    random: false,
  },
  coalMining,
)

dataset(
  {
    key: 'karateClub',
    name: "Zachary's karate club",
    summary: 'A 34-member friendship graph labelled by the club each member joined after the split.',
    task: 'classification',
    output: 'dataset',
    knobs: space({}),
    truth: false,
    random: false,
    notes: ['graph-convolutional-network', 'label-propagation'],
  },
  karateClub,
)
