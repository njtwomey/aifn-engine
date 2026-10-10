/**
 * Small real datasets, embedded: Iris, Old Faithful, Anscombe's quartet, the coal-mining disasters and Zachary's karate
 * club.
 *
 * Each function takes no arguments and builds its dataset afresh from the constants of `embedded.ts`, which hold the
 * published values verbatim; nothing is random. Features are a float64 matrix `x`, one row per observation; class
 * labels, where there are any, are int32 indices into `meta.labelNames`. Every dataset's `meta` names its source
 * (author, year, journal) and, where there is one, a URL. Each is also registered under `data/real`.
 */

import { dense, fromData } from 'aifn-compute/foundation/tensor'
import { fromEdges, type Graph } from 'aifn-compute/graph'
import { ANSCOMBE_DATA, COAL_MINING_DATA, FAITHFUL_DATA, IRIS_DATA, KARATE_CLUBS, KARATE_EDGES } from './embedded'
import { labels, matrix, vector, type Dataset } from '../types'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { space } from 'aifn-compute/foundation/space'

/**
 * Fisher's Iris data: 150 flowers (50 each of setosa, versicolor and virginica) with four measurements in cm: sepal
 * length, sepal width, petal length and petal width. The rows are in species order. Source: Fisher (1936), "The use of
 * multiple measurements in taxonomic problems", Annals of Eugenics 7(2), in the UCI Machine Learning Repository copy
 * as scikit-learn distributes it.
 *
 * @returns The dataset: `x` ($150 \times 4$, cm), `y` the species (0 setosa, 1 versicolor, 2 virginica), and `meta`
 *   with the feature and species names.
 *
 * @example Shape, columns and the first rows
 * const d = iris()
 * print('x:', d.x.shape, ' features:', d.meta.featureNames)
 * print('first rows:', toArray(d.x).slice(0, 3))
 * print('species:', d.meta.labelNames, ' first labels:', toArray(d.y).slice(0, 3))
 *
 * @example Mean petal length per species
 * const d = iris()
 * const x = toArray(d.x)
 * const y = toArray(d.y)
 * for (const c of [0, 1, 2]) {
 *   const petal = x.filter((_, i) => y[i] === c).map((r) => r[2])
 *   print(d.meta.labelNames[c], petal.reduce((a, b) => a + b) / petal.length, 'cm')
 * }
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
 * Old Faithful geyser data: 272 eruptions with duration and waiting time to the next eruption, both in minutes. There
 * are no labels, but the data form two clear clusters (short eruptions followed by short waits, long by long). Source:
 * Härdle (1991), "Smoothing Techniques with Implementation in S", as R's `datasets::faithful`.
 *
 * @returns The dataset: `x` ($272 \times 2$: eruption duration, waiting time), no `y`, and `meta` with the column
 *   names.
 *
 * @example Shape, columns and the first rows
 * const d = oldFaithful()
 * print('x:', d.x.shape, ' features:', d.meta.featureNames)
 * print('first rows:', toArray(d.x).slice(0, 3))
 *
 * @example The two clusters, split at a 3-minute eruption
 * const x = toArray(oldFaithful().x)
 * const short = x.filter((r) => r[0] < 3)
 * const long = x.filter((r) => r[0] >= 3)
 * const meanWait = (rows) => rows.reduce((a, r) => a + r[1], 0) / rows.length
 * print('short eruptions:', short.length, ' mean wait:', meanWait(short), 'min')
 * print('long eruptions:', long.length, ' mean wait:', meanWait(long), 'min')
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
 * Anscombe's quartet: four sets of eleven $(x, y)$ points with the same means, variances, correlation ($0.816$) and
 * least-squares line ($y = 3 + 0.5x$), yet very different shapes: a noisy line, a curve, a line with one outlier, and
 * a vertical stack with one point far to the right. Source: Anscombe (1973), "Graphs in statistical analysis", The
 * American Statistician 27(1).
 *
 * @returns The four sets, I to IV in order, each a dataset with `x` ($11 \times 1$) and the targets `y` (length
 *   $11$).
 *
 * @example The four sets share their summary statistics
 * const mean = (v) => v.reduce((a, b) => a + b) / v.length
 * for (const d of anscombe()) {
 *   const x = toArray(d.x).map((r) => r[0])
 *   const y = toArray(d.y)
 *   const [mx, my] = [mean(x), mean(y)]
 *   const slope = mean(x.map((v, i) => (v - mx) * (y[i] - my))) / mean(x.map((v) => (v - mx) ** 2))
 *   print(d.meta.name, ' mean x:', mx, ' mean y:', my, ' slope:', slope, ' intercept:', my - slope * mx)
 * }
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
 * British coal-mining disasters (explosions that killed ten or more men) per year, 1851–1962. The classic Poisson
 * changepoint series: the rate drops from about 3 to about 1 per year around 1890, and Adams and MacKay (2007) ran
 * BOCPD on it with a Poisson–gamma segment model. No truth: the changepoint is what is inferred. Source: Jarrett
 * (1979), "A note on the intervals between coal-mining disasters", Biometrika 66(1), counted by year as in Carlin,
 * Gelfand and Smith (1992), Applied Statistics 41(2).
 *
 * @returns The dataset: `x` the counts ($112 \times 1$, one row per year), `t` the calendar years (length $112$), and
 *   no `y`.
 *
 * @example The years and the first counts
 * const d = coalMining()
 * print('x:', d.x.shape, ' years:', toArray(d.t)[0], 'to', toArray(d.t).at(-1))
 * print('first five years:', toArray(d.x).slice(0, 5).map((r) => r[0]))
 *
 * @example The rate before and after 1890
 * const d = coalMining()
 * const counts = toArray(d.x).map((r) => r[0])
 * const years = toArray(d.t)
 * const rate = (keep) => {
 *   const c = counts.filter((_, i) => keep(years[i]))
 *   return c.reduce((a, b) => a + b) / c.length
 * }
 * print('disasters in all:', counts.reduce((a, b) => a + b))
 * print('per year, 1851-1889:', rate((t) => t < 1890), ' 1890-1962:', rate((t) => t >= 1890))
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
  /** The graph on the rows: node $i$ is row $i$ of `x` and entry $i$ of `y`. */
  readonly graph: Graph
}

/**
 * Zachary's karate club: 34 members joined by 78 friendships, labelled by the club each joined when the club split (0:
 * Mr. Hi, 1: the Officer). There are no measured features: each member is its own one-hot feature, as in Kipf and
 * Welling's experiment. Source: Zachary (1977), "An information flow model for conflict and fission in small groups",
 * Journal of Anthropological Research 33(4), as networkx's `karate_club_graph`.
 *
 * @returns The dataset: `x` the $34 \times 34$ identity, `y` the club of each member, and `graph` the undirected
 *   friendship graph (34 nodes, 78 edges).
 *
 * @example The graph and the two clubs
 * const d = karateClub()
 * print('x:', d.x.shape, ' nodes:', d.graph.nodes, ' edges:', d.graph.edges.length)
 * print('first edges:', d.graph.edges.slice(0, 3).map((e) => [e.from, e.to]))
 * const y = toArray(d.y)
 * print('Mr. Hi:', y.filter((c) => c === 0).length, ' Officer:', y.filter((c) => c === 1).length)
 *
 * @example The two leaders have the most friends
 * const d = karateClub()
 * const degree = new Array(d.graph.nodes).fill(0)
 * for (const e of d.graph.edges) {
 *   degree[e.from]++
 *   degree[e.to]++
 * }
 * print('member 0 (Mr. Hi):', degree[0], ' member 33 (the Officer):', degree[33])
 * print('largest degree of anyone else:', Math.max(...degree.slice(1, 33)))
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
