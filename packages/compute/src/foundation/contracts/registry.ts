/**
 * Registry metadata (design S §3.1): one `Info` pattern for every kind of named entry (primitives, metrics, losses,
 * distribution families, kernels, windows, datasets, models, algorithms, …). Keys are addresses: `metrics/auroc`
 * names one entry across the lab, the site, URLs, fixtures and workers. Metric and loss metadata specialise it.
 */

import type { DomainKind } from './gym'
import type { Capability } from './model'
import type { Space } from './space'

/** The kinds of registry entries. The lab adds `view`. */
export type EntryKind =
  | 'primitive'
  | 'metric'
  | 'loss'
  | 'distribution'
  | 'bijector'
  | 'kernel'
  | 'window'
  | 'wavelet'
  | 'filter-design'
  | 'dataset'
  | 'modifier'
  | 'objective'
  | 'model'
  | 'environment'
  | 'agent'
  | 'log-density'
  | 'algorithm'
  | 'function'
  | 'test'
  | 'engine'
  | 'kl-rule'
  | 'likelihood'
  | 'link'

/** How settled an entry is: stable entries change only through deprecation. */
export type Stability = 'stable' | 'experimental' | 'deprecated'

/** What every registry entry declares. */
export interface Info {
  /** Unique within its kind; equals the export name (`auroc`, `Normal`, `moons`, `adam`). */
  readonly key: string
  /** What kind of entry it is, which fixes the rest of its metadata. */
  readonly kind: EntryKind
  /** The module that defines it, e.g. `metrics`. */
  readonly module: string
  /** Display name, plain text. */
  readonly name: string
  /** The display name in TeX, where it has maths. */
  readonly tex?: string
  /** One sentence, for the catalog and search. */
  readonly summary?: string
  /** Site note slugs; the first is the defining note. */
  readonly notes?: readonly string[]
  /** A `content/glossary.yaml` key. */
  readonly glossary?: string
  /** `content/references.yaml` keys. */
  readonly cite?: readonly string[]
  /** Free keywords, for search and filtering. */
  readonly tags?: readonly string[]
  /** How settled the entry is. */
  readonly stability: Stability
  /** For deprecated entries: `module/key` of the replacement. */
  readonly replacedBy?: string
  /** True when it takes a stream (the lab then adds a seed control). */
  readonly random?: boolean
}

/** A value carrying its registry metadata. */
export type Entry<T, I extends Info = Info> = T & { readonly info: I }

// ── Metrics and losses ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What a metric reads from a model or a prediction, so a caller can feed it the right output: `labels` (decisions),
 * `sets` (multi-label 0/1 rows), `scores` (an ordering), `probabilities`, `distribution` (a predictive), `values`
 * (point predictions of a real target), `ranking`, `partitions`, `features`, `ratings`, `boxes`, `masks`, `points`,
 * `sequences`, `images`, `signals`, `vectors` and `exposure`.
 */
export type InputKind =
  | 'labels'
  | 'sets'
  | 'scores'
  | 'probabilities'
  | 'distribution'
  | 'values'
  | 'ranking'
  | 'partitions'
  | 'features'
  | 'ratings'
  | 'boxes'
  | 'masks'
  | 'points'
  | 'sequences'
  | 'images'
  | 'signals'
  | 'vectors'
  | 'exposure'

/** The capabilities a metric can need from a model: hard decisions, an ordering of cases, or a predictive. */
export type MetricCapability = Extract<Capability, 'decide' | 'score' | 'predictive'>

/** Metric metadata: the registry `Info` plus what the metric reads, its direction and its range. */
export interface MetricInfo extends Info {
  /** The entry kind of a metric. */
  readonly kind: 'metric'
  /** What it reads from a model or a prediction. */
  readonly inputs: InputKind
  /** `higher` when a larger value is better, `lower` when a smaller one is. */
  readonly direction: 'higher' | 'lower'
  /** The smallest and largest values it can take (either may be infinite). */
  readonly range: readonly [number, number]
  /** What it needs of a model, when it evaluates one. */
  readonly capability?: MetricCapability
}

/** The family a loss belongs to, as the site's notes group them. */
export type LossFamily =
  | 'classification'
  | 'regression'
  | 'ranking'
  | 'retrieval'
  | 'representation'
  | 'divergence'
  | 'adversarial'
  | 'energy'
  | 'preference'

/**
 * What a loss reads from a model: `logits`, `probabilities`, `margins` (scores with labels in $\{-1, +1\}$), `values`,
 * `distribution` (predictive parameters), `scores` (of the items of a list), `embeddings` or `distributions` (two to
 * compare), or `log-probabilities` (of whole responses under a policy and a reference, as preference losses read
 * them). Shared names mean the same as in `InputKind`.
 */
export type LossInput =
  | 'logits'
  | 'probabilities'
  | 'margins'
  | 'values'
  | 'distribution'
  | 'scores'
  | 'embeddings'
  | 'distributions'
  | 'log-probabilities'

/** Loss metadata: the registry `Info` plus its family, inputs, and the metric its minimiser optimises. */
export interface LossInfo extends Info {
  /** The entry kind of a loss. */
  readonly kind: 'loss'
  /** The family it belongs to. */
  readonly family: LossFamily
  /** What it reads from a model. */
  readonly inputs: LossInput
  /** What the model's output estimates at the loss's minimiser, in words (e.g. the log-odds of $y = 1$). */
  readonly target?: string
  /** The key of the metric this loss's minimiser optimises (`metrics/<key>`). */
  readonly pairedMetric?: string
}

// ── Probability ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Where a family puts its mass, as a name the catalog and pickers can read without building a distribution: `real`,
 * `positive` $(0, \infty)$, `non-negative` $[0, \infty)$, `unit-interval` $(0, 1)$, `interval` (set by parameters),
 * `circle`, `integers` (bounded by parameters), `non-negative-integers`, `positive-integers`, `binary` $\{0, 1\}$,
 * `categories` $\{0, \dots, K - 1\}$, `simplex`, `real-vector`, `count-vector` and `positive-definite`. Mixtures and
 * compositions use `varies`.
 */
export type SupportName =
  | 'real'
  | 'positive'
  | 'non-negative'
  | 'unit-interval'
  | 'interval'
  | 'circle'
  | 'integers'
  | 'non-negative-integers'
  | 'positive-integers'
  | 'binary'
  | 'categories'
  | 'simplex'
  | 'real-vector'
  | 'count-vector'
  | 'positive-definite'
  | 'varies'

/**
 * A distribution family (a constructor such as `Normal`): its parameters in constructor order, support and structure.
 */
export interface DistributionInfo extends Info {
  /** The entry kind of a distribution family. */
  readonly kind: 'distribution'
  /** The constructor's parameters, in argument order, with ranges and defaults for pickers and tests. */
  readonly params: Space
  /** Where the family puts its mass. */
  readonly support: SupportName
  /** True for a family over integers (mass functions). */
  readonly discrete: boolean
  /** 0 for scalar events, 1 for vectors, 2 for matrices. */
  readonly eventRank: number
  /** True when instances expose `expFamily` (natural parameters, sufficient statistics, log-partition). */
  readonly expFamily: boolean
  /** True for a constructor that builds on other distributions (Mixture, Independent, Transformed). */
  readonly composite?: boolean
}

/** A bijector (or a bijector factory): its domain and codomain as support names, and its parameters. */
export interface BijectorInfo extends Info {
  /** The entry kind of a bijector. */
  readonly kind: 'bijector'
  /** Where the map is defined. */
  readonly domain: SupportName
  /** The map's image. */
  readonly codomain: SupportName
  /** Factory arguments; empty for a fixed map. */
  readonly params: Space
  /** True when the entry is a factory returning a bijector rather than a bijector. */
  readonly factory: boolean
}

/** A closed-form $\KL(p \,\|\, q)$ rule between two registered families. */
export interface KlRuleInfo extends Info {
  /** The entry kind of a KL rule. */
  readonly kind: 'kl-rule'
  /** The family key of $p$. */
  readonly p: string
  /** The family key of $q$. */
  readonly q: string
}

/** A GLM link function. */
export interface LinkInfo extends Info {
  /** The entry kind of a link function. */
  readonly kind: 'link'
  /** The mean space the link maps from, as a support name. */
  readonly meanSpace: SupportName
}

/** An exponential-dispersion family (a factory returning a `Family`). */
export interface LikelihoodInfo extends Info {
  /** The entry kind of a likelihood family. */
  readonly kind: 'likelihood'
  /** The response's support. */
  readonly support: SupportName
  /** The key of the family's canonical link entry. */
  readonly canonicalLink: string
  /** The links the family is used with (keys of link entries). */
  readonly links: readonly string[]
  /** True when the dispersion is estimated rather than fixed at 1. */
  readonly dispersion: boolean
  /** Factory arguments (the negative binomial's $\theta$); empty otherwise. */
  readonly params: Space
}

// ── Kernels ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A covariance kernel factory: its hyperparameters and whether it depends on $\xvec - \yvec$ only. */
export interface KernelInfo extends Info {
  /** The entry kind of a kernel. */
  readonly kind: 'kernel'
  /** The kernel's hyperparameters, with their defaults. */
  readonly hyper: Space
  /** True when $k(\xvec, \yvec)$ depends only on $\xvec - \yvec$. */
  readonly stationary: boolean
  /** True for a combinator of other kernels (sum, product). */
  readonly composite?: boolean
}

// ── Signals ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A window function: its parameters and its spectral figures of merit at the default parameters (Harris, 1978),
 * measured on the periodic window: the main-lobe width between the first nulls, in DFT bins, and the peak side-lobe
 * level relative to the main lobe, in dB.
 */
export interface WindowInfo extends Info {
  /** The entry kind of a window function. */
  readonly kind: 'window'
  /** The window's parameters, with their defaults. */
  readonly params: Space
  /**
   * The main-lobe width between the first nulls, in DFT bins. Omitted when it depends on the length (a Gaussian
   * window's width is set in samples).
   */
  readonly mainLobeWidth?: number
  /** The peak side-lobe level relative to the main lobe, in dB (negative). */
  readonly sideLobeDb?: number
}

/** A wavelet: its family, vanishing moments and filter length (orthogonal wavelets), or its parameters (continuous). */
export interface WaveletInfo extends Info {
  /** The entry kind of a wavelet. */
  readonly kind: 'wavelet'
  /** The wavelet's family. */
  readonly family: 'haar' | 'daubechies' | 'morlet'
  /** True for a wavelet of the continuous transform. */
  readonly continuous: boolean
  /** True for an orthogonal wavelet of the discrete transform. */
  readonly orthogonal: boolean
  /** Vanishing moments of $\psi$ (orthogonal wavelets). */
  readonly vanishingMoments?: number
  /** The number of taps of each filter (orthogonal wavelets). */
  readonly taps?: number
  /** The wavelet's parameters (continuous wavelets); empty otherwise. */
  readonly params: Space
}

/** A filter design method: FIR or IIR, and the specification fields it honours. */
export interface FilterDesignInfo extends Info {
  /** The entry kind of a filter design method. */
  readonly kind: 'filter-design'
  /** Whether it designs infinite or finite impulse response filters. */
  readonly family: 'iir' | 'fir'
  /** The band types it can design. */
  readonly bands: readonly ('lowpass' | 'highpass' | 'bandpass' | 'bandstop')[]
  /** The specification fields it reads: `order`, `numtaps`, `cutoff`, `passRippleDb`, `stopAttenDb`, `window`. */
  readonly honours: readonly string[]
  /** The method's own parameters, with their defaults. */
  readonly params: Space
}

// ── Algorithms ───────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What an algorithm's factory takes: `objective` (a function to minimise), `least-squares` (residuals), `root` (a
 * scalar equation), `system` (a nonlinear system), `integral`, `ode`, `sde`, `pde` (a discretised partial
 * differential equation), `map` (an iterated map), `log-density` (a target to sample), `factor-graph`, `chain` (chain
 * potentials), `gaussian-model`, `graph`, `flow-network`, `linear-program`, `quadratic-program`, `integer-program`,
 * `assignment`, `dynamic-program`, `riccati`, `transport`, `signal`, `lti-system`, `network` (a model to train),
 * `sequence` (a data stream to filter), `corpus` (words or documents of text, e.g. a tokeniser to train),
 * `search-space` (a root, a refinement operator and a quality to search over), `table` (named columns to mine for
 * patterns, e.g. subgroups), `logic-program` (clauses with a query, or examples to learn clauses from).
 */
export type AlgorithmProblem =
  | 'objective'
  | 'least-squares'
  | 'root'
  | 'system'
  | 'integral'
  | 'ode'
  | 'sde'
  | 'pde'
  | 'map'
  | 'log-density'
  | 'factor-graph'
  | 'chain'
  | 'gaussian-model'
  | 'graph'
  | 'flow-network'
  | 'linear-program'
  | 'quadratic-program'
  | 'integer-program'
  | 'assignment'
  | 'dynamic-program'
  | 'riccati'
  | 'transport'
  | 'signal'
  | 'lti-system'
  | 'network'
  | 'sequence'
  | 'corpus'
  | 'logic-program'
  | 'search-space'
  | 'table'

/** The `Status` flags a state may set. */
export type StatusFlag = 'converged' | 'diverged' | 'stalled' | 'terminated'

/**
 * Which state fields play the roles a generic trace view plots by default (design S §2.3): the `iterate` (x, the
 * position, the weights), the `objective` value being optimised or tracked (a loss, a log-likelihood, an ELBO, a
 * residual norm), its `grad`, and the `stepSize`. Each names a top-level field of the state; a role an algorithm has no
 * field for is omitted. `flags` lists the `Status` flags its states set.
 */
export interface StateRoles {
  /** The field holding the iterate. */
  readonly iterate?: string
  /** The field holding the value optimised or tracked. */
  readonly objective?: string
  /** The field holding the gradient of the objective. */
  readonly grad?: string
  /** The field holding the step size. */
  readonly stepSize?: string
  /** The `Status` flags the algorithm's states set. */
  readonly flags: readonly StatusFlag[]
}

/** An algorithm factory: the problem it takes and the roles of its state's fields. */
export interface AlgorithmInfo extends Info {
  /** The entry kind of an algorithm factory. */
  readonly kind: 'algorithm'
  /** What the factory takes. */
  readonly problem: AlgorithmProblem
  /** Which state fields play the roles a trace view plots. */
  readonly state: StateRoles
}

// ── Functions ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What a registered function computes, for grouping in the catalog and the lab: a `transform` of data or a signal (a
 * DFT, a Hilbert transform, a Box–Cox map), an `estimator` of a quantity from a sample (a correlation, a periodogram, a
 * KDE bandwidth), a `test` (a statistic with its p-value), a `construction` of an object from parameters (a state-space
 * system, a noise schedule, a model specification), a `property` of an object (poles, stability, margins, an ARMA
 * process's roots), a `fit` that returns a fitted object in one call, a `simulation` that draws a path or a sample, a
 * `solver` that returns a solution in one call, and an `inference` that answers a query of a probabilistic model (a
 * posterior marginal, a full conditional).
 */
export type FunctionRole =
  'transform' | 'estimator' | 'test' | 'construction' | 'property' | 'fit' | 'simulation' | 'solver' | 'inference'

/**
 * A named computation that is not stepped through (one call, a result): a transform, an estimator, a construction or a
 * one-shot solver. An iterative procedure a note walks through is an `algorithm` instead, and the function that runs
 * it to the end may be registered here beside it.
 */
export interface FunctionInfo extends Info {
  /** The entry kind of a function. */
  readonly kind: 'function'
  /** What the function computes, for grouping. */
  readonly role: FunctionRole
  /** The `kind` brand of the result where it is a displayable object (`spectrum`, `lti`, `time-frequency`, …). */
  readonly returns?: string
}

// ── Hypothesis tests ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What a hypothesis test reads: `one-sample` (values against a reference value or distribution), `two-sample`
 * (independent samples), `paired` (matched pairs), `k-sample` (several groups), `table` (counts: a goodness-of-fit
 * vector or a contingency table), `series` (an ordered sequence), `survival` (times with censoring flags and groups) or
 * `p-values` (a family of tests, for multiple-testing procedures).
 */
export type TestData =
  'one-sample' | 'two-sample' | 'paired' | 'k-sample' | 'table' | 'series' | 'survival' | 'p-values'

/**
 * A hypothesis test (`aifn-compute/probability/tests`): a function of data returning a `TestResult` (statistic, null
 * law, p-value, alternative, interval and effect size), or, for a multiple-testing procedure, adjusted p-values.
 * `statistic` is the statistic's symbol in TeX, `null` the registry key of the family of its null law (`StudentT`,
 * `ChiSquare`, ...; `exact` for a law the test builds itself), and `alternatives` the alternatives it accepts.
 */
export interface TestInfo extends Info {
  /** The entry kind of a hypothesis test. */
  readonly kind: 'test'
  /** What the test reads. */
  readonly data: TestData
  /** The statistic's symbol, in TeX. */
  readonly statistic: string
  /** The registry key of the family of the null law, or `exact`. */
  readonly null: string
  /** The alternative hypotheses it accepts. */
  readonly alternatives: readonly ('two-sided' | 'less' | 'greater')[]
  /** True for a test that assumes a parametric model of the data (normality, a known variance). */
  readonly parametric: boolean
}

// ── Primitives (the catalog's view of the primitive table) ──────────────────────────────────────────────────────────

/** A primitive as the catalog lists it: its arity, rule sources and documentation. */
export interface PrimitiveInfo extends Info {
  /** The entry kind of a primitive. */
  readonly kind: 'primitive'
  /** Whether it acts elementwise (with broadcasting) or is a general primitive. */
  readonly primitive: 'elementwise' | 'general'
  /** The number of inputs, or `variadic`. */
  readonly arity: number | 'variadic'
  /** How each rule (`vjp`, `jvp`, `batch`, `shape`) was obtained: `own`, `derived` or `missing`. */
  readonly rules: { readonly vjp: string; readonly jvp: string; readonly batch: string; readonly shape: string }
  /** The defining formula, in TeX, where it has one. */
  readonly formula?: string
}

// ── Environments ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Environment metadata: a named environment for sequential decisions (a bandit, a Markov decision process, a control
 * problem) with the space of its scalar parameters. Environments on the protocol (`Environment`) also declare the kinds
 * of their observation and action domains and their optional capabilities, so the lab offers only valid pairs.
 */
export interface EnvironmentInfo extends Info {
  /** The entry kind of an environment. */
  readonly kind: 'environment'
  /** The kind of decision problem. */
  readonly family: 'bandit' | 'contextual-bandit' | 'mdp' | 'control'
  /** Its scalar parameters, with their defaults. */
  readonly params: Space
  /** The kind of its observation domain (environments on the protocol). */
  readonly observation?: DomainKind
  /** The kind of its action domain (environments on the protocol). */
  readonly action?: DomainKind
  /** The optional capabilities it has (environments on the protocol). */
  readonly capabilities?: readonly ('model' | 'oracle' | 'render')[]
}

/**
 * What an agent needs of an environment: domain kinds, an explicit model for planners, and the environment families it
 * is meant for (any when omitted; a bandit policy reads arms, a linear one an arms-by-features context).
 */
export interface AgentRequires {
  /** The kind of observation domain it needs; any when omitted. */
  readonly observation?: DomainKind
  /** The kind of action domain it needs; any when omitted. */
  readonly action?: DomainKind
  /** The kind of explicit model it plans with; none needed when omitted. */
  readonly model?: 'tabular' | 'dynamics'
  /** The environment families it is meant for; any when omitted. */
  readonly families?: readonly EnvironmentInfo['family'][]
}

/** Agent metadata: a learning or acting agent (`Agent`), its hyperparameters and what it requires. */
export interface AgentInfo extends Info {
  /** The entry kind of an agent. */
  readonly kind: 'agent'
  /** Its hyperparameters, with their defaults. */
  readonly params: Space
  /** What it needs of an environment. */
  readonly requires: AgentRequires
}
