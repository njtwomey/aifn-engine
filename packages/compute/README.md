# aifn-compute

The numerical core of the AI Field Notes engine: tested, traceable, tensor-native numerics and the protocols the applications
implement. What belongs here, and what belongs in `aifn-methods`, is set by `../../README.md`. This file is the contract every compute module follows; applications follow it too.

## Layout

- `src/<family>/<module>/index.ts` is a module's public surface, imported as `aifn-compute/<family>/<module>` (e.g.
  `aifn-compute/numerics/linalg`); `src/<family>/index.ts` is the family's common surface, and the package root `aifn-compute`
  (`src/index.ts`) re-exports foundation's (tensors, `grad` and friends, streams, runners) for learners. A module may
  split its code into several files in its folder; only `index.ts` is public. Shared files at a family's root (e.g.
  `optim/options.ts`) serve its modules.
- `test/<family>/<module>/*.test.ts` mirrors the source tree and holds the vitest tests. Root files hold the generated
  suites: `test/primitives.test.ts` (every registered primitive), `test/registries.ts` (loads every registry) and the protocol helpers (`test/protocol.ts`).
- `test/fixtures/<family>/…json` holds golden values written in Python by `test/fixtures/generate.py` and the scripts
  in `test/fixtures/gen/` (`make fixtures`, or `make fixtures FIXTURES="numerics/linalg …"`). Tests read fixtures;
  they never call Python.
- `bench/compute.bench.ts` holds micro-benchmarks (`make bench`; reported, not gated).
- Modules import each other only through `aifn-compute/<family>/<module>`, never by relative path across modules (except a
  family's shared files), and only downwards in the tier order (no cycles; `../../README.md`, Import rules and Layers).
- No dependency outside this repository. No React, no DOM: aifn runs in Node, in the browser and in a Web Worker.

## Running

| Command         | What it does                                                                                                                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `make test`     | the layer lint, then vitest for compute, methods and render                                                                                                                                                         |
| one file        | `npx vitest run --config packages/compute/vitest.config.ts test/numerics/linalg/lanczos.test.ts`                                                                                                                    |
| `make bench`    | `vitest bench` over `bench/compute.bench.ts`: scalar dispatch, elementwise, reductions, gradients, linear algebra, small matrices, normal draws, first-order methods and Runge–Kutta on primitives against plain JS |
| `make fixtures` | regenerate the golden values from Python (numpy, scipy, scikit-learn, torch, mpmath)                                                                                                                                |
| `make lint`     | oxlint, Prettier, the layer lint (`node scripts/layers.ts`), the name-collision lint and ruff                                                                                                                       |

## Layers

A module may import (values or types) only from modules in strictly lower tiers, and never from `aifn-methods`. The
tiers are listed in `../../modules.json`; the table is in `../../README.md`, generated from it, and
`node scripts/layers.ts` (run by `make lint` and `make test`, and so by `make check`) fails on any import that
goes up or across a tier, on a relative import into another module's folder, and on a module missing from the file.

Reading the order: `foundation` first (contracts, tensor with its primitives, autodiff, random, trace), then
`numerics` (special functions, linear algebra, polynomials, quadrature, roots, implicit differentiation, geometry,
interpolation), `graph`, then `probability`, `optim` and `systems`, then `inference`, `dynamics`, `signal` and
`transport`, then `learning` (estimators, kernels, losses, metrics, compositions, validation), and `nn` last. Datasets
and named models are applications. Protocol types are defined once in `aifn-compute/foundation/contracts`, so modules that
share one (`LogDensity` for MCMC and VI) never import each other for it.

## Style

- TypeScript, strict. Readable before clever: these implementations are read by people learning the ideas. Comment the
  non-obvious _why_ and cite the source of an algorithm in a comment (author, year, equation or section).
- Functions over classes, plain data over hidden state. Where a stateful object is natural (a fitted model, a
  distribution) it is a plain object whose fields are public and documented.
- Every exported function has a TSDoc comment stating inputs (with shapes), outputs and conventions.
- Numerical failure is reported, never hidden: a result carries `jitter`, `singular`, `diverged` or `converged` flags
  where they apply. No silent clamping or flooring.
- Names spell ideas out (`choleskyDecomposition` may be shortened to `cholesky`; never cryptic abbreviations).

## Numbers: `aifn-compute/foundation/tensor`

```ts
type DType = 'bool' | 'int32' | 'float32' | 'float64' | 'complex128'
interface Tensor {
  readonly shape: readonly number[] // [] for a scalar tensor
  readonly strides: readonly number[] // in elements (complex elements for complex128), row-major by default
  readonly offset: number
  readonly dtype: DType
  readonly data: Float64Array | Float32Array | Int32Array | Uint8Array // complex128: interleaved re, im
}
type Vector = Tensor // rank 1
type Matrix = Tensor // rank 2
```

- Tensors are immutable by convention: operations return new tensors (views share `data` when no copy is needed).
- Constructors: `tensor(nested | flat, shape?)`, `zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `fromRows`.
- Converters for the chart boundary: `toArray` (nested `number[]…`), `toRows` (`number[][]`), `toFlat` (`number[]`).
- Dtypes mix by one promotion table (`tensor/dtype.ts`); comparisons give `bool`. Complex128 (design K §8.1):
  `complex(re, im)`, `tensor([{ re, im }, …])`, `zeros(shape, 'complex128')`; `conj`, `realPart`/`imagPart`
  (zero-copy float64 views), `abs` (modulus), `angle`, `expj`; arithmetic, exp/log/sqrt/pow, sum/mean/cumsum,
  matmul/einsum and every structural primitive accept it, ordering operations throw `DTypeError`. `toFlat`
  interleaves (re, im), `toArray` adds a trailing pair axis, `toComplexFlat`/`toComplexArray` give `{ re, im }`.
  Autodiff treats complex values as pairs of reals (docs/aifn-autodiff.md §12).
- Fourier (`aifn-compute/foundation/fourier`, design K §8.2): `fft`, `ifft`, `rfft`, `irfft` are linear primitives on
  complex128 along any axis (`{ axis, n, norm: 'backward' | 'ortho' | 'forward' }`; radix-2 or Bluestein per line, so
  leading axes batch). Their transposes are the ℝ² adjoints (fft ↔ ifft with the dual norm; irfft's weights cₖ = 2
  on interior bins). `dftMatrix(n)` and `dft(x)` (a matmul) are the O(n²) definition; `fftn`/`fft2`, `fftshift`,
  `fftfreq`, `rfftfreq` are compositions or constructors; `dct`/`idct` are products with `dctMatrix(n)`.
- Convolution (`aifn-compute/foundation/convolution`, design K §8.3): one family. `conv(x, w, { layout, stride, dilation,
groups, padding, flip, method })` on [N, C, ...S] × [O, C/groups, ...K] (a rank-1 signal is [1, 1, n]; `flip`
  true = convolution, false = correlation as in nn; `method` direct | fft | overlapAdd | auto changes only the
  kernel); `convTranspose` is its input adjoint. Three bilinear primitives (conv, convTranspose, convWeight) are each
  other's transposes, and batch into N or into channel groups without a loop. `pad` (constant, reflect, symmetric,
  edge, wrap) is a linear primitive. dsp `convolve`/`correlate`/`fftConvolve`/`upfirdn`, image
  `correlate2d`/`convolve2d` (pad, then valid), nn `conv1d`/`conv2d`/`avgPool*` and stats' lagged products are
  compositions over it.
- Elementwise operations broadcast (NumPy rules). Reductions take `axis?: number | number[]` and `keepDims?: boolean`.
- Performance-critical inner loops may work on `data` directly (the `dense` namespace has the shared kernels for this),
  but public functions take and return `Tensor` (or plain numbers), not raw arrays. Data arguments may also be
  `VectorLike` (a rank-1 tensor or an array of numbers) or `MatrixLike` (a rank-2 tensor or rows of numbers).

## One definition per operation

Every mathematical operation is defined **once**, as a primitive with its forward computation and its derivative rule.
There is no separate scalar, tensor and differentiable version of the same function.

- A primitive accepts numbers, tensors and traced values: `softplus(2)` returns a number, `softplus(t)` a tensor
  (elementwise), and inside `grad(f)`, `jvp` or `vmap` the same call is handed to that transform's interpreter.
  Autodiff is a mode, not a parallel set of functions.
- Primitives are declared with `elementwise({ id, f, derivative })` (elementwise with broadcasting: a scalar rule and
  **one** derivative per argument, written with primitives, so derivatives of any order work) and
  `definePrimitive`/`defineOp` (general, e.g. reductions and matmul: a vjp and a jvp written with primitives, or a
  `transpose` for a linear primitive, plus optional batching and shape rules), all from `aifn-compute/foundation/tensor`. Derivatives are written next to the forward rule and tested against finite differences.
- Modules above `tensor` (e.g. `numerics/special` for erf, logΓ, softplus, sigmoid) register their functions through
  `elementwise`, so each function exists in one place and is differentiable wherever its derivative is known. A
  primitive without a derivative (e.g. a discrete sampler) says so, and differentiating through it is an error, not a
  silent zero.
- `aifn-compute/foundation/autodiff` supplies three interpreters (reverse, forward, batch) and the transforms built on them:
  `grad`, `valueAndGrad`, `vjp`, `jvp`, `linearize`, `hvp`, `jacobian`, `hessian`, `vmap`, `stopGradient`. It does
  not redefine operations. Custom rules for composite functions (`customVjp`, `customJvp`, `defineCustomVjp`) and
  `checkpoint` live there too; `aifn-compute/foundation/trace` differentiates an `Algorithm` by its steps (`unrolled`).
  Implicit differentiation of solvers (`implicitFixedPoint`, `implicitRoot`, and `atConvergence` for an `Algorithm`)
  needs a linear solve, so it is `aifn-compute/numerics/implicit`, on `aifn-compute/numerics/linalg`'s `solve`.
- **Shapes, not scalars, are the public surface.** Scalar kernels are internal to their module and never exported.
  Every public function of numbers accepts `Scalar | Tensor` (any rank) and broadcasts over all its arguments, so
  `erf(0.5)`, `erf(vector)` and `erf(matrix)` all work and keep their shapes. Samplers likewise take tensor-valued
  parameters with broadcasting and an optional `shape`: `normal(s, mean, sd, { shape })` returns a tensor of draws.
- Every elementwise primitive is defined with `elementwise` (its scalar rule and one derivative per argument, written
  with primitives); general ones with `definePrimitive` or `defineOp`. `sumLike` (reduce a broadcast cotangent
  back to an input's shape) is exported for modules that define their own ops.
- Outputs are tensors: any function whose result is an array of numbers returns a `Tensor` (ranks, z-scores,
  histograms' counts, autocorrelations), numbers stay numbers.
- Samplers return float64 tensors for values and int32 tensors for indices (`categorical`, `choice`, `permutation`);
  counts that can be NaN for invalid parameters (Poisson, binomial) stay float64. Sampler parameters may not be traced:
  pathwise draws are the distributions' `rsample` (location–scale and inverse-transform families, `Transformed`, the
  multivariate normal through its Cholesky factor), differentiable in the parameters.

## Randomness: `aifn-compute/foundation/random`

```ts
interface Key {
  readonly path: string
  readonly hash: readonly [number, number, number, number]
}
interface Stream {
  readonly key: Key
  position: number
} // plain data: structured-cloneable, no methods
function stream(seed: number | string): Stream
function child(s: Stream | Key, ...path: (string | number)[]): Stream // an independent stream at position 0
function randomBits(s: Stream, n: number): Uint32Array // the one random primitive; advances s.position by n
```

- Philox4x32-10 (Salmon et al., 2011) keyed by a 128-bit hash of the key path (e.g. `7/chain:3/env`). Path elements
  are compared as strings, so `child(s, 3)` and `child(s, '3')` are the same stream.
- A draw depends only on the key and the position: `child(s, 'chain', k)` gives the same draws however many values
  `s` or other children have drawn, and a stream sent to a worker draws the same values there.
- Every sampler takes the stream first and draws a whole block of words at once: `uniform(s, a, b, { shape })`,
  `normal(s, mean, sd, { shape })`, `integers(s, n)`; named families (`gammaVariate`, `beta`, `poisson`,
  `multivariateNormal`, …) are in `aifn-compute/probability/samplers`. Rejection samplers key each element by its own child
  stream (`drawEach`), so a variable number of trials never shifts another element's draws.
- `replicate(n, s, fn)` runs `fn(child(s, k), k)` for `k < n` and caches results by key.

## Traces: `aifn-compute/foundation/trace`

```ts
interface Algorithm<Start, State extends Status> {
  name: string
  init(start: Start, s: Stream): State
  step(state: State, ctx: StepContext): State // pure; ctx = { t, stream: child(root, 'step', t) }
  done?(state: State): boolean
}
interface Status {
  t: number
  converged?: boolean
  diverged?: boolean
  stalled?: boolean
  terminated?: boolean
}
```

- Algorithms are factories: `method(problem…, options?)` returns an `Algorithm` whose `init` takes only an optional
  start. States are plain data (never a stream): a stochastic step draws from `ctx.stream`.
- Runners: `run(alg, start, n, { stream })`, `trace(alg, start, n, { every, record, checkpointEvery, keep, timing })`,
  `seek(alg, start, i, { checkpoints, stream })`, `extend(previous, alg, m)`, `live(alg, start)`, `timeSliced(…)`.
  A `Trace` (`kind: 'trace'`) holds `index` (Int32Array), `series` (tensors over kept steps), `steps` and
  `checkpoints` (when kept), `final`, `timing` and `meta` (`start`, `key`, `keep`, `timing`, `recorders`).
- The runner stops on `diverged`, then on `converged`, `terminated` or `done(state)`. A recorded value that is
  infinite, or NaN after the series has been finite, also counts as divergence; NaN before any finite value means
  "not defined yet".

## Distributions: `aifn-compute/probability/distributions`

Plain objects (`kind: 'distribution'`) with `logProb`, `prob`, `cdf`/`logcdf` (where defined), `quantile`
(univariate), `sample(s, { shape })`, `rsample(s, { shape })` where a pathwise draw exists, and the methods `mean()`,
`variance()` or `covariance()`, `stddev()`, `entropy()`, `mode()`, plus `support`, `batchShape`, `eventShape`.
Gaussians are parameterised by mean and standard deviation (or covariance), like scipy's `scale`; exponential
families also expose `expFamily` (natural parameters, sufficient statistics, log-partition).

- Draws have shape `[...sampleShape, ...batchShape, ...eventShape]`, as in PyTorch.
- `rsample` is written with primitives, so it is differentiable in the parameters (location–scale and
  inverse-transform families, `Transformed`, `MultivariateNormal`); families whose draws need gamma variates (Gamma,
  Beta, Dirichlet, Student t, …) have none yet and say so.
- Moments are methods, not fields, so they are computed only when asked for.
- `Univariate` accepts any univariate distribution; `Univariate<number>` is one with number parameters, whose
  methods return numbers.
- Elementwise functions that are not about distributions (`xlogy`, `xlog1py`, Bessel functions) belong in `special`.

## Conventions fixed once

Kernels by lengthscale ℓ. Eigenvalues in descending order. Histograms return `{ edges, counts, density }` and count
values equal to the last edge in the last bin. Ranks average ties. nDCG gain explicit, default 2^rel − 1. The stream
comes first in every sampler. Angles in radians. Probabilities as probabilities, log-probabilities named `log…`.

## Conventions (draft, pending the architecture review)

These rules come from the consolidation audit (`.scratch/aifn/consolidation.md` §9). They are a draft: the
architecture review of the whole system (tensor and autodiff included) may change them. Each rule is followed by its
reason. Code written now should follow them; existing code is brought into line module by module.

### Naming

- **N1. Algorithms are factories.** `method(problem…, options?)` returns an `Algorithm<Start, State>` whose
  `init(start, s?)` takes only an optional initialisation (`{ x0 }`, `{ centroids }`), so `trace(alg, {}, n)` always
  works. _Reason:_ most Algorithms already have this form; a generic player, `seek` and the lab cannot drive an
  Algorithm whose `init` needs the whole problem, and cannot tell the two forms apart from the type.
- **N2. An Algorithm is named for its method** (`adam`, `hmc`, `brent`, `heatEquation`). It takes the suffix `Steps`
  only when the module also exports a one-call convenience with the method's natural name (`kmeansSteps` and
  `kmeans`). No other suffixes (`Fitter`, `Search`). _Reason:_ this is what almost every module already does, and it
  keeps short names where the runner is generic (`minimize`, `solveIvp`, `sampleChains`).
- **N3. Runners are verbs or the method's name**: `solveX`, `fitX`, `findX`, `integrate`, or the bare method name
  paired with N2. Estimators are nouns (`gaussianMixture`). _Reason:_ one reading per name.
- **N4. PascalCase only for distribution constructors and nn layers** (`Normal`, `Linear`); everything else is
  camelCase. _Reason:_ already true; it mirrors torch and separates `Normal` (an object) from `normal` (a sampler).
- **N5. No two modules export different things under one name.** A deliberate re-export of the same object is allowed.
  _Reason:_ figures import from several modules at once, and a name that means two things (`trace`, `adam`,
  `shuffle`) is a bug waiting in every such file.

### Arguments

- **A1. The stream comes first** in every function whose output is random (`normal(s, mean, sd)`,
  `bootstrap(s, x, statistic, n)`); Algorithms take it in `init(start, s)`; estimators and runners take it as
  `options.stream`. No `() => number` sources. _Reason:_ keyed, reproducible randomness (see Randomness); an unkeyed
  function source is a door for draws that no stream accounts for.
- **A2. Inputs and outputs.** Public numeric functions accept `number | Tensor` and broadcast. Data arguments may be
  `VectorLike` or `MatrixLike`, defined once in `aifn-compute/foundation/contracts`; no module defines its own input alias. Any result that
  is an array of numbers is a `Tensor` (int32 for indices, counts of items and lags); index lists passed in may be
  `number[]`. _Reason:_ one boundary relaxation instead of a dozen local ones, and one output type that every chart
  helper and every downstream function reads.

### Results and state

- **R1. State flags.** Every Algorithm state carries `t` (steps taken). Iterative algorithms carry `converged` and
  `diverged`; finite ones carry `done`; exact solvers may add a `status` enum. `stalled`, `terminated`, `singular`
  and `jitter` appear where they apply. _Reason:_ the trace stops on `diverged`, and readouts and players need `t`
  without knowing the algorithm.
- **R2. Field names.** The iterate is `x` (a domain name where the iterate is a named object: `weights`, `params`,
  `position`); `value` is an objective being optimised, `loss` a training loss, `logLikelihood` and `elbo` are named;
  the gradient is `grad`. _Reason:_ charts and readouts read the same key across algorithms.
- **R3. Option names.** `tolerance` (or `atol`/`rtol`), `maxSteps` for any step budget, `stepSize` in optim, mcmc, vi
  and ode, `learningRate` for learners. _Reason:_ names are spelled out (see Style), and `maxSteps` matches
  `trace(…, n)`.

### Capabilities

- **C1. Every fitted model is built by an `Estimator`** and declares what it can do. Classifiers: `decide`, `score`,
  and `predictive` when probabilistic (no separate `probabilities` method). Regressors: `decide` and `expect`, and
  `predictive` when there is a noise model. Clusterers: `decide` when inductive; transductive models (agglomerative,
  spectral, OPTICS) say so with `transductive: true` and expose `labels`. Embedders: `transform` when they map new
  points, otherwise `embedding` and `transductive: true`. Inverse maps are `inverseTransform`. Iterative fits are
  `Trained` and copy `converged` and `diverged` from the final state. _Reason:_ figures branch on capabilities
  (`hasPredictive`, `hasTransform`), not on model kinds; a model that does not declare what it can do forces every
  figure to know it by name.

### Shared helpers, defined once

- Inputs and inner loops: `VectorLike`, `MatrixLike` (`aifn-compute/foundation/contracts`, re-exported by tensor) and the
  `dense` kernels (`dense.toF64`, `dense.toMatrixF64`, `dense.dot`, `dense.matVec`, `dense.matMul`, `dense.axpy`, …) in
  `aifn-compute/foundation/tensor`; indexed reads and writes `gather`,
  `scatterAdd` and `take` there too.
- Linear algebra: `squaredDistances` and `pairwiseDistances`, `solveDense` (small systems in inner loops), the
  general eigenproblem `eig`, `expm`, `matrixTrace`, and `LinearOperator` (a matrix or a function v ↦ Av) with the
  matrix-free Lanczos eigensolver `eigsh`, in `aifn-compute/numerics/linalg`.
- Elementwise functions: `xlogy`, `xlog1py`, `besselI0`, `besselI1`, `logBesselI0`, `besselRatio` in
  `aifn-compute/numerics/special`; `logsumexp` in `aifn-compute/foundation/tensor` (no aliases).
- Scalar minimisation: `minimizeScalar` (Brent, golden section) in `aifn-compute/numerics/roots`.
- Implicit differentiation: `implicitFixedPoint`, `implicitRoot`, `atConvergence` in `aifn-compute/numerics/implicit`.
- Sequences and weights: `autocorrelation` and `autocovariance` (FFT for long sequences) and
  `importanceEffectiveSampleSize` in `aifn-compute/probability/stats`.
- Protocol types: `LogDensity` (an unnormalised log-density; the old name `Target` is gone) and the other protocols in
  `aifn-compute/foundation/contracts`.
- Optimisation test surfaces (`rosenbrock`, `himmelblau`, …) are data, in `aifn-methods/data/objectives`.

A module that needs one of these imports it; it does not keep a private copy.

## Tests

- Run them with `make test` (both packages) or one file with `npx vitest run --config packages/compute/vitest.config.ts
<path>`.
- Generated suites cover every registry entry without a hand-kept list: every primitive (values, broadcasting, dtypes,
  vjp and jvp against finite differences and each other, batch rules, input immutability), every algorithm (the trace
  protocol on a case keyed by its address; a new algorithm without a case fails), every distribution family (info,
  Kolmogorov–Smirnov against its cdf, moments) and the metadata of windows, wavelets, kernels, bijectors, links,
  likelihoods, filter designs and KL rules (`../../README.md`, Registries).
- Each exported function has tests. Deterministic numerics are checked against Python fixtures (numpy, scipy,
  scikit-learn, torch) at stated tolerances. Stochastic code gets statistical tests (moments, KS against a reference
  cdf) with fixed streams, plus determinism tests.
- Every `Algorithm` gets protocol tests: same seed → same trace; `seek(i)` equals `run(i)`; `extend` equals a longer
  `trace`.

## Examples

`make examples` starts the `aifn-render` gallery (http://localhost:5192/): one recipe per page, each a live figure
with its own source (`../../examples/README.md`).
