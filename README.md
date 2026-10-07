# aifn-engine

The AI Field Notes engine: tensor-native numerics, named models and methods, and the rendering layer for interactive
figures, as three workspace packages with an examples gallery.

| Package        | Folder              | Imported as                                                                       | Holds                                                                     |
| -------------- | ------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `aifn-compute` | `packages/compute/` | `aifn-compute/<family>/<module>` (`aifn-compute` for foundation's common surface) | Tensor-native numerics, generic engines and every protocol                |
| `aifn-methods` | `packages/methods/` | `aifn-methods/<area>/…`                                                           | Named models, problems, environments, datasets and worked examples        |
| `aifn-render`  | `packages/render/`  | `aifn-render` (and `aifn-render/<part>`)                                          | React figures, plots, diagrams, controls and gym views over the other two |

```bash
make install   # npm install (workspaces link the packages to each other)
make check     # lints, types, tests, examples render
make examples  # the aifn-render gallery on http://localhost:5192/
```

## Installing

The three packages are released as tarballs attached to this repository's
[GitHub releases](https://github.com/njtwomey/aifn-engine/releases), not to the npm registry. Depend on a release by
URL:

```json
{
  "dependencies": {
    "aifn-compute": "https://github.com/njtwomey/aifn-engine/releases/download/v0.2.0/aifn-compute-0.2.0.tgz",
    "aifn-methods": "https://github.com/njtwomey/aifn-engine/releases/download/v0.2.0/aifn-methods-0.2.0.tgz",
    "aifn-render": "https://github.com/njtwomey/aifn-engine/releases/download/v0.2.0/aifn-render-0.2.0.tgz"
  }
}
```

```ts
import { tensor } from 'aifn-compute'
import { cholesky } from 'aifn-compute/numerics/linalg'
import { datasetRegistry } from 'aifn-methods/data'
import { Curve, Figure, Plot } from 'aifn-render'
```

- The packages are plain ES modules with type declarations: no TypeScript, no aliases. `aifn-compute` and
  `aifn-methods` need no bundler and run in Node, the browser and a Web Worker.
- `aifn-methods` depends on the `aifn-compute` of the same release, and `aifn-render` on both, so list only the ones
  you import. Every package of a release has the same version; to update, change it in both places of each URL.
- `aifn-render` needs React 19 from the app, and its stylesheet: see `packages/render/README.md` for the two ways to
  take the styles (with or without Tailwind) and for the compute worker.
- To work on the engine and a consumer together, link a local checkout in place of the release (`npm link`, or an
  `overrides` entry pointing at the tarballs `make packages-smoke` leaves in `dist/packages`).

**Releasing.** `make release VERSION=x.y.z` tags `v<version>` on `main` and pushes the tag. The release workflow then
runs `make check`, installs the packed tarballs into an empty project and uses them (`make packages-smoke`), packs them
for publication (`make packages`), and publishes the GitHub release with the tarballs attached. The smoke test does
not open a browser: before a release that changes `aifn-render`'s build, install the tarballs into a small Vite app and
look at a figure that uses the worker.

Inside this repository the packages export their TypeScript source (`exports` in each `package.json`), linked to each
other by the npm workspace, so nothing is built during development.

`aifn-compute` and `aifn-methods` are trees: compute has 14 families in tiers, each holding modules in local
tiers and optional shared files at the family root; applications (the modules of `aifn-methods`) have 17 areas holding
groups and modules, where a group's root files are its shared layer. `modules.json` is the one source for the tree:
families, modules, local tiers, shared files, areas and their DAG (`dependsOn`). The layer lint
(`node scripts/layers.ts`, in `make lint` and `make test`), the tables below, the package build and two tests read it.

## What compute is

A module or export belongs in compute when it passes all of C1–C5, or C6 alone:

- **C1** Its interface names no model, problem or dataset.
- **C2** At least two areas (or compute modules) use it.
- **C3** It is testable against a reference (NumPy, SciPy, scikit-learn, a textbook value) or a law.
- **C4** It is Tensor-native: inputs are `number | Tensor` or a declared protocol object; outputs are Tensors, numbers
  or protocol objects.
- **C5** It does not change when a note or a figure does.
- **C6** Other compute code depends on it. Foundations such as `zeros`, the `*Like` input types and `Value` are compute
  because the rest of compute is built on them, not because they are capabilities.

Compute is the numerical library: primitives and their derivatives; linear algebra, special functions, randomness and
statistics; the solvers (optimisation, roots, quadrature, ODE and SDE integration, mathematical programming, optimal
transport, interpolation); the generic inference engines (message passing, EP, MCMC, VI, Kalman-type filters); signal
and system operations; general learning parts (kernels, standard losses and metrics, neural-network layers, pipelines
and validation); and every protocol the rest implements. Its module contract is `compute/README.md`.

## What an application is

An application is a named model, problem, environment, dataset or worked example built from compute: k-means, a GMM
fitted by EM, the dishonest casino, a multi-armed bandit, the heat equation. Applications live in 17 areas named after
their subjects. Didactic code that is itself the
lesson lives here too, written to be read, and notes show it rather than copy it. Details: `packages/methods/README.md`.

## Import rules

| From              | May import                                                                                                                                                             | May not                                                                                            |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| compute file      | modules of lower family tiers; modules of its own family in lower local tiers (D5); its ancestors' shared files by relative path; a family index only for shared names | `aifn-methods`; the package root; its own family's index; anything outside aifn; the DOM           |
| compute test      | compute                                                                                                                                                                | `aifn-methods`                                                                                     |
| application file  | any compute node; its ancestors' shared files by relative path; areas below it in the DAG                                                                              | siblings or other nodes of its own area (siblings share through the parent); anything outside aifn |
| render, consumers | compute and applications, by public path                                                                                                                               | defining numerics; relative paths into aifn                                                        |

- No relative import leaves its module except upward to an ancestor's shared file; a parent never imports a child
  (except an `index.ts` re-exporting it).
- Tests of a compute combinator with an application (a pipeline of `standardScaler` and a model) are application tests.

## Promotion and demotion

- **Promote** an application export to compute when C1–C5 hold: its interface names no model, a second area (or a compute
  module) needs it, a reference fixture exists, and its surface is Tensor-native.
- **Demote** a compute export to an application when it names a problem, has consumers in one area only, or exists for
  one figure.
- **How.** Move the file, update `modules.json`,
  move or add the fixture, and rewrite importers in the same change.

## Registries

Every named variant in aifn is a registry entry: `define(info, value)` (`aifn-compute/foundation/registry`) attaches a frozen
`info` to the value itself and returns it, so `adam.info`, `Normal.info` and `metricRegistry.auroc.info` are the same
kind of object. `definer(kind, module)` makes a `define` for one module, with `stability` defaulting to
`experimental`. Each module builds its registry statically from its own namespaces with `entries(kind, …)` (no global
`register()`), usually in a `registry.ts` beside the code, and exports the table from its index.

| Kind                 | Table (compute)                                                 | Kind-specific info                                                                                         |
| -------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `distribution`       | `distributionRegistry`                                          | `params: Space` (argument order), `support`, `discrete`, `eventRank`, `expFamily`                          |
| `kl-rule`            | `klRegistry` (keyed `p\|q`; `kl` dispatches on it)              | `p`, `q` family keys                                                                                       |
| `bijector`           | `bijectorRegistry`                                              | `domain`, `codomain`, `params`, `factory`                                                                  |
| `link`, `likelihood` | `linkRegistry`, `likelihoodRegistry`                            | mean space; support, canonical link, links, dispersion                                                     |
| `kernel`             | `kernelRegistry`                                                | `hyper: Space`, `stationary`                                                                               |
| `window`             | `windowRegistry` (functions of the length)                      | `params`, `mainLobeWidth` (bins), `sideLobeDb`                                                             |
| `wavelet`            | `waveletRegistry`                                               | `family`, `vanishingMoments`, `taps`, `continuous`                                                         |
| `filter-design`      | `filterDesignRegistry`                                          | `family: 'iir' \| 'fir'`, `bands`, `honours`, `params`                                                     |
| `algorithm`          | `<module>Algorithms` in each module (`firstOrderAlgorithms`, …) | `problem`, `state` roles (`iterate`, `objective`, `grad`, `stepSize`, `flags`), `random`                   |
| `function`           | `<module>Functions` in each module (`statsFunctions`, …)        | `role` (transform, estimator, test, construction, property, fit, simulation, solver, inference), `returns` |
| `metric`, `loss`     | `metricRegistry`, `lossRegistry`                                | inputs, direction, range, capability; family, paired metric                                                |
| `primitive`          | the primitive table (`registry.list()` in `foundation/tensor`)  | arity, rule sources                                                                                        |

Applications register models, datasets, modifiers, environments, objectives and log-densities the same way (table in
`packages/methods/README.md`). Every entry carries `key` (its export or lookup name), `kind`, `module`, `name`,
`stability`, and where they apply `notes`, `glossary`, `cite` (keys into the AI Field Notes site's content, carried as given), `summary` and
`random`. An entry's address is `<module>/<key>` (`optim/first-order/adam`); applications may prefix `applied/`.

**Generated tests.** The compute tests load every registry (`test/registries.ts`) and test every
entry without a hand-kept list: every algorithm passes the trace protocol on a case keyed by its address and has the
fields its state roles name (`test/foundation/trace/algorithms.test.ts`, which fails for an algorithm without a case);
every distribution family matches its info, its cdf (Kolmogorov–Smirnov) and its moments
(`test/probability/distributions/families.test.ts`); windows, wavelets, kernels, bijectors, links, likelihood families,
filter designs and KL rules match their declared metadata (`test/foundation/registry/conformance.test.ts`).

## Running

| Command                                                          | What it does                                                                                                                                                  |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `make test`                                                      | the layer lint, then vitest for compute, methods and render                                                                                                   |
| `npx vitest run --config packages/<pkg>/vitest.config.ts <path>` | one package, folder or file (`<pkg>` is `compute`, `methods` or `render`; paths are relative to the package, e.g. `test/numerics/linalg`)                     |
| `make bench`                                                     | compute micro-benchmarks (`packages/compute/bench/compute.bench.ts`): primitive dispatch, kernels, gradients, linear algebra, draws, algorithms on primitives |
| `make fixtures`                                                  | regenerate golden values from Python for both packages (`FIXTURES="numerics/linalg …"` for some)                                                              |
| `make layers`, `make names`                                      | the layer lint and the name-collision lint (both in `make lint`); `node scripts/layers.ts --write` regenerates the tables below                               |
| `make examples`, `make examples-check`                           | the aifn-render examples gallery (`examples/`, port 5192) and its render check                                                                                |

`make check` runs the lints, the type check, the tests and `make examples-check`; `make bench` and `make fixtures` run
on demand.

## Layers (generated)

Families import only strictly lower tiers; modules of a family import only lower local tiers of it.

<!-- aifn-layers:start -->

<!-- Generated from modules.json by `node scripts/layers.ts --write`; do not edit. -->

| Tier | Family      | Modules (local tiers, low to high; * gap)                                                                                                  | Shared             |
| ---- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| 0    | foundation  | contracts, errors · registry · tensor · pytree, fourier · convolution, autodiff, random · space, trace                                     |                    |
| 1    | numerics    | special · linalg · polynomial, quadrature, roots, implicit, geometry, neighbours, robust, factorisation · interpolate                      |                    |
| 2    | graph       | traversal, shortest-paths, spanning-trees, structures, matrices · flows, structured, propagation                                           | graph, tree, heap  |
| 3    | probability | stats, bijectors, samplers · distributions · likelihoods, information, tests, extremes, privacy · markov                                   |                    |
| 3    | optim       | line-search, search · first-order, second-order, proximal, derivative-free, programming, online · minimize                                 | options, schedules |
| 3    | systems     | (one module)                                                                                                                               |                    |
| 4    | inference   | model · exact, message-passing, expectation-propagation, variational, stochastic, filtering · engines                                      |                    |
| 4    | dynamics    | ode, sde · fields, control                                                                                                                 |                    |
| 4    | signal      | windows, sparse · filters, spectral, time-frequency, wavelets, statistical, cepstrum, similarity, image · multirate, audio, decompositions | signal             |
| 4    | transport   | (one module)                                                                                                                               |                    |
| 4    | text        | normalise, tokenise, stem, hyphenation · vocabulary · subword, features, cooccurrence · pipeline, representations · statistics             | aligned            |
| 4    | logic       | terms · resolution, induction                                                                                                              |                    |
| 5    | learning    | estimators, kernels, conformal, subgroups · losses, metrics, compose, validate · calibration, off-policy, explain                          |                    |
| 6    | nn          | functional, init, decoding · layers, quantise · attention, training, experts, graph · sequence                                             |                    |
| 7    | interpreter | (one module)                                                                                                                               |                    |

<!-- aifn-layers:end -->

## Areas (generated)

An area imports compute freely and the areas it depends on (transitively).

<!-- aifn-areas:start -->

<!-- Generated from modules.json by `node scripts/layers.ts --write`; do not edit. -->

| Area         | Nodes (group/{children} [shared]; * gap)                                                                                                                                                                                                                                                                                                                          | Depends on                                                                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| learning     | generalised/{glm, gam, ordinal} [irls, residuals, backfitting, smoothing, registry], linear, generative-classifiers, kernel-methods, gaussian-processes, trees-and-ensembles/{bagging, boosting} [tree, registry], neighbours, reductions, preprocessing, mixture-of-experts, mixture-density, weak-supervision, survival, explanation, transfer [util, isotonic] | unsupervised                                                                                                                              |
| unsupervised | clustering, anomaly, embedding/{linear, manifold, neighbour} [neighbourhoods, centring, util]                                                                                                                                                                                                                                                                     |                                                                                                                                           |
| inference    | sequence-models, topic-models, rating-models, learner-models, lattice-models, mixture-models, conjugate-models, classifier-models                                                                                                                                                                                                                                 |                                                                                                                                           |
| timeseries   | (one module)                                                                                                                                                                                                                                                                                                                                                      |                                                                                                                                           |
| text         | corpora, tokenisers, hyphenation                                                                                                                                                                                                                                                                                                                                  | inference                                                                                                                                 |
| vision       | two-view                                                                                                                                                                                                                                                                                                                                                          |                                                                                                                                           |
| dynamics     | maps, pde, nonlinear, control                                                                                                                                                                                                                                                                                                                                     |                                                                                                                                           |
| gym          | environments/{control} [bandits, gridworlds, registry], agents/{control, policy} [random, bandits, tabular, planning, dqn, registry] [rollout, mdp, train, registry]                                                                                                                                                                                              |                                                                                                                                           |
| generative   | diffusion, gan, energy, autoencoders, flows, boltzmann [densities]                                                                                                                                                                                                                                                                                                | neural                                                                                                                                    |
| neural       | language-models, contrastive, grokking, ode, ode-mixtures, full-batch, quantisation, privacy, graph                                                                                                                                                                                                                                                               |                                                                                                                                           |
| retrieval    | losses, ann, recommenders                                                                                                                                                                                                                                                                                                                                         | evaluation                                                                                                                                |
| evaluation   | text, detection, quality, generative, fairness, beyond-accuracy                                                                                                                                                                                                                                                                                                   | algorithms                                                                                                                                |
| information  | channels, coding, projection                                                                                                                                                                                                                                                                                                                                      |                                                                                                                                           |
| algorithms   | dynamic-programming                                                                                                                                                                                                                                                                                                                                               |                                                                                                                                           |
| theory       | bias-variance, double-descent, capacity, concentration [regression]                                                                                                                                                                                                                                                                                               |                                                                                                                                           |
| data         | synthetic, real/{fonts, hyphenation, ecg} [embedded, real, titanic], objectives, targets, signals [truth, sizes, types, rows, define, recipe]                                                                                                                                                                                                                     | learning, unsupervised, inference, timeseries, vision, dynamics, generative, neural, retrieval, evaluation, information, algorithms, text |
| interpreter  | (one module)                                                                                                                                                                                                                                                                                                                                                      | learning                                                                                                                                  |

<!-- aifn-areas:end -->

## Presentation code lives in render

Presentation helpers live in `aifn-render`
(`packages/render/src/viz/drawing`): `fields.ts` samples direction and slope fields, contours, level sets and nullclines for
drawing (on compute's `aifn-compute/dynamics/fields` grids), and `decimate.ts` holds `lttb` and `minMaxDecimate`.
