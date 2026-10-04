/**
 * `aifn-compute/foundation/contracts`: the shapes and signatures of aifn, each defined once. Types only; nothing here runs, so every
 * module can import it (tier 0). Implementations are checked against these with `satisfies` and type tests
 * (`test/contracts.test.ts`); modules re-export the ones they serve under their established names.
 *
 * - Numbers: `Scalar` (a mathematical value), the integer metadata `Size`, `Index`, `Axis`, `Axes`, `Shape`; `DType`,
 *   `Tensor`, `Traced`, `Value`, `Raw`, `Aval` (shape and dtype without data), and the inputs `TensorLike`, `VectorLike`,
 *   `MatrixLike`, `DataLike`, `NestedArray`.
 * - Randomness: `Key`, `Stream` (plain data), `SampleOptions`.
 * - Iteration: `Algorithm`, `Status`, `StepContext`, `AlgorithmRef`, `Trace`, `TraceOptions`, `TraceMeta`,
 *   `TraceTiming`, `Checkpoints`, `Recorder`, `KeepStates`, `TimingMode`, `TraceWire`.
 * - Probability: `Distribution`, `AnyUnivariate`, `AnyMultivariate` and their typed forms, `Kind`, `Support`,
 *   `ExponentialFamily`, `LogDensity`, `Bijector`, `Interval`.
 * - Decisions: `Environment`, `EnvironmentShape`, `Step`, `Domain`, `DomainKind`, `Agent`, `Decision`, `Transition`,
 *   `TabularModel`, `Outcome`, `EnvironmentModel`, `EnvironmentOracle`, `RenderSpec`, `GridRender`.
 * - Optimisation: `Objective`, `ObjectiveFn`, `Evaluation`, `ValueFunction`, `Hessian`, `Schedule`, `IterateState`,
 *   `StoppingOptions`.
 * - Learning: `Kernel`, `Model` and the capabilities, `Dataset`, `DatasetMeta`, `Recipe`, `Truth`, `Features`;
 *   registry `Info`, `EntryKind`, `Stability` and the per-kind infos (`MetricInfo`, `LossInfo`, `DistributionInfo`,
 *   `BijectorInfo`, `KlRuleInfo`, `LinkInfo`, `LikelihoodInfo`, `KernelInfo`, `WindowInfo`, `WaveletInfo`,
 *   `FilterDesignInfo`, `AlgorithmInfo` with `StateRoles`, `FunctionInfo` with `FunctionRole`, `TestInfo` with
 *   `TestData`, `PrimitiveInfo`, and the applications' kinds).
 * - Structure and signals: `Graph`, `Tree`, `Signal`, `Spectrum`, `TimeFrequency`, `LtiSystem`, `Decomposition`,
 *   `Curve`; parameter spaces `Space`, `Dim`; error kinds.
 * - Function families: `Unary`, `Binary`, `Reduction`, `Sampler`, `KernelFn`, `MetricFn`, `LossFn`.
 *
 * Each protocol has this one definition. Modules re-export the types they serve under the same names, never renamed.
 */

export type * from './algorithm'
export type * from './data'
export type * from './distribution'
export type * from './gym'
export type * from './errors'
export type * from './functions'
export type * from './graph'
export type * from './kernel'
export type * from './kinds'
export type * from './model'
export type * from './numbers'
export type * from './objective'
export type * from './random'
export type * from './registry'
export type * from './results'
export type * from './signal'
export type * from './space'
