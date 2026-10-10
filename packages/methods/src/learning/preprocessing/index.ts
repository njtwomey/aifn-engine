/**
 * `aifn-methods/learning/preprocessing`: transforms fitted on training data and applied to new data, and resamplers
 * for imbalanced classes, as scikit-learn's `preprocessing` and imbalanced-learn.
 *
 * - Scaling each column: `standardScaler` (mean and population standard deviation), `minMaxScaler` (onto a range),
 *   `robustScaler` (median and interquartile range, robust to outliers) and `maxAbsScaler` (keeps zeros).
 * - Encoding categories: `oneHotEncoder` (an indicator per category), `ordinalEncoder` (an index per category) and
 *   `targetEncoder` (a shrunk mean target per category), with `targetEncodeCrossFit` to encode training rows out of
 *   fold.
 * - Missing values and skew: `simpleImputer` (mean, median, mode or a constant) and `powerTransform` (Box–Cox or
 *   Yeo–Johnson with $\lambda$ by maximum likelihood; the scalar transforms `boxCox`, `yeoJohnson`, their inverses and
 *   $\lambda$ searches are re-exported from `aifn-compute/probability/stats`).
 * - Feature maps: `polynomialFeatures` (monomials), `splineFeatures` (a B-spline basis per column) and
 *   `randomFourierFeatures` (cosine features approximating a squared-exponential kernel).
 * - Decorrelation: `whitening`, PCA or ZCA, to an identity sample covariance.
 * - Resampling imbalanced classes: `randomOverSample`, `randomUnderSample`, `smote`, `borderlineSmote` (with
 *   `borderStatus`), `adasyn` (with `adasynWeights`), and `tomekLinks` with `removeTomekLinks` for cleaning.
 * - Helpers: `fitTransform` fits and transforms in one call; `checkColumns` checks a fitted transform's input width.
 *
 * Each transform is an estimator: `fit({ x, y? })` returns a fitted model whose `transform` (and `inverseTransform`,
 * where one exists) applies to any matrix of the fitted number of columns, which `transform` checks. The resamplers are
 * plain functions of a random stream, features and integer labels, returning the new rows with where each came from.
 */

export { checkColumns, fitTransform, type FittedTransform, type Invertible, type Transformer } from './transformer'
export {
  maxAbsScaler,
  minMaxScaler,
  robustScaler,
  standardScaler,
  type AffineScaler,
  type MaxAbsScaler,
  type MinMaxScaler,
  type RobustScaler,
  type StandardScaler,
} from './scaling'
export {
  oneHotEncoder,
  ordinalEncoder,
  targetEncodeCrossFit,
  targetEncoder,
  type CategoricalInput,
  type Category,
  type OneHotEncoder,
  type OrdinalEncoder,
  type TargetEncoder,
} from './encoding'
export { simpleImputer, type SimpleImputer } from './impute'
export {
  polynomialFeatures,
  randomFourierFeatures,
  splineFeatures,
  type PolynomialFeatures,
  type RandomFourierFeatures,
  type SplineFeatures,
} from './features'
export { whitening, type Whitening } from './whitening'
export {
  boxCox,
  boxCoxInverse,
  boxCoxLambda,
  powerTransform,
  yeoJohnson,
  yeoJohnsonInverse,
  yeoJohnsonLambda,
  type PowerLambda,
  type PowerTransform,
} from './power'
export {
  adasyn,
  adasynWeights,
  borderlineSmote,
  borderStatus,
  randomOverSample,
  randomUnderSample,
  removeTomekLinks,
  smote,
  tomekLinks,
  type BorderStatus,
  type Resampled,
  type SmoteOptions,
} from './imbalanced'
export { preprocessingFunctions } from './registry'
