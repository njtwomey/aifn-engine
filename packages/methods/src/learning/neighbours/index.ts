/**
 * `aifn-methods/learning/neighbours`: $k$-nearest-neighbour classification and regression by exhaustive search, as
 * scikit-learn's `KNeighborsClassifier` and `KNeighborsRegressor`.
 *
 * - `kNearestNeighbours`: the class shares among the $k$ nearest training rows, as scores and a predictive.
 * - `kNearestNeighboursRegression`: the mean of their targets, with their spread and expectations of any function.
 *
 * Both weight neighbours equally or by inverse distance, measure distance by any Minkowski order (Euclidean,
 * Manhattan, Chebyshev), break distance ties by training index, and expose each query's `neighbours`.
 */

export {
  kNearestNeighbours,
  kNearestNeighboursRegression,
  type Metric,
  type Neighbours,
  type NeighboursClassifier,
  type NeighboursParams,
  type NeighboursRegressor,
} from './neighbours'
