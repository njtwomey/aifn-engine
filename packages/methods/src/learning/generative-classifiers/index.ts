/**
 * `aifn-methods/learning/generative-classifiers`: classifiers that model each class's density and apply Bayes' rule,
 * as scikit-learn's `naive_bayes` and `discriminant_analysis`.
 *
 * - Naive Bayes, features independent given the class: `gaussianNaiveBayes` (real features), `multinomialNaiveBayes`
 *   (counts, such as words) and `bernoulliNaiveBayes` (presence or absence, where an absent feature is evidence too).
 * - Discriminant analysis, Gaussian classes: `linearDiscriminant` (one shared covariance, so linear boundaries; also
 *   projects onto at most $K - 1$ discriminant directions) and `quadraticDiscriminant` (a covariance per class, so
 *   quadric boundaries).
 *
 * Every model's `forward` is the $m \times K$ joint log-likelihoods $\log p(\xvec, y = k)$ up to a constant shared by
 * the classes, `predictive` the class law, and `decide` the most probable class. Labels are $0, \dots, K - 1$, and
 * the priors are the training frequencies unless `priors` are given.
 */

export {
  bernoulliNaiveBayes,
  gaussianNaiveBayes,
  linearDiscriminant,
  multinomialNaiveBayes,
  quadraticDiscriminant,
  type DiscreteNaiveBayesModel,
  type DiscriminantModel,
  type GaussianNaiveBayesModel,
  type GenerativeClassifier,
  type LinearDiscriminantModel,
} from './bayes'
