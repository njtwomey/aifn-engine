/**
 * `aifn-methods/neural/tuning`: tuning a network's training hyperparameters, as studies that run in a worker.
 *
 * - `learningRateSweep`: a small ReLU network on the two-moons task, trained once per learning rate and seed by Adam
 *   or SGD with momentum for a fixed number of steps; the validation cross-entropy of every run, with the runs that
 *   diverged flagged (`LearningRateSweepOptions`, `LearningRateSweep`).
 *
 * Inputs and results are plain numbers and arrays, and the streams are named from a seed string, so a page gets the
 * same numbers in a worker as on its own thread. `tuningFunctions` registers the functions.
 */

export { learningRateSweep, type LearningRateSweep, type LearningRateSweepOptions } from './learning-rate-sweep'
export { tuningFunctions } from './registry'
