/**
 * Simulated learners for equitable ability estimation (Twomey, McMullan, Elhalal, Poyiadzi and Vaquero, 2022, "Equitable
 * Ability Estimation in Neurodivergent Student Populations with Zero-Inflated Learner Models", Table 1). Students have
 * an ability θ ~ N(0, 1) drawn independently of their neurodivergent conditions (NDCs: dyslexia, dyscalculia and sensory
 * processing disorder, SPD, each an independent Bernoulli draw). Items have a difficulty, discrimination and guessing
 * floor, a subject, a content type, an information density and a delivery and response type (DRT).
 *
 * A response is a structural zero with probability π, the zero-inflation probability (one minus the learning quality
 * factor, LQF); otherwise it follows the three-parameter IRT model p = c + (1 − c) σ(a(θ − b)). So
 * Pr(Y = 0) = π + (1 − π)(1 − p) and Pr(Y = 1) = (1 − π)p, the paper's Eqn (1). π depends on the student's conditions and
 * the item's DRT: condition k has an unsuitability u_k(item) ∈ [0, 1], linear in the item's features, and
 *
 *   logit π = logit π₀ + Σ_k z_k u_k(item) (logit r_k − logit π₀),
 *
 * with z_k ∈ {0, 1} the student's conditions, π₀ the rate for any student on a suitable item and r_k the rate of a
 * student with condition k on a fully unsuitable item. The paper's own weight vectors are in its implementation; the
 * unsuitabilities here follow its descriptions: dyslexia is affected by reading (delivery and response) and letters,
 * dyscalculia by digits, SPD by combined reading and listening and by dense, mixed content.
 */

import type { FunctionInfo } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'

/** The neurodivergent conditions, in column order of `conditions`. */
export const LEARNER_CONDITIONS = ['dyslexia', 'dyscalculia', 'spd'] as const
export type LearnerCondition = (typeof LEARNER_CONDITIONS)[number]

/** The item features, in column order of `itemFeatures`: one-hot delivery, response and content type, then density. */
export const LEARNER_ITEM_FEATURES = [
  'read',
  'listen',
  'read-and-listen',
  'written',
  'speak',
  'click-picture',
  'click-read',
  'letters',
  'digits',
  'letters-and-digits',
  'density',
  'maths',
] as const

const F = LEARNER_ITEM_FEATURES.length
const K = LEARNER_CONDITIONS.length

/**
 * Unsuitability weights w_k (one row per condition, one column per item feature): u_k(item) = w_k · x(item). Each row's
 * largest attainable sum is 1 (an item that is unsuitable in every respect).
 */
export const LEARNER_UNSUITABILITY: readonly (readonly number[])[] = [
  // read, listen, both | written, speak, picture, click-read | letters, digits, both | density | maths
  [0.5, 0, 0.25, 0.25, 0, 0, 0.25, 0.15, 0, 0.1, 0.1, 0],
  [0.15, 0, 0.1, 0.15, 0, 0, 0.1, 0, 0.7, 0.4, 0, 0],
  [0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.25, 0.25, 0],
]

/** Options of `learnerResponses`. Defaults are the paper's Table 1. */
export interface LearnerResponsesOptions {
  /** Students (default 300). */
  students?: number
  /** Items in the bank (default 60). */
  items?: number
  /** Items each student attempts, drawn without replacement from the bank (default 20; the rest are NaN). */
  attempts?: number
  /** Prevalence of each condition (dyslexia 0.1, dyscalculia 0.06, SPD 0.11). */
  prevalence?: Partial<Record<LearnerCondition, number>>
  /** r_k: the zero-inflation rate of a student with condition k on a fully unsuitable item (default 0.6 each). */
  inflation?: Partial<Record<LearnerCondition, number>>
  /** π₀: the zero-inflation rate on a suitable item, for every student (default 0.02). */
  baseRate?: number
  /** Difficulties are uniform on [−spread, spread] (default 2). */
  difficultySpread?: number
  /** Discriminations are uniform on [low, high] (default [0.5, 4]). */
  discrimination?: readonly [number, number]
  /** Guessing floors are uniform on [0, guessing] (default 0.15; 0 gives the 2PL model). */
  guessing?: number
}

/** Simulated learners, items and responses, with every generating quantity as truth. */
export interface LearnerResponses {
  /** Responses [students, items]: 1 correct, 0 a zero (incorrect or not answered), NaN not attempted. */
  responses: Tensor
  /** Conditions [students, K] of 0 and 1 (columns `LEARNER_CONDITIONS`). */
  conditions: Tensor
  /** Item features [items, F] (columns `LEARNER_ITEM_FEATURES`). */
  itemFeatures: Tensor
  /** True abilities θ, one per student. */
  ability: Float64Array
  /** True difficulty b, discrimination a and guessing floor c, one per item. */
  difficulty: Float64Array
  discrimination: Float64Array
  guessing: Float64Array
  /** True zero-inflation probabilities π [students × items] (row-major) and which zeros were structural (1). */
  pi: Float64Array
  structural: Uint8Array
  /** Unsuitability u_k(item) [items × K]. */
  unsuitability: Float64Array
  /** Each student's number of conditions, and their group: 0 none, 1 + k for condition k alone, 4 for two or more. */
  conditionCount: Int32Array
  group: Int32Array
  conditionNames: readonly string[]
  featureNames: readonly string[]
  groupNames: readonly string[]
}

const logit = (p: number) => Math.log(p / (1 - p))
const clampRate = (p: number) => Math.min(1 - 1e-6, Math.max(1e-6, p))
/** An index drawn from unnormalised weights by one uniform. */
const pick = (u: number, weights: readonly number[]) => {
  const total = weights.reduce((a, b) => a + b, 0)
  let v = u * total
  for (let j = 0; j < weights.length; j++) {
    v -= weights[j]
    if (v < 0) return j
  }
  return weights.length - 1
}

/**
 * Students answering items under the paper's simulation (Table 1), with zero inflation driven by the fit between each
 * student's conditions and each item's delivery and response type. Deterministic in the stream.
 */
export function learnerResponses(s: Stream, options: LearnerResponsesOptions = {}): LearnerResponses {
  const { students = 300, items = 60, baseRate = 0.02, difficultySpread = 2, guessing = 0.15 } = options
  const attempts = Math.min(options.attempts ?? 20, items)
  const [aLow, aHigh] = options.discrimination ?? [0.5, 4]
  if (students < 1 || items < 1 || attempts < 1)
    throw new DomainError('learnerResponses', 'learnerResponses: students, items and attempts must be positive')
  const prevalence = { dyslexia: 0.1, dyscalculia: 0.06, spd: 0.11, ...options.prevalence }
  const inflation = { dyslexia: 0.6, dyscalculia: 0.6, spd: 0.6, ...options.inflation }
  // Students: ability independent of the conditions (the paper's assumption).
  const ability = standardNormals(child(s, 'ability'), students)
  const uc = units(child(s, 'conditions'), students * K)
  const z = new Float64Array(students * K)
  const conditionCount = new Int32Array(students)
  const group = new Int32Array(students)
  for (let p = 0; p < students; p++) {
    let last = -1
    LEARNER_CONDITIONS.forEach((name, k) => {
      if (uc[p * K + k] < prevalence[name]) {
        z[p * K + k] = 1
        conditionCount[p]++
        last = k
      }
    })
    group[p] = conditionCount[p] === 0 ? 0 : conditionCount[p] === 1 ? 1 + last : 4
  }
  // Items: IRT parameters and DRT features.
  const ui = units(child(s, 'items'), items * 9)
  const density = standardNormals(child(s, 'density'), items)
  const difficulty = new Float64Array(items)
  const discrimination = new Float64Array(items)
  const guess = new Float64Array(items)
  const x = new Float64Array(items * F)
  for (let i = 0; i < items; i++) {
    const u = ui.subarray(i * 9, i * 9 + 9)
    difficulty[i] = difficultySpread * (2 * u[0] - 1)
    discrimination[i] = aLow + (aHigh - aLow) * u[1]
    guess[i] = guessing * u[2]
    const maths = u[3] < 0.5
    // Content: English items are letters; Maths items letters, digits or both with weights 0.1, 0.5, 0.6.
    const content = maths ? pick(u[4], [0.1, 0.5, 0.6]) : 0
    const delivery = pick(u[5], [0.3, 0.3, 0.4])
    const response = pick(u[6], [0.4, 0.2, 0.2, 0.2])
    const row = x.subarray(i * F, (i + 1) * F)
    row[delivery] = 1
    row[3 + response] = 1
    row[7 + content] = 1
    row[10] = Math.min(1, Math.max(0.1, 0.35 + 0.15 * density[i]))
    row[11] = maths ? 1 : 0
  }
  const unsuitability = new Float64Array(items * K)
  for (let i = 0; i < items; i++)
    for (let k = 0; k < K; k++) {
      let v = 0
      for (let f = 0; f < F; f++) v += LEARNER_UNSUITABILITY[k][f] * x[i * F + f]
      unsuitability[i * K + k] = v
    }
  // Responses: each student attempts `attempts` items, by a partial Fisher–Yates shuffle of the bank.
  const l0 = logit(clampRate(baseRate))
  const lift = LEARNER_CONDITIONS.map((name) => logit(clampRate(inflation[name])) - l0)
  const ua = units(child(s, 'attempts'), students * attempts)
  const ur = units(child(s, 'responses'), students * items * 2)
  const y = new Float64Array(students * items).fill(NaN)
  const pi = new Float64Array(students * items)
  const structural = new Uint8Array(students * items)
  const pool = Array.from({ length: items }, (_, i) => i)
  for (let p = 0; p < students; p++) {
    for (let i = 0; i < items; i++) {
      let eta = l0
      for (let k = 0; k < K; k++) eta += z[p * K + k] * unsuitability[i * K + k] * lift[k]
      pi[p * items + i] = 1 / (1 + Math.exp(-eta))
    }
    for (let m = 0; m < items; m++) pool[m] = m
    for (let m = 0; m < attempts; m++) {
      const j = m + Math.floor(ua[p * attempts + m] * (items - m))
      ;[pool[m], pool[j]] = [pool[j], pool[m]]
      const i = pool[m]
      const at = p * items + i
      if (ur[2 * at] < pi[at]) {
        structural[at] = 1
        y[at] = 0
      } else {
        const q = guess[i] + (1 - guess[i]) / (1 + Math.exp(-discrimination[i] * (ability[p] - difficulty[i])))
        y[at] = ur[2 * at + 1] < q ? 1 : 0
      }
    }
  }
  return {
    responses: fromData(y, [students, items]),
    conditions: fromData(z, [students, K]),
    itemFeatures: fromData(x, [items, F]),
    ability,
    difficulty,
    discrimination,
    guessing: guess,
    pi,
    structural,
    unsuitability,
    conditionCount,
    group,
    conditionNames: LEARNER_CONDITIONS,
    featureNames: LEARNER_ITEM_FEATURES,
    groupNames: ['no condition', 'dyslexia', 'dyscalculia', 'SPD', 'two or more'],
  }
}

const fn = definer<FunctionInfo>('function', 'data/synthetic')

fn(
  {
    key: 'learnerResponses',
    name: 'Neurodivergent learner responses',
    summary:
      'Students with abilities and conditions (dyslexia, dyscalculia, SPD) answering items whose delivery and response type can cause structural zeros.',
    role: 'simulation',
    random: true,
    notes: ['item-response-theory', 'group-fairness-metrics'],
    cite: ['lord1968', 'barton1981'],
  },
  learnerResponses,
)
