/**
 * Version spaces of conjunctive hypotheses and Mitchell's candidate-elimination algorithm, one training example per
 * step.
 *
 * Mitchell (1982), "Generalization as search", §4; Mitchell (1997), "Machine Learning", §2.5. The version space is
 * the set of hypotheses consistent with the examples so far, represented by its boundaries: $S$, the maximally specific
 * consistent hypotheses, and $G$, the maximally general ones; a hypothesis is in the space exactly when it lies
 * between them. A positive example removes from $G$ every member that does not cover it and replaces each member of
 * $S$ that does not cover it by its minimal generalisation, kept when some member of $G$ is above it. A negative
 * example removes from $S$ every member that covers it and replaces each member of $G$ that covers it by its minimal
 * specialisations that exclude it and are above some member of $S$. The run ends when the examples are used up or a
 * boundary empties (no conjunction is consistent: noise, or a target outside the language). The boundary sets follow
 * Mitchell exactly except that every empty conjunction is the one `null`. Hypotheses are those of `conjunctions.ts`.
 */
import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import {
  allConjunctions,
  consistent,
  covers,
  generaliseToCover,
  moreGeneralOrEqual,
  mostGeneral,
  sameConjunction,
  specialiseToExclude,
  strictlyMoreGeneral,
  type AttributeInstance,
  type AttributeSpace,
  type Conjunction,
  type LabelledInstance,
} from './conjunctions'

/** What one step of candidate elimination did, told apart by `kind`. */
export type CandidateEliminationEvent =
  | {
      /** The initial state. */
      readonly kind: 'start'
    }
  | {
      /** A positive example was used. */
      readonly kind: 'positive'
      /** The example's index. */
      readonly example: number
      /** Members of $G$ dropped because they do not cover the example. */
      readonly droppedFromG: readonly Conjunction[]
      /** Members of $S$ replaced, with what replaced each (empty when no member of $G$ is above it). */
      readonly generalised: readonly { readonly from: Conjunction; readonly to: readonly Conjunction[] }[]
    }
  | {
      /** A negative example was used. */
      readonly kind: 'negative'
      /** The example's index. */
      readonly example: number
      /** Members of $S$ dropped because they cover the example. */
      readonly droppedFromS: readonly Conjunction[]
      /** Members of $G$ replaced, with the minimal specialisations kept for each. */
      readonly specialised: readonly { readonly from: Conjunction; readonly to: readonly Conjunction[] }[]
    }

/** The state of `candidateElimination`. */
export interface CandidateEliminationState extends Status {
  /** The specific boundary: the maximally specific hypotheses consistent with the examples so far. */
  readonly S: readonly Conjunction[]
  /** The general boundary: the maximally general hypotheses consistent with the examples so far. */
  readonly G: readonly Conjunction[]
  /** Examples used so far. */
  readonly seen: number
  /** True once a boundary is empty: no hypothesis of the language is consistent with the examples. */
  readonly collapsed: boolean
  /** What the step that made this state did. */
  readonly event: CandidateEliminationEvent
  /** True after the last example or once a boundary is empty. */
  terminated: boolean
}

/**
 * The distinct hypotheses of a list, first occurrences kept in order.
 *
 * @param hs The hypotheses.
 * @returns Them without repeats.
 */
const unique = (hs: readonly Conjunction[]): Conjunction[] =>
  hs.filter((h, i) => hs.findIndex((k) => sameConjunction(h, k)) === i)

/**
 * Candidate elimination over conjunctions of the attributes, as a step-through algorithm: one example per step.
 * Step 0 has $S = \{\emptyset\}$ (the empty hypothesis) and $G = \{\langle ?, \dots, ? \rangle\}$. Deterministic:
 * the stream is not used.
 *
 * @param space The attributes and their values.
 * @param examples The labelled instances, in the order they are used.
 * @returns The algorithm; `init` takes no argument, and the state is terminated after the last example or when a
 *   boundary empties.
 *
 * @example Mitchell's EnjoySport after four examples
 * const space = {
 *   names: ['sky', 'air', 'humidity', 'wind', 'water', 'forecast'],
 *   values: [['sunny', 'cloudy', 'rainy'], ['warm', 'cold'], ['normal', 'high'], ['strong', 'weak'],
 *     ['warm', 'cool'], ['same', 'change']],
 * }
 * const examples = [
 *   { x: [0, 0, 0, 0, 0, 0], positive: true },
 *   { x: [0, 0, 1, 0, 0, 0], positive: true },
 *   { x: [2, 1, 1, 0, 0, 1], positive: false },
 *   { x: [0, 0, 1, 0, 1, 1], positive: true },
 * ]
 * const s = run(candidateElimination(space, examples), undefined, 10)
 * print('S =', s.S.map((h) => conjunctionText(h, space)))
 * print('G =', s.G.map((h) => conjunctionText(h, space)))
 * print('members:', versionSpaceMembers(space, s.S, s.G).length)
 */
export function candidateElimination(
  space: AttributeSpace,
  examples: readonly LabelledInstance[],
): Algorithm<void, CandidateEliminationState> {
  return {
    name: 'candidateElimination',
    init: () => ({
      t: 0,
      S: [null],
      G: [mostGeneral(space)],
      seen: 0,
      collapsed: false,
      event: { kind: 'start' },
      terminated: examples.length === 0,
    }),
    step: (state) => {
      if (state.terminated) return { ...state, t: state.t + 1 }
      const k = state.seen
      const { x, positive } = examples[k]
      let S = state.S.slice()
      let G = state.G.slice()
      let event: CandidateEliminationEvent
      if (positive) {
        const droppedFromG = G.filter((g) => !covers(g, x))
        G = G.filter((g) => covers(g, x))
        const generalised: { from: Conjunction; to: Conjunction[] }[] = []
        const next: Conjunction[] = []
        for (const s of S) {
          if (covers(s, x)) {
            next.push(s)
            continue
          }
          const h = generaliseToCover(s, x)
          const to = G.some((g) => moreGeneralOrEqual(g, h)) ? [h] : []
          generalised.push({ from: s, to })
          next.push(...to)
        }
        S = unique(next)
        S = S.filter((s) => !S.some((o) => strictlyMoreGeneral(s, o)))
        event = { kind: 'positive', example: k, droppedFromG, generalised }
      } else {
        const droppedFromS = S.filter((s) => covers(s, x))
        S = S.filter((s) => !covers(s, x))
        const specialised: { from: Conjunction; to: Conjunction[] }[] = []
        const next: Conjunction[] = []
        for (const g of G) {
          if (!covers(g, x)) {
            next.push(g)
            continue
          }
          const to = specialiseToExclude(g, x, space).filter((h) => S.some((s) => moreGeneralOrEqual(h, s)))
          specialised.push({ from: g, to })
          next.push(...to)
        }
        G = unique(next)
        G = G.filter((g) => !G.some((o) => strictlyMoreGeneral(o, g)))
        event = { kind: 'negative', example: k, droppedFromS, specialised }
      }
      const collapsed = S.length === 0 || G.length === 0
      return {
        t: state.t + 1,
        S,
        G,
        seen: k + 1,
        collapsed,
        event,
        terminated: collapsed || k + 1 >= examples.length,
      }
    },
  }
}

/**
 * Find-S: the most specific conjunction that covers every positive example, by generalising from $\emptyset$ one
 * positive at a time. It ignores the negatives, so it may cover some (check with `consistent`).
 *
 * @param examples The labelled examples; the positives are used in order.
 * @returns The hypothesis (the empty one when there is no positive example).
 *
 * @example Find-S on three examples
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * const ex = [{ x: [0, 0], positive: true }, { x: [1, 1], positive: false }, { x: [0, 1], positive: true }]
 * print(conjunctionText(findS(ex), space))
 */
export function findS(examples: readonly LabelledInstance[]): Conjunction {
  let h: Conjunction = null
  for (const e of examples) if (e.positive) h = generaliseToCover(h, e.x)
  return h
}

/**
 * Every hypothesis of the language between the boundaries: above some member of $S$ and below some member of $G$.
 *
 * @param space The attributes.
 * @param S The specific boundary.
 * @param G The general boundary.
 * @returns The members of the version space, in the order of `allConjunctions`.
 *
 * @example The space before any example is the whole language
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * print(versionSpaceMembers(space, [null], [mostGeneral(space)]).length)
 */
export function versionSpaceMembers(
  space: AttributeSpace,
  S: readonly Conjunction[],
  G: readonly Conjunction[],
): Conjunction[] {
  return allConjunctions(space).filter(
    (h) => S.some((s) => moreGeneralOrEqual(h, s)) && G.some((g) => moreGeneralOrEqual(g, h)),
  )
}

/**
 * Every hypothesis of the language consistent with the examples, by checking each one: the version space by brute
 * force, for checking the boundaries on small spaces.
 *
 * @param space The attributes.
 * @param examples The labelled examples.
 * @returns The consistent conjunctions, in the order of `allConjunctions`.
 *
 * @example The consistent hypotheses of two examples
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * const ex = [{ x: [0, 0], positive: true }, { x: [1, 1], positive: false }]
 * print(consistentSet(space, ex).map((h) => conjunctionText(h, space)))
 */
export function consistentSet(space: AttributeSpace, examples: readonly LabelledInstance[]): Conjunction[] {
  return allConjunctions(space).filter((h) => consistent(h, examples))
}

/**
 * How the version space classifies an instance: the number of its members that cover it and that do not, and the
 * verdict. Every member says positive exactly when every member of $S$ covers the instance; every member says negative
 * exactly when no member of $G$ covers it; otherwise the members disagree.
 *
 * @param members The members of the version space, as `versionSpaceMembers` returns them.
 * @param x The instance to classify.
 * @returns The counts of members voting positive and negative, and the verdict (`empty` when there are no members).
 *
 * @example A unanimous and a split vote
 * const space = { names: ['sky', 'wind'], values: [['sunny', 'rainy'], ['weak', 'strong']] }
 * const ex = [{ x: [0, 0], positive: true }, { x: [1, 1], positive: false }]
 * const members = consistentSet(space, ex)
 * print(versionSpaceVote(members, [0, 0]), versionSpaceVote(members, [0, 1]))
 */
export function versionSpaceVote(
  members: readonly Conjunction[],
  x: AttributeInstance,
): { positive: number; negative: number; verdict: 'positive' | 'negative' | 'disagree' | 'empty' } {
  const positive = members.filter((h) => covers(h, x)).length
  const negative = members.length - positive
  const verdict =
    members.length === 0 ? 'empty' : negative === 0 ? 'positive' : positive === 0 ? 'negative' : 'disagree'
  return { positive, negative, verdict }
}
