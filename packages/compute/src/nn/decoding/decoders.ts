/**
 * Decoding a language model: turning next-token distributions into a sequence, one token per step, over any
 * `logits(prefix)` function. Greedy decoding takes the most probable token; sampling draws from the processed
 * distribution (temperature, top-k, top-p, repetition penalty) with the step's stream; beam search keeps the B best
 * partial sequences by log-probability with a length penalty (Wu et al., 2016); speculative decoding drafts several
 * tokens with a cheap model and verifies them with the target in one pass, accepting each with probability
 * min(1, p/q) and resampling the first rejection from the normalised residual max(0, p − q), which leaves the output
 * distributed exactly as sampling from the target (Leviathan, Kalman and Matias, 2023; Chen et al., 2023).
 *
 * Every decoder is a step-through `Algorithm`: states are plain data, randomness comes from the runner's per-step
 * stream, so a decoding can be scrubbed, replayed and compared.
 */

import type { Algorithm, Size, Status, StepContext } from 'aifn-compute/foundation/contracts'
import { categorical, child, uniform } from 'aifn-compute/foundation/random'
import { logitsOf, nextTokenDistribution, softmaxOf, type LogitsFn, type SamplingOptions } from './processors'

/** What every decoder takes. */
export type DecodingOptions = {
  /** The prompt's token ids (may be empty). */
  prompt: readonly number[]
  /** Tokens to generate at most. */
  maxTokens: Size
  /** Tokens that end the sequence when generated (kept in the output). */
  stop?: readonly number[]
}

/** The state of greedy and sampling decoders. */
export interface DecodingState extends Status {
  /** Prompt and generated tokens. */
  readonly tokens: readonly number[]
  /** Generated tokens so far. */
  readonly generated: Size
  /** The token chosen by the last step (−1 at t = 0). */
  readonly token: number
  /** The distribution the last token was chosen from, before processing (softmax of the logits). */
  readonly probs: readonly number[]
  /** The processed distribution it was drawn from. */
  readonly filtered: readonly number[]
  /** Tokens in the processed support. */
  readonly kept: readonly boolean[]
  /** Σ log p(token) of the generated tokens under the raw model distribution. */
  readonly logProb: number
  readonly terminated: boolean
}

function start(options: DecodingOptions): DecodingState {
  return {
    t: 0,
    tokens: [...options.prompt],
    generated: 0,
    token: -1,
    probs: [],
    filtered: [],
    kept: [],
    logProb: 0,
    terminated: options.maxTokens <= 0,
  }
}

function advance(
  state: DecodingState,
  options: DecodingOptions,
  token: number,
  dist: ReturnType<typeof nextTokenDistribution>,
): DecodingState {
  const generated = state.generated + 1
  return {
    t: state.t + 1,
    tokens: [...state.tokens, token],
    generated,
    token,
    probs: dist.probs,
    filtered: dist.filtered,
    kept: dist.kept,
    logProb: state.logProb + Math.log(dist.probs[token]),
    terminated: generated >= options.maxTokens || (options.stop?.includes(token) ?? false),
  }
}

/** The index of the largest entry (the first among ties). */
const argmaxOf = (v: readonly number[]) => v.reduce((best, x, i) => (x > v[best] ? i : best), 0)

/** Greedy decoding: each step appends the most probable next token. Deterministic. */
export function greedyDecoding(logits: LogitsFn, options: DecodingOptions): Algorithm<void, DecodingState> {
  return {
    name: 'greedy-decoding',
    init: () => start(options),
    step: (state) => {
      const dist = nextTokenDistribution(logitsOf(logits(state.tokens)), state.tokens)
      return advance(state, options, argmaxOf(dist.probs), dist)
    },
  }
}

/**
 * Ancestral sampling: each step draws the next token from the processed distribution (`SamplingOptions`: repetition
 * penalty, temperature, top-k, top-p) with the step's stream.
 */
export function samplingDecoding(
  logits: LogitsFn,
  options: DecodingOptions & SamplingOptions,
): Algorithm<void, DecodingState> {
  return {
    name: 'sampling-decoding',
    init: () => start(options),
    step: (state, ctx: StepContext) => {
      const dist = nextTokenDistribution(logitsOf(logits(state.tokens)), state.tokens, options)
      return advance(state, options, categorical(ctx.stream, dist.filtered), dist)
    },
  }
}

// ── Beam search ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** A node of the beam-search tree: one token appended to its parent's hypothesis. */
export type BeamNode = {
  readonly id: number
  /** The parent node (−1 for the root, the prompt). */
  readonly parent: number
  readonly token: number
  /** The step at which it was added (0 for the root). */
  readonly step: Size
  /** Σ log p of the hypothesis's generated tokens. */
  readonly logProb: number
  /** Kept in the beam after its step (else pruned). */
  readonly kept: boolean
  /** It ended with a stop token. */
  readonly finished: boolean
}

/** A hypothesis: its tree node, tokens (prompt included), log-probability and length-penalised score. */
export type Hypothesis = {
  readonly node: number
  readonly tokens: readonly number[]
  readonly logProb: number
  readonly score: number
}

/** The state of `beamSearch`. */
export interface BeamState extends Status {
  /** The live hypotheses, best first. */
  readonly beams: readonly Hypothesis[]
  /** Hypotheses that ended with a stop token, best first. */
  readonly finished: readonly Hypothesis[]
  /** Every node expanded so far (a tree rooted at node 0), for drawing. */
  readonly tree: readonly BeamNode[]
  /** The best hypothesis so far (finished or live), by score. */
  readonly best: Hypothesis
  readonly terminated: boolean
}

/** Options of `beamSearch`. */
export type BeamOptions = DecodingOptions & {
  /** Beam width B. */
  beams: Size
  /** Length penalty α: scores are log p / ((5 + |Y|)/6)^α (Wu et al., 2016, eq. 14); 0 for none. Default 0. */
  lengthPenalty?: number
  /** Candidate tokens considered per live beam (default B). */
  expand?: Size
}

/** GNMT's length penalty ((5 + n)/6)^α of a hypothesis of n generated tokens. */
export function lengthPenalty(n: Size, alpha: number): number {
  return ((5 + n) / 6) ** alpha
}

/**
 * Beam search: keep the B best partial sequences; each step extends every live one by its most probable tokens, and
 * keeps the B best extensions by score = Σ log p / lp(|Y|) with GNMT's length penalty lp. A hypothesis ending in a
 * stop token is set aside; the search ends when `maxTokens` are generated or no live hypothesis can beat the best
 * finished one. `tree` records every expansion, kept or pruned, so the search can be drawn. Deterministic.
 */
export function beamSearch(logits: LogitsFn, options: BeamOptions): Algorithm<void, BeamState> {
  const { beams: width, lengthPenalty: alpha = 0, maxTokens } = options
  const expand = options.expand ?? width
  const prompt = [...options.prompt]
  const scoreOf = (logProb: number, n: Size) => logProb / lengthPenalty(n, alpha)
  const root: Hypothesis = { node: 0, tokens: prompt, logProb: 0, score: 0 }
  return {
    name: 'beam-search',
    init: () => ({
      t: 0,
      beams: [root],
      finished: [],
      tree: [{ id: 0, parent: -1, token: -1, step: 0, logProb: 0, kept: true, finished: false }],
      best: root,
      terminated: maxTokens <= 0,
    }),
    step: (state) => {
      const step = state.t + 1
      const tree = [...state.tree]
      const candidates: { hyp: Hypothesis; node: BeamNode }[] = []
      for (const b of state.beams) {
        const probs = softmaxOf(logitsOf(logits(b.tokens)))
        const order = probs
          .map((_, i) => i)
          .sort((x, y) => probs[y] - probs[x])
          .slice(0, expand)
        for (const token of order) {
          const logProb = b.logProb + Math.log(probs[token])
          const id = tree.length + candidates.length
          const finished = options.stop?.includes(token) ?? false
          candidates.push({
            hyp: { node: id, tokens: [...b.tokens, token], logProb, score: scoreOf(logProb, step) },
            node: { id, parent: b.node, token, step, logProb, kept: false, finished },
          })
        }
      }
      candidates.sort((x, y) => y.hyp.score - x.hyp.score)
      const finished = [...state.finished]
      const live: Hypothesis[] = []
      const keptIds = new Set<number>()
      for (const c of candidates) {
        if (live.length >= width) break
        keptIds.add(c.node.id)
        if (c.node.finished) finished.push(c.hyp)
        else live.push(c.hyp)
      }
      for (const c of candidates) tree.push({ ...c.node, kept: keptIds.has(c.node.id) })
      finished.sort((x, y) => y.score - x.score)
      const pool = [...finished, ...live].sort((x, y) => y.score - x.score)
      const best = pool[0] ?? state.best
      // Log-probabilities only fall, so with α = 0 a live beam below the best finished score cannot overtake it.
      const hopeless = finished.length > 0 && alpha === 0 && live.every((h) => h.score < finished[0].score)
      return {
        t: step,
        beams: live,
        finished,
        tree,
        best,
        terminated: step >= maxTokens || live.length === 0 || hopeless,
      }
    },
  }
}

// ── Speculative decoding ─────────────────────────────────────────────────────────────────────────────────────────────

/** One round of speculative decoding: what was drafted, what the target kept, and why. */
export type SpeculativeRound = {
  /** The γ draft tokens. */
  readonly drafted: readonly number[]
  /** min(1, p/q) of each draft token. */
  readonly acceptance: readonly number[]
  /** How many draft tokens were accepted (a prefix of `drafted`). */
  readonly accepted: Size
  /** The token appended after them: the resampled rejection or the bonus token. */
  readonly correction: number
  /** True when the correction came from the residual max(0, p − q) (a rejection), false for the bonus token. */
  readonly rejected: boolean
}

/** The state of `speculativeDecoding`. */
export interface SpeculativeState extends Status {
  readonly tokens: readonly number[]
  readonly generated: Size
  /** The last round (null at t = 0). */
  readonly round: SpeculativeRound | null
  /** Draft tokens proposed and accepted so far; their ratio is the acceptance rate α̂. */
  readonly proposed: Size
  readonly acceptedTotal: Size
  /** Calls of the target model (one verification pass per round) and of the draft model. */
  readonly targetCalls: Size
  readonly draftCalls: Size
  readonly terminated: boolean
}

/** Options of `speculativeDecoding`. */
export type SpeculativeOptions = DecodingOptions & {
  /** Draft tokens per round γ (default 4). */
  lookahead?: Size
  /** Sampling temperature applied to both models (default 1). */
  temperature?: number
}

/**
 * Speculative sampling (Leviathan, Kalman and Matias, 2023; Chen et al., 2023): each step is one round. The draft model
 * q proposes γ tokens by sampling; the target p scores all γ + 1 prefixes (one batched pass in practice, counted as
 * one target call); draft token x_i is accepted with probability min(1, p(x_i)/q(x_i)); at the first rejection a token
 * is drawn from norm(max(0, p − q)) and the round ends, and if all γ are accepted a bonus token is drawn from p. The
 * output has exactly the target's sampling distribution; the speed-up comes from accepting several tokens per target
 * call, at the expected rate (1 − α^{γ+1})/(1 − α) tokens per call for acceptance rate α.
 */
export function speculativeDecoding(
  target: LogitsFn,
  draft: LogitsFn,
  options: SpeculativeOptions,
): Algorithm<void, SpeculativeState> {
  const { lookahead = 4, temperature = 1, maxTokens } = options
  const dist = (f: LogitsFn, prefix: readonly number[]) =>
    nextTokenDistribution(logitsOf(f(prefix)), prefix, { temperature }).filtered
  const stops = (tok: number) => options.stop?.includes(tok) ?? false
  return {
    name: 'speculative-decoding',
    init: () => ({
      t: 0,
      tokens: [...options.prompt],
      generated: 0,
      round: null,
      proposed: 0,
      acceptedTotal: 0,
      targetCalls: 0,
      draftCalls: 0,
      terminated: maxTokens <= 0,
    }),
    step: (state, ctx: StepContext) => {
      const room = maxTokens - state.generated
      const gamma = Math.max(0, Math.min(lookahead, room - 1))
      const drafted: number[] = []
      const qs: number[][] = []
      let prefix = [...state.tokens]
      for (let i = 0; i < gamma; i++) {
        const q = dist(draft, prefix)
        const tok = categorical(child(ctx.stream, 'draft', i), q)
        qs.push(q)
        drafted.push(tok)
        prefix = [...prefix, tok]
        if (stops(tok)) break
      }
      const ps = Array.from({ length: drafted.length + 1 }, (_, i) =>
        dist(target, [...state.tokens, ...drafted.slice(0, i)]),
      )
      const acceptance = drafted.map((tok, i) => Math.min(1, ps[i][tok] / qs[i][tok]))
      let accepted = 0
      while (accepted < drafted.length && uniform(child(ctx.stream, 'accept', accepted)) < acceptance[accepted])
        accepted++
      let correction: number
      let rejected = false
      if (accepted < drafted.length) {
        const residual = ps[accepted].map((p, j) => Math.max(0, p - qs[accepted][j]))
        const z = residual.reduce((a, b) => a + b, 0)
        correction = categorical(child(ctx.stream, 'residual'), z > 0 ? residual : ps[accepted])
        rejected = true
      } else correction = categorical(child(ctx.stream, 'bonus'), ps[drafted.length])
      let appended = [...drafted.slice(0, accepted), correction]
      const stopAt = appended.findIndex(stops)
      if (stopAt >= 0) appended = appended.slice(0, stopAt + 1)
      const generated = state.generated + appended.length
      return {
        t: state.t + 1,
        tokens: [...state.tokens, ...appended],
        generated,
        round: { drafted, acceptance, accepted, correction, rejected },
        proposed: state.proposed + drafted.length,
        acceptedTotal: state.acceptedTotal + accepted,
        targetCalls: state.targetCalls + 1,
        draftCalls: state.draftCalls + drafted.length,
        terminated: generated >= maxTokens || stopAt >= 0,
      }
    },
  }
}

/** The expected tokens per target call of speculative decoding with acceptance rate α and lookahead γ. */
export function expectedTokensPerCall(alpha: number, lookahead: Size): number {
  return alpha === 1 ? lookahead + 1 : (1 - alpha ** (lookahead + 1)) / (1 - alpha)
}
