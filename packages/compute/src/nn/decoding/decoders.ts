/**
 * Decoding a language model: turning next-token distributions into a sequence, one token per step, over any
 * `logits(prefix)` function. Greedy decoding takes the most probable token; sampling draws from the processed
 * distribution (temperature, top-k, top-p, repetition penalty) with the step's stream; beam search keeps the $B$ best
 * partial sequences by log-probability with a length penalty (Wu et al., 2016); speculative decoding drafts several
 * tokens with a cheap model $q$ and verifies them with the target $p$ in one pass, accepting each with probability
 * $\min(1, p/q)$ and resampling the first rejection from the normalised residual $\max(0, p - q)$, which leaves the
 * output distributed exactly as sampling from the target (Leviathan, Kalman and Matias, 2023; Chen et al., 2023).
 *
 * Every decoder is a step-through `Algorithm`: states are plain data, randomness comes from the runner's per-step
 * stream, so a decoding can be scrubbed, replayed and compared. Run one with `run(alg, undefined, steps)`, passing
 * `{ stream }` for the sampling decoders; each stops by itself (`terminated`) at `maxTokens` or a stop token.
 */

import type { Algorithm, Size, Status, StepContext } from 'aifn-compute/foundation/contracts'
import { categorical, child, uniform } from 'aifn-compute/foundation/random'
import { logitsOf, nextTokenDistribution, softmaxOf, type LogitsFn, type SamplingOptions } from './processors'

/** What every decoder takes: the prompt, the length limit and the stop tokens. */
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
  /** The token chosen by the last step ($-1$ at $t = 0$). */
  readonly token: number
  /** The distribution the last token was chosen from, before processing (softmax of the logits). */
  readonly probs: readonly number[]
  /** The processed distribution it was drawn from. */
  readonly filtered: readonly number[]
  /** Tokens in the processed support. */
  readonly kept: readonly boolean[]
  /** $\sum \log p(x_t)$ over the generated tokens $x_t$, under the raw model distribution. */
  readonly logProb: number
  /** True once `maxTokens` tokens are generated or a stop token is (at once when `maxTokens` is 0). */
  readonly terminated: boolean
}

/**
 * The state before any token is generated: the prompt alone.
 *
 * @param options The prompt and `maxTokens` (a limit of 0 or less terminates at once).
 * @returns The initial state.
 */
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

/**
 * The state after appending one token: its log-probability is added and the stopping rules are checked.
 *
 * @param state The state before the token.
 * @param options The length limit and the stop tokens.
 * @param token The token appended.
 * @param dist The distribution it was chosen from, raw and processed; its raw probability gives the log-probability.
 * @returns The next state.
 */
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

/**
 * The index of the largest entry (the first among ties).
 *
 * @param v The values, e.g. probabilities.
 * @returns The index of the first largest value.
 */
const argmaxOf = (v: readonly number[]) => v.reduce((best, x, i) => (x > v[best] ? i : best), 0)

/**
 * Greedy decoding: each step appends the most probable next token (the first among ties), with no processors.
 * Deterministic. Greedy is not the most probable sequence: a likely first token can lead to unlikely ones (see
 * `beamSearch`).
 *
 * @param logits The model: the next-token logits after a prefix.
 * @param options The prompt, `maxTokens` and stop tokens.
 * @returns The algorithm, one token per step.
 *
 * @example A bigram model of four tokens; token 3 stops
 * const probs = [[0, 0.5, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4], [0, 0.05, 0.05, 0.9], [0.25, 0.25, 0.25, 0.25]]
 * const model = (prefix) => log(tensor(probs[prefix.at(-1)]))
 * const end = run(greedyDecoding(model, { prompt: [0], maxTokens: 5, stop: [3] }), undefined, 10)
 * print('tokens:', end.tokens, 'steps:', end.t)
 * print('probability:', Math.exp(end.logProb))
 * print('last distribution:', end.probs)
 */
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
 * penalty, temperature, top-k, top-p) with the step's stream. `logProb` is still under the raw distribution.
 *
 * @param logits The model: the next-token logits after a prefix.
 * @param options The prompt, `maxTokens` and stop tokens, and the processors.
 * @returns The algorithm, one token per step; run it with a `stream`.
 *
 * @example Greedy against sampling with three seeds, and top-1 sampling, which is greedy
 * const probs = [[0, 0.5, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4], [0, 0.05, 0.05, 0.9], [0.25, 0.25, 0.25, 0.25]]
 * const model = (prefix) => log(tensor(probs[prefix.at(-1)]))
 * const options = { prompt: [0], maxTokens: 5, stop: [3] }
 * print('greedy:', run(greedyDecoding(model, options), undefined, 10).tokens)
 * print('seed 0:', run(samplingDecoding(model, options), undefined, 10, { stream: stream(0) }).tokens)
 * print('seed 1:', run(samplingDecoding(model, options), undefined, 10, { stream: stream(1) }).tokens)
 * print('seed 2:', run(samplingDecoding(model, options), undefined, 10, { stream: stream(2) }).tokens)
 * print('top-1:', run(samplingDecoding(model, { ...options, topK: 1 }), undefined, 10, { stream: stream(0) }).tokens)
 *
 * @example The distribution of one step before and after top-2 at temperature 0.5
 * const probs = [[0, 0.5, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4], [0, 0.05, 0.05, 0.9], [0.25, 0.25, 0.25, 0.25]]
 * const model = (prefix) => log(tensor(probs[prefix.at(-1)]))
 * const options = { prompt: [1], maxTokens: 1, topK: 2, temperature: 0.5 }
 * const one = run(samplingDecoding(model, options), undefined, 1, { stream: stream(0) })
 * print('raw:', one.probs)
 * print('filtered:', one.filtered)
 * print('drew:', one.token)
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
  /** The node's index in `tree` (0 for the root). */
  readonly id: number
  /** The parent node ($-1$ for the root, the prompt). */
  readonly parent: number
  /** The token appended ($-1$ for the root). */
  readonly token: number
  /** The step at which it was added (0 for the root). */
  readonly step: Size
  /** $\sum \log p$ of the hypothesis's generated tokens. */
  readonly logProb: number
  /** Kept after its step, in the beam or set aside as finished (else pruned). */
  readonly kept: boolean
  /** It ended with a stop token. */
  readonly finished: boolean
}

/** A hypothesis: a partial or finished sequence of beam search. */
export type Hypothesis = {
  /** Its node in the tree. */
  readonly node: number
  /** Its tokens, prompt included. */
  readonly tokens: readonly number[]
  /** $\sum \log p$ of its generated tokens. */
  readonly logProb: number
  /** Its log-probability divided by the length penalty, by which hypotheses are ranked. */
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
  /** True once the search has ended (see `beamSearch`). */
  readonly terminated: boolean
}

/** Options of `beamSearch`. */
export type BeamOptions = DecodingOptions & {
  /** Beam width $B$. */
  beams: Size
  /**
   * Length penalty $\alpha$: scores are $\log p / ((5 + \lvert Y \rvert)/6)^\alpha$ for $\lvert Y \rvert$ generated
   * tokens (Wu et al., 2016, eq. 14); 0 for none. Default 0.
   */
  lengthPenalty?: number
  /** Candidate tokens considered per live beam, the most probable (default $B$). */
  expand?: Size
}

/**
 * GNMT's length penalty $((5 + n)/6)^\alpha$ of a hypothesis of $n$ generated tokens (Wu et al., 2016, eq. 14). A
 * score divided by it favours longer hypotheses as $\alpha$ grows; $\alpha = 0$ gives 1.
 *
 * @param n The number of generated tokens $n$.
 * @param alpha The exponent $\alpha$.
 * @returns The penalty, 1 at $n = 1$.
 *
 * @example The penalty grows with length, and is 1 for alpha = 0
 * print('n = 1:', lengthPenalty(1, 0.6))
 * print('n = 10:', lengthPenalty(10, 0.6))
 * print('n = 10, alpha 0:', lengthPenalty(10, 0))
 */
export function lengthPenalty(n: Size, alpha: number): number {
  return ((5 + n) / 6) ** alpha
}

/**
 * Beam search: keep the $B$ best partial sequences; each step extends every live one by its `expand` most probable
 * tokens (raw distribution, no processors), and keeps the $B$ best extensions by score
 * $\sum \log p / \mathrm{lp}(\lvert Y \rvert)$ with GNMT's length penalty (`lengthPenalty`). A hypothesis ending in a
 * stop token is set aside as finished, without taking a live place. The search ends when `maxTokens` are generated,
 * when no live hypothesis is left, or (with $\alpha = 0$ only) when no live hypothesis can beat the best finished one.
 * `tree` records every expansion, kept or pruned, so the search can be drawn. Deterministic.
 *
 * @param logits The model: the next-token logits after a prefix.
 * @param options The prompt, `maxTokens`, stop tokens, beam width, length penalty and candidates per beam.
 * @returns The algorithm, one generated position per step.
 *
 * @example Greedy takes the likelier first token; a beam of two finds the likelier pair
 * const probs = [[0, 0.5, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4], [0, 0.05, 0.05, 0.9], [0.25, 0.25, 0.25, 0.25]]
 * const model = (prefix) => log(tensor(probs[prefix.at(-1)]))
 * const greedy = run(greedyDecoding(model, { prompt: [0], maxTokens: 2 }), undefined, 10)
 * const beam = run(beamSearch(model, { prompt: [0], maxTokens: 2, beams: 2 }), undefined, 10)
 * print('greedy:', greedy.tokens, 'probability:', Math.exp(greedy.logProb))
 * print('beam of 2:', beam.best.tokens, 'probability:', Math.exp(beam.best.logProb))
 * print('tree (parent, token, kept):', beam.tree.map((n) => [n.parent, n.token, n.kept]))
 *
 * @example A stop token sets hypotheses aside as finished
 * const probs = [[0, 0.5, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4], [0, 0.05, 0.05, 0.9], [0.25, 0.25, 0.25, 0.25]]
 * const model = (prefix) => log(tensor(probs[prefix.at(-1)]))
 * const end = run(beamSearch(model, { prompt: [0], maxTokens: 4, beams: 2, stop: [3] }), undefined, 10)
 * print('finished:', end.finished.map((h) => [h.tokens, Math.exp(h.logProb)]))
 * print('best:', end.best.tokens, 'after', end.t, 'steps')
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
  /** The $\gamma$ draft tokens (fewer near `maxTokens`, or when one is a stop token). */
  readonly drafted: readonly number[]
  /** $\min(1, p(x_i)/q(x_i))$ of each draft token $x_i$. */
  readonly acceptance: readonly number[]
  /** How many draft tokens were accepted (a prefix of `drafted`). */
  readonly accepted: Size
  /** The token appended after them: the resampled rejection or the bonus token. */
  readonly correction: number
  /**
   * True when the correction came from the residual $\max(0, p - q)$ (a rejection), false for the bonus token.
   */
  readonly rejected: boolean
}

/** The state of `speculativeDecoding`. */
export interface SpeculativeState extends Status {
  /** Prompt and generated tokens. */
  readonly tokens: readonly number[]
  /** Generated tokens so far. */
  readonly generated: Size
  /** The last round (null at $t = 0$). */
  readonly round: SpeculativeRound | null
  /** Draft tokens proposed so far; `acceptedTotal / proposed` is the acceptance rate $\hat{\alpha}$. */
  readonly proposed: Size
  /** Draft tokens accepted so far. */
  readonly acceptedTotal: Size
  /** Calls of the target model, one verification pass per round. */
  readonly targetCalls: Size
  /** Calls of the draft model, one per draft token. */
  readonly draftCalls: Size
  /** True once `maxTokens` tokens are generated or a stop token is. */
  readonly terminated: boolean
}

/** Options of `speculativeDecoding`. */
export type SpeculativeOptions = DecodingOptions & {
  /** Draft tokens per round $\gamma$ (default 4). */
  lookahead?: Size
  /** Sampling temperature applied to both models (default 1). */
  temperature?: number
}

/**
 * Speculative sampling (Leviathan, Kalman and Matias, 2023; Chen et al., 2023): each step is one round. The draft model
 * $q$ proposes $\gamma$ tokens by sampling; the target $p$ scores all $\gamma + 1$ prefixes (one batched pass in
 * practice, counted as one target call); draft token $x_i$ is accepted with probability $\min(1, p(x_i)/q(x_i))$; at
 * the first rejection a token is drawn from the normalised $\max(0, p - q)$ and the round ends, and if all $\gamma$
 * are accepted a bonus token is drawn from $p$. Both models are sampled at `temperature`, with no other processor. The
 * output has exactly the target's sampling distribution; the speed-up comes from accepting several tokens per target
 * call, at the expected rate `expectedTokensPerCall` for acceptance rate $\alpha$.
 *
 * @param target The target model $p$, whose distribution the output follows.
 * @param draft The draft model $q$, cheap, over the same vocabulary.
 * @param options The prompt, `maxTokens`, stop tokens, lookahead $\gamma$ and temperature.
 * @returns The algorithm, one round per step; run it with a `stream`.
 *
 * @example A uniform draft: one round's acceptance probabilities and the correction
 * const probs = [[0, 0.5, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4], [0, 0.05, 0.05, 0.9], [0.25, 0.25, 0.25, 0.25]]
 * const target = (prefix) => log(tensor(probs[prefix.at(-1)]))
 * const draft = () => tensor([0, 0, 0, 0])
 * const alg = speculativeDecoding(target, draft, { prompt: [0], maxTokens: 8, lookahead: 3 })
 * const one = run(alg, undefined, 1, { stream: stream(0) })
 * print('round:', one.round)
 * print('tokens:', one.tokens)
 *
 * @example A draft equal to the target is always accepted: four tokens per target call
 * const probs = [[0, 0.5, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4], [0, 0.05, 0.05, 0.9], [0.25, 0.25, 0.25, 0.25]]
 * const target = (prefix) => log(tensor(probs[prefix.at(-1)]))
 * const options = { prompt: [0], maxTokens: 8, lookahead: 3 }
 * const end = run(speculativeDecoding(target, target, options), undefined, 100, { stream: stream(0) })
 * print('tokens:', end.tokens)
 * print('target calls:', end.targetCalls, 'acceptance rate:', end.acceptedTotal / end.proposed)
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

/**
 * The expected tokens per target call of speculative decoding with acceptance rate $\alpha$ and lookahead $\gamma$,
 * $(1 - \alpha^{\gamma + 1})/(1 - \alpha)$ (Leviathan, Kalman and Matias, 2023, eq. 1), and $\gamma + 1$ at
 * $\alpha = 1$.
 *
 * @param alpha The acceptance rate $\alpha$, the chance that a draft token is accepted, in $[0, 1]$.
 * @param lookahead The draft tokens per round $\gamma$.
 * @returns The expected number of tokens generated per target call, from 1 to $\gamma + 1$.
 *
 * @example From no acceptance (one token per call) to full acceptance (five)
 * print('alpha 0:', expectedTokensPerCall(0, 4))
 * print('alpha 0.8:', expectedTokensPerCall(0.8, 4))
 * print('alpha 1:', expectedTokensPerCall(1, 4))
 */
export function expectedTokensPerCall(alpha: number, lookahead: Size): number {
  return alpha === 1 ? lookahead + 1 : (1 - alpha ** (lookahead + 1)) / (1 - alpha)
}
