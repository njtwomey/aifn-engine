/**
 * A tiny GPT (Radford et al., 2018, 2019): a decoder-only transformer trained to predict the next character of a toy
 * corpus. Token embeddings plus absolute positions (learned or sinusoidal) or relative ones inside attention (RoPE,
 * ALiBi), a stack of causal pre-norm transformer blocks from `aifn-compute/nn/attention`, a final normalisation, and an
 * output layer tied to the embedding (Press and Wolf, 2017): with $\Emat$ the $V \times d$ embedding and $\Hmat$ the
 * final hidden states, the logits are $\Hmat\Emat^\top$. Small enough to train in the browser in well under a
 * minute, and its `logits` plug into every decoder of `aifn-compute/nn/decoding`.
 *
 * Training minimises the mean next-token cross-entropy over every position of the windows of the corpus, by Adam with
 * global-norm gradient clipping, through `aifn-compute/nn/training`'s `trainingLoop`. Runs are deterministic from
 * their stream.
 */

import { child, stream, type Stream } from 'aifn-compute/foundation/random'
import type { Size } from 'aifn-compute/foundation/contracts'
import {
  add,
  fromData,
  matmul,
  ones,
  reshape,
  shapeOfValue,
  slice,
  sub,
  take,
  transpose,
  unwrap,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import type { Estimator, FitOptions, Scores, Trained } from 'aifn-compute/learning/estimators'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import {
  learnedPositions,
  sinusoidalPositions,
  transformerBlock,
  TransformerBlock,
  type FeedForwardKind,
  type TransformerBlockOptions,
  type TransformerBlockParams,
} from 'aifn-compute/nn/attention'
import type { LogitsFn } from 'aifn-compute/nn/decoding'
import { normalInit } from 'aifn-compute/nn/init'
import { childContext, layerNorm, rmsNorm, tap, type Context, type NormParams } from 'aifn-compute/nn/layers'
import { trainingLoop, type TrainingState } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { charCorpus } from './corpus'
import type { TokenCorpus } from './ngram'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Where position enters a tiny GPT. */
export type GptPosition = 'learned' | 'sinusoidal' | 'rope' | 'alibi' | 'none'

/** The architecture of a tiny GPT. */
export type GptConfig = {
  /** Vocabulary size $V$. */
  vocabulary: Size
  /** Context length (the longest prefix it reads). Default 32. */
  context?: Size
  /** Width $d$ of the residual stream ($d_{\mathrm{model}}$). Default 32; even for sinusoidal positions. */
  width?: Size
  /** Transformer blocks. Default 2. */
  layers?: Size
  /** Attention heads. Default 4. */
  heads?: Size
  /** Key–value heads (grouped-query attention). Default `heads`. */
  kvHeads?: Size
  /** Positional scheme. Default `learned`. */
  position?: GptPosition
  /** Feed-forward kind. Default `mlp` (GELU). */
  feedForward?: FeedForwardKind
  /** `layer` (default) or `rms` normalisation. */
  norm?: 'layer' | 'rms'
}

/** Parameters of a tiny GPT. */
export type GptParams = {
  /** Token embeddings, $V \times d$, also the output layer (tied). */
  embedding: Tensor
  /** Learned positions, `context` $\times d$ (with `position: 'learned'` only). */
  positions?: Tensor
  /** The parameters of each transformer block, first to last. */
  blocks: TransformerBlockParams[]
  /** The final normalisation's scale, and shift for layer normalisation. */
  finalNorm: NormParams
}

/** A tiny GPT: initialise parameters, map ids `[..., T]` to next-token logits `[..., T, V]`. */
export type Gpt = {
  /** The architecture, with every default filled in. */
  readonly config: Required<GptConfig>
  /** A one-line description for legends: layers, width, heads and positions. */
  readonly label: string
  /**
   * Fresh parameters from stream `s`: embeddings from $\Gauss(0, 0.3^2)$, learned positions from
   * $\Gauss(0, 0.1^2)$, each block by its own `init`, and the final normalisation at the identity.
   */
  init(s: Stream): GptParams
  /**
   * Logits `[..., T, V]` for ids `[..., T]` ($T \le$ `context`, else `DomainError`), each position's logits
   * predicting the token after it. With a tapping context it records `embedding.tokens`,
   * `embedding.positions` (absolute schemes), `embedding` (the residual stream entering the first block), every block's
   * activations below `blocks.<i>` (see `transformerBlock`), `final` (the last normalisation) and `logits`.
   */
  apply(params: GptParams, ids: Tensor | readonly number[], ctx?: Context): Value
}

/**
 * A configuration with every default filled in: context and width 32, 2 layers of 4 heads (and as many key-value
 * heads as heads), learned positions, a GELU MLP and layer normalisation.
 *
 * @param c The configuration as given; its fields override the defaults.
 * @returns The complete configuration.
 */
const defaults = (c: GptConfig): Required<GptConfig> => ({
  context: 32,
  width: 32,
  layers: 2,
  heads: 4,
  kvHeads: c.heads ?? 4,
  position: 'learned',
  feedForward: 'mlp',
  norm: 'layer',
  ...c,
})

/**
 * A decoder-only transformer language model (a tiny GPT) with the given architecture. Throws `DomainError` from
 * `apply` when the ids are longer than the context.
 *
 * @param config The architecture: the vocabulary size, and optionally the context, width, depth, heads, positional
 *   scheme, feed-forward kind and normalisation.
 * @returns The model: its full configuration, a label, `init` and `apply`.
 *
 * @example A one-block GPT maps three ids to three rows of logits
 * const gpt = Gpt({ vocabulary: 5, context: 8, width: 8, layers: 1, heads: 2 })
 * const params = gpt.init(stream(0))
 * print(gpt.label)
 * print('logits of 3 ids:', shapeOfValue(gpt.apply(params, [0, 1, 2])))
 */
export function Gpt(config: GptConfig): Gpt {
  const c = defaults(config)
  const blockOptions: TransformerBlockOptions = {
    heads: c.heads,
    kvHeads: c.kvHeads,
    causal: true,
    placement: 'pre',
    norm: c.norm,
    feedForward: c.feedForward,
    position: c.position === 'rope' ? 'rope' : c.position === 'alibi' ? 'alibi' : 'none',
  }
  const block = TransformerBlock(c.width, blockOptions)
  const normalise = (p: NormParams, h: Value) =>
    c.norm === 'rms' ? rmsNorm(h, p.gamma) : layerNorm(h, p.gamma, p.beta)
  const sinusoid = c.position === 'sinusoidal' ? sinusoidalPositions(c.context, c.width) : null
  return {
    config: c,
    label: `Gpt(${c.layers} × ${c.width}, ${c.heads} heads, ${c.position} positions)`,
    init: (s) => ({
      embedding: normalInit(0.3)(child(s, 'embedding'), [c.vocabulary, c.width], {
        fanIn: c.vocabulary,
        fanOut: c.width,
      }),
      ...(c.position === 'learned'
        ? {
            positions: normalInit(0.1)(child(s, 'positions'), [c.context, c.width], {
              fanIn: c.context,
              fanOut: c.width,
            }),
          }
        : {}),
      blocks: Array.from({ length: c.layers }, (_, i) => block.init(child(s, 'block', i))),
      finalNorm: c.norm === 'rms' ? { gamma: ones([c.width]) } : { gamma: ones([c.width]), beta: zeros([c.width]) },
    }),
    apply: (params, ids, ctx) => {
      const idTensor = Array.isArray(ids)
        ? fromData(Int32Array.from(ids as number[]), [(ids as number[]).length])
        : (ids as Tensor)
      const T = idTensor.shape[idTensor.shape.length - 1]
      if (T > c.context) throw new DomainError('Gpt', `Gpt: ${T} tokens exceed the context of ${c.context}`)
      const embedding = childContext(ctx, 'embedding')
      const tokens = tap(embedding, take(params.embedding, idTensor), 'tokens')
      let h: Value = tokens
      const positions = Array.from({ length: T }, (_, i) => i)
      if (params.positions) h = learnedPositions(params.positions, h, positions)
      if (sinusoid) h = add(h, slice(sinusoid, [0, T]))
      if (ctx?.tap && h !== tokens) tap(embedding, sub(h, tokens), 'positions')
      h = tap(embedding, h)
      params.blocks.forEach((bp, i) => {
        h = transformerBlock(bp, h, blockOptions, { positions }, childContext(childContext(ctx, 'blocks'), i)).output
      })
      h = tap(childContext(ctx, 'final'), normalise(params.finalNorm, h))
      return tap(childContext(ctx, 'logits'), matmul(h, transpose(params.embedding)))
    },
  }
}

/**
 * Windows of a token sequence for next-token training: inputs `x` and targets `y`, both $N \times T$, the targets
 * the inputs shifted by one. A window starts every `stride` tokens while its target still fits in the sequence.
 *
 * @param ids The token sequence.
 * @param context The window length $T$.
 * @param stride The step between window starts (default half the context, at least 1), so windows overlap.
 * @returns `x`, the $N$ windows as int32 ids, and `y`, each window's next tokens.
 *
 * @example Windows of three, one token apart
 * const { x, y } = nextTokenWindows([0, 1, 2, 3, 4, 5, 6], 3)
 * print('x:', x)
 * print('y:', y)
 */
export function nextTokenWindows(
  ids: readonly number[],
  context: Size,
  stride: Size = Math.max(1, Math.floor(context / 2)),
): { x: Tensor; y: Tensor } {
  const starts: number[] = []
  for (let s = 0; s + context < ids.length; s += stride) starts.push(s)
  const x = new Int32Array(starts.length * context)
  const y = new Int32Array(starts.length * context)
  starts.forEach((s, i) => {
    for (let j = 0; j < context; j++) {
      x[i * context + j] = ids[s + j]
      y[i * context + j] = ids[s + j + 1]
    }
  })
  return { x: fromData(x, [starts.length, context]), y: fromData(y, [starts.length, context]) }
}

/**
 * The mean next-token cross-entropy (nats per token) of logits against targets, over every position
 * (differentiable).
 *
 * @param logits The logits, $N \times T \times V$ (any leading shape, flattened with the targets).
 * @param targets The target ids, $N \times T$.
 * @returns The mean cross-entropy, a scalar.
 *
 * @example Uniform logits cost $\log V$ nats a token
 * const targets = fromData(Int32Array.from([0, 1, 2]), [1, 3])
 * print('uniform over 4 tokens:', nextTokenLoss(zeros([1, 3, 4]), targets), ' log 4 =', Math.log(4))
 */
export function nextTokenLoss(logits: Value, targets: Tensor): Value {
  const V = shapeOfValue(logits).at(-1)!
  return softmaxCrossEntropy(reshape(logits, [-1, V]), reshape(targets, [-1]) as Tensor)
}

/** Options of `gptTraining` and `charGpt`. */
export type GptTrainingOptions = {
  /** Windows per step (default 16). */
  batchSize?: Size
  /** Adam's step size (default 0.01). */
  stepSize?: number
  /** Rescale gradients above this global norm (default 1). */
  clipNorm?: number
}

/**
 * Next-token training of a tiny GPT on a corpus, as a traceable `trainingLoop`: minibatch Adam, with gradients
 * clipped to a global norm, on the cross-entropy of every position of the corpus's overlapping windows of the context
 * length (`nextTokenWindows`), shuffled each epoch.
 *
 * @param model The GPT; its context sets the window length.
 * @param corpus The token corpus to train on; its ids must be below the model's vocabulary size.
 * @param options The batch size, step size and clipping norm.
 * @returns The algorithm, to run with `run` or `trace` from `{ params }`.
 *
 * @example The loss falls on a repeating string
 * const corpus = charCorpus('abcabcabcabcabcabcabcabc')
 * const model = Gpt({ vocabulary: 3, context: 4, width: 8, layers: 1, heads: 2 })
 * const alg = gptTraining(model, corpus, { batchSize: 4, stepSize: 0.03 })
 * const record = { loss: (s) => s.loss }
 * const tr = trace(alg, { params: model.init(stream(0)) }, 30, { stream: stream(1), every: 10, record })
 * print('step:', tr.index)
 * print('loss:', tr.series.loss)
 */
export function gptTraining(
  model: Gpt,
  corpus: TokenCorpus,
  options: GptTrainingOptions = {},
): Algorithm<{ params: GptParams }, TrainingState<GptParams>> {
  const data = nextTokenWindows(corpus.ids, model.config.context)
  return trainingLoop<GptParams, { x: Tensor; y: Tensor }>({
    loss: (p, b, ctx) => nextTokenLoss(model.apply(p, b.x, ctx), b.y),
    data,
    batchSize: Math.min(options.batchSize ?? 16, data.x.shape[0]),
    optimizer: adamRule({ stepSize: options.stepSize ?? 0.01 }) as never,
    clipNorm: options.clipNorm ?? 1,
  })
}

/**
 * The next-token logits of a GPT as a function of the prefix, for `aifn-compute/nn/decoding`. It reads the last
 * `context` tokens of the prefix; an empty prefix reads token 0 instead.
 *
 * @param model The GPT.
 * @param params Its parameters.
 * @returns A function from a prefix of ids to the $V$ logits of the token after it.
 *
 * @example Next-character probabilities after training on a repeating string
 * const corpus = charCorpus('abcabcabcabcabcabcabcabc')
 * const model = Gpt({ vocabulary: 3, context: 4, width: 8, layers: 1, heads: 2 })
 * const alg = gptTraining(model, corpus, { batchSize: 4, stepSize: 0.03 })
 * const { params } = run(alg, { params: model.init(stream(0)) }, 30, { stream: stream(1) })
 * const next = gptLogits(model, params)
 * for (const prefix of ['ab', 'abca']) {
 *   const p = toFlat(next(encodeChars(corpus, prefix))).map(Math.exp)
 *   const total = p.reduce((a, b) => a + b, 0)
 *   print(`P(next | "${prefix}") over a, b, c:`, p.map((v) => v / total))
 * }
 */
export function gptLogits(model: Gpt, params: GptParams): LogitsFn {
  return (prefix) => {
    const window = prefix.length === 0 ? [0] : prefix.slice(-model.config.context)
    const logits = unwrap(model.apply(params, window)) as Tensor
    const V = model.config.vocabulary
    return fromData(Float64Array.from(logits.data as Float64Array).slice((window.length - 1) * V, window.length * V), [
      V,
    ])
  }
}

/** A fitted tiny GPT. */
export type CharGptModel = Scores<Tensor> &
  Trained<TrainingState<GptParams>> & {
    /** The estimator kind. */
    readonly kind: 'model'
    /** The estimator's name. */
    readonly name: 'char-gpt'
    /** The GPT, with the corpus's vocabulary size. */
    readonly model: Gpt
    /** The fitted parameters. */
    readonly params: GptParams
    /** The next-token logits after a prefix, as `gptLogits` gives them. */
    readonly logits: LogitsFn
    /** Next-token logits for contexts `[N, k]` of ids: `[N, V]`, each row from the last `context` ids of its row. */
    score(contexts: Tensor): Tensor
  }

/**
 * Hyperparameters of `charGpt`: the architecture without the vocabulary size, the training options, and `steps`, the
 * number of Adam steps (default 300).
 */
export type CharGptOptions = Omit<GptConfig, 'vocabulary'> & GptTrainingOptions & { steps?: Size }

/**
 * A tiny GPT fitted to a token corpus by `steps` Adam steps (default 300) of next-token training (`gptTraining`). The
 * vocabulary size is the corpus's. The fit's stream (default `stream('char-gpt')`) seeds the initialisation and the
 * minibatches, and the training trace records the loss every `trace.every` steps (default every step).
 *
 * @param options The architecture, the training options and the number of steps.
 * @returns The estimator; `fit` returns the GPT, its parameters, its training trace and its next-token scores.
 *
 * @example Fit to a repeating string, then score one-letter contexts
 * const corpus = charCorpus('abcabcabcabcabcabcabcabc')
 * const gpt = charGpt({ width: 8, layers: 1, heads: 2, context: 4, steps: 30, stepSize: 0.03, batchSize: 4 })
 * const fitted = gpt.fit(corpus, { stream: stream(0), trace: { every: 10 } })
 * print('loss every 10 steps:', fitted.training.series.loss)
 * print('logits after "a", "b", "c":', fitted.score(fromData(Int32Array.from([0, 1, 2]), [3, 1])))
 */
export function charGpt(options: CharGptOptions = {}): Estimator<TokenCorpus, CharGptModel> {
  const { steps = 300, batchSize, stepSize, clipNorm, ...arch } = options
  return {
    name: 'char-gpt',
    params: options,
    fit(corpus, fit: FitOptions = {}) {
      const model = Gpt({ ...arch, vocabulary: corpus.vocabulary.tokens.length })
      const s = fit.stream ?? stream('char-gpt')
      const training: Trace<TrainingState<GptParams>> = trace(
        gptTraining(model, corpus, { batchSize, stepSize, clipNorm }),
        { params: model.init(child(s, 'init')) },
        steps,
        { stream: s, every: fit.trace?.every ?? 1, record: { loss: (st) => st.loss } },
      )
      const params = training.final.params
      const logits = gptLogits(model, params)
      return {
        kind: 'model',
        name: 'char-gpt',
        model,
        params,
        training,
        logits,
        score: (contexts: Tensor) => {
          const [n, k] = contexts.shape
          const v = Array.from(contexts.data as ArrayLike<number>)
          const V = model.config.vocabulary
          const out = new Float64Array(n * V)
          for (let i = 0; i < n; i++)
            out.set((unwrap(logits(v.slice(i * k, (i + 1) * k))) as Tensor).data as Float64Array, i * V)
          return fromData(out, [n, V])
        },
      }
    },
  }
}

/** A snapshot of `gptTrainingRun`: the steps taken, the loss of every step so far, and the parameters. */
export type GptTrainingSnapshot = {
  /** Steps taken. */
  readonly step: Size
  /** Steps in the whole run. */
  readonly steps: Size
  /** The minibatch loss at the start and after each step so far: `step` $+ 1$ values. */
  readonly losses: readonly number[]
  /** The model's full configuration. */
  readonly config: Required<GptConfig>
  /** The parameters after `step` steps. */
  readonly params: GptParams
}

/** Options of `gptTrainingRun`. */
export type GptTrainingRunOptions = Omit<GptConfig, 'vocabulary'> &
  GptTrainingOptions & {
    /** The corpus text (default the nursery rhymes). */
    text?: string
    /** Adam steps (default 300). */
    steps?: Size
    /** Yield a snapshot every this many steps (default 25). */
    every?: Size
    /** The root stream's seed (default 'char-gpt'). */
    seed?: string | number
  }

/**
 * Train a tiny GPT on a character corpus, yielding a snapshot every `every` steps and at the end: a generator, so a
 * worker can stream the run to a page that shows the loss falling and decodes from the latest parameters. The
 * vocabulary is the text's own alphabet (`charCorpus`), and the run is deterministic from its seed.
 *
 * @param options The text, the architecture, the training options, the number of steps, how often to yield and the
 *   seed.
 * @returns A generator of snapshots, the last at step `steps`.
 *
 * @example Three snapshots of a short run
 * const options = { text: 'abcabcabcabcabcabcabcabc', width: 8, layers: 1, heads: 2, context: 4, steps: 30, every: 10 }
 * for (const s of gptTrainingRun({ ...options, batchSize: 4, stepSize: 0.03, seed: 0 }))
 *   print('step', s.step, ' loss', s.losses.at(-1))
 */
export function* gptTrainingRun(options: GptTrainingRunOptions = {}): Generator<GptTrainingSnapshot> {
  const { text, steps = 300, every = 25, seed = 'char-gpt', batchSize, stepSize, clipNorm, ...arch } = options
  const corpus = charCorpus(text)
  const model = Gpt({ ...arch, vocabulary: corpus.vocabulary.tokens.length })
  const root = stream(seed)
  const alg = gptTraining(model, corpus, { batchSize, stepSize, clipNorm })
  let state = alg.init({ params: model.init(child(root, 'init')) }, child(root, 'init'))
  const losses = [state.loss]
  for (let t = 0; t < steps; t++) {
    state = alg.step(state, { t, stream: child(root, 'step', t) })
    losses.push(state.loss)
    if ((t + 1) % every === 0 || t + 1 === steps)
      yield { step: t + 1, steps, losses: [...losses], config: model.config, params: state.params }
  }
}
