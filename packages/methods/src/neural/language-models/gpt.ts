/**
 * A tiny GPT (Radford et al., 2018, 2019): a decoder-only transformer trained to predict the next character of a toy
 * corpus. Token embeddings plus absolute positions (learned or sinusoidal) or relative ones inside attention (RoPE,
 * ALiBi), a stack of causal pre-norm transformer blocks from `aifn-compute/nn/attention`, a final normalisation, and an output
 * layer tied to the embedding (Press and Wolf, 2017). Small enough to train in the browser in well under a minute,
 * and its `logits` plug into every decoder of `aifn-compute/nn/decoding`.
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
  /** Vocabulary size V. */
  vocabulary: Size
  /** Context length (the longest prefix it reads). Default 32. */
  context?: Size
  /** Width d_model. Default 32. */
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
  /** Token embeddings [V, d], also the output layer (tied). */
  embedding: Tensor
  /** Learned positions [context, d] (with `position: 'learned'`). */
  positions?: Tensor
  blocks: TransformerBlockParams[]
  finalNorm: NormParams
}

/** A tiny GPT: initialise parameters, map ids [..., T] to next-token logits [..., T, V]. */
export type Gpt = {
  readonly config: Required<GptConfig>
  readonly label: string
  init(s: Stream): GptParams
  /**
   * Logits [..., T, V] for ids [..., T] (T ≤ context). With a tapping context it records `embedding.tokens`,
   * `embedding.positions` (absolute schemes), `embedding` (the residual stream entering the first block), every block's
   * activations below `blocks.<i>` (see `transformerBlock`), `final` (the last normalisation) and `logits`.
   */
  apply(params: GptParams, ids: Tensor | readonly number[], ctx?: Context): Value
}

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

/** A decoder-only transformer language model (a tiny GPT) with the given architecture. */
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

/** Windows of a token sequence for next-token training: x [N, T] and the targets y [N, T] (x shifted by one). */
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

/** The mean next-token cross-entropy (nats per token) of logits [N, T, V] against targets [N, T]. */
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
 * Next-token training of a tiny GPT on a corpus, as a traceable `trainingLoop` (minibatch Adam on the cross-entropy of
 * every position of random windows of the context length).
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

/** The next-token logits [V] of a GPT after a prefix (the last `context` tokens; an empty prefix reads token 0). */
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
    readonly kind: 'model'
    readonly name: 'char-gpt'
    readonly model: Gpt
    readonly params: GptParams
    readonly logits: LogitsFn
    /** Next-token logits for contexts [N, k] of ids: [N, V]. */
    score(contexts: Tensor): Tensor
  }

/** Hyperparameters of `charGpt`. */
export type CharGptOptions = Omit<GptConfig, 'vocabulary'> & GptTrainingOptions & { steps?: Size }

/**
 * A tiny GPT fitted to a token corpus by `steps` Adam steps (default 300) of next-token training. The vocabulary size
 * is the corpus's.
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
  readonly step: Size
  readonly steps: Size
  readonly losses: readonly number[]
  readonly config: Required<GptConfig>
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
 * worker can stream the run to a page that shows the loss falling and decodes from the latest parameters.
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
