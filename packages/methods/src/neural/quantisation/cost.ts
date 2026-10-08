/**
 * Back-of-envelope costs of a transformer: the memory and decoding throughput of serving it, and the memory, compute
 * and price of fine-tuning it fully, with LoRA or with QLoRA.
 *
 * Serving: the weights and key–value cache at a given precision, and the decoding throughput a device's memory
 * bandwidth and arithmetic allow (the roofline argument: generating one token reads every weight once, so small-batch
 * decoding is bound by bandwidth, not FLOPs). Kaplan et al. (2020) count $2N$ FLOPs per token for $N$ parameters; the
 * KV cache holds $2 \cdot \text{layers} \cdot \text{heads} \cdot \text{head size}$ values per token.
 *
 * Fine-tuning follows the derivations of the AI Field Notes on QLoRA ("QLoRA and the memory cost of fine-tuning") and
 * on bootstrapping a language model for a new task. The model state is $16\Psi$ bytes for full fine-tuning with
 * mixed-precision Adam (Rajbhandari et al., 2020), $2\Psi + 16\Psi_a$ for LoRA on a bf16 base, and
 * $0.516\,\Psi_{\text{lin}} + 2(\Psi - \Psi_{\text{lin}}) + 16\Psi_a$ for QLoRA, whose linear layers are stored in NF4
 * with double quantisation (Dettmers et al., 2023). Activations follow Korthikanti et al. (2022); logits take
 * $4 b s V$ bytes when materialised. Training costs $2\Psi$ FLOPs per token forward, $2\Psi$ for the gradients with
 * respect to activations, $2\Psi$ for the weight gradients of full fine-tuning (or $2\Psi_a$ for adapters), and $2\Psi$
 * more when checkpointing recomputes the forward pass: $6\Psi$ for full fine-tuning without checkpointing (Kaplan et
 * al., 2020). Sizes are in bytes, and a GB is $10^9$ bytes.
 */

/** A decoder-only transformer, for counting. */
export interface TransformerShape {
  /** Parameters N. */
  parameters: number
  layers: number
  /** Key–value heads (fewer than query heads under grouped-query attention) and their size. */
  kvHeads: number
  headDim: number
}

/** A serving configuration. */
export interface ServingSetup {
  /** Bits per weight (16, 8, 4, …) and per cached key or value. */
  weightBits: number
  kvBits: number
  /** Sequences decoded together and tokens held per sequence. */
  batch: number
  context: number
}

/** A device, for the roofline. */
export interface Device {
  /** Memory in GB ($10^9$ bytes), bandwidth in GB/s and peak dense throughput in TFLOP/s at the compute precision. */
  memoryGb: number
  bandwidthGbs: number
  tflops: number
}

/** The memory of a served model. */
export interface ServingMemory {
  /** Bytes of weights: $N \cdot \text{bits} / 8$. */
  weights: number
  /** Bytes of KV cache: $2 \cdot \text{layers} \cdot \text{kvHeads} \cdot \text{headDim} \cdot \text{context} \cdot \text{batch} \cdot \text{kvBits} / 8$. */
  kvCache: number
  total: number
  /** Whether the total fits the device. */
  fits: boolean
}

/**
 * The memory a served model's weights and key–value cache take.
 *
 * @param model The model's parameter count, layers, key–value heads and head size.
 * @param setup The bits per weight and per cached value, the batch and the context length in tokens.
 * @param device The device whose memory `fits` is checked against; without it `fits` is true.
 * @returns The bytes of weights and of cache, their total, and whether it fits the device.
 *
 * @example An 8B model in 16 and 4 bits
 * const llama = { parameters: 8.03e9, layers: 32, kvHeads: 8, headDim: 128 }
 * for (const weightBits of [16, 4]) {
 *   const m = servingMemory(llama, { weightBits, kvBits: 16, batch: 8, context: 4096 }, { memoryGb: 24, bandwidthGbs: 1000, tflops: 165 })
 *   print(`${weightBits}-bit weights: ${(m.total / 1e9).toFixed(1)} GB, fits 24 GB: ${m.fits}`)
 * }
 */
export function servingMemory(
  model: TransformerShape,
  setup: ServingSetup,
  device?: Pick<Device, 'memoryGb'>,
): ServingMemory {
  const weights = (model.parameters * setup.weightBits) / 8
  const kvCache = (2 * model.layers * model.kvHeads * model.headDim * setup.context * setup.batch * setup.kvBits) / 8
  const total = weights + kvCache
  return { weights, kvCache, total, fits: device ? total <= device.memoryGb * 1e9 : true }
}

/** Decoding throughput bounds of one step (one token for each sequence of the batch). */
export interface DecodeThroughput {
  /** Seconds per step if bound by memory traffic (weights once + every sequence's cache) and by arithmetic (2N per token). */
  memoryTime: number
  computeTime: number
  /** The step time, the larger of the two, and tokens per second over the batch. */
  stepTime: number
  tokensPerSecond: number
  /** Which bound is active, and the arithmetic intensity (FLOPs per byte read) of a step. */
  bound: 'memory' | 'compute'
  intensity: number
  /** The batch above which decoding becomes compute-bound on this device (ignoring the cache). */
  ridgeBatch: number
}

/**
 * The roofline estimate of decoding throughput: one step reads every weight and every sequence's cache once and does
 * $2N$ FLOPs per sequence, and takes the longer of the memory time and the compute time.
 *
 * @param model The model's parameter count, layers, key–value heads and head size.
 * @param setup The bits per weight and per cached value, the batch and the context length in tokens.
 * @param device The device's bandwidth (GB/s) and peak throughput (TFLOP/s).
 * @returns The memory and compute times of a step, the step time, the tokens per second over the batch, which bound is
 *   active, the arithmetic intensity, and the batch at which decoding turns compute-bound.
 *
 * @example Small batches are bound by bandwidth
 * const llama = { parameters: 8.03e9, layers: 32, kvHeads: 8, headDim: 128 }
 * const h100 = { memoryGb: 80, bandwidthGbs: 3350, tflops: 990 }
 * for (const batch of [1, 64]) {
 *   const d = decodeThroughput(llama, { weightBits: 16, kvBits: 16, batch, context: 2048 }, h100)
 *   print(`batch ${batch}: ${Math.round(d.tokensPerSecond)} tokens/s, ${d.bound}-bound`)
 * }
 */
export function decodeThroughput(model: TransformerShape, setup: ServingSetup, device: Device): DecodeThroughput {
  const mem = servingMemory(model, setup)
  const bytes = mem.weights + mem.kvCache
  const flops = 2 * model.parameters * setup.batch
  const memoryTime = bytes / (device.bandwidthGbs * 1e9)
  const computeTime = flops / (device.tflops * 1e12)
  const stepTime = Math.max(memoryTime, computeTime)
  // Ridge point: 2N·B / (weight bytes) = peak FLOPs / bandwidth.
  const ridgeBatch = (device.tflops * 1e12 * mem.weights) / (device.bandwidthGbs * 1e9 * 2 * model.parameters)
  return {
    memoryTime,
    computeTime,
    stepTime,
    tokensPerSecond: setup.batch / stepTime,
    bound: memoryTime >= computeTime ? 'memory' : 'compute',
    intensity: flops / bytes,
    ridgeBatch,
  }
}

// ── Fine-tuning ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A decoder-only transformer with grouped-query attention and a SwiGLU feed-forward block, for counting the cost of
 * training it. Each block has seven weight matrices: the query, key, value and output projections and the gate, up
 * and down projections.
 */
export interface TransformerTrainShape extends TransformerShape {
  /** The model width $h$. */
  width: number
  /** The feed-forward width $f$. */
  ffnWidth: number
  /** The vocabulary size $V$. */
  vocab: number
  /** Query heads $a$ (each of size `headDim`); `kvHeads` key–value heads share them. */
  queryHeads: number
  /** Whether the input embedding and the output matrix are one matrix ($Vh$ parameters) or two ($2Vh$). */
  tiedEmbeddings: boolean
}

/** A linear layer of a block that can carry an adapter. */
export type AdapterTarget = 'q' | 'k' | 'v' | 'o' | 'gate' | 'up' | 'down'

/** Every linear layer of a block, the default adapter targets. */
const ALL_TARGETS: readonly AdapterTarget[] = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']

/** A fine-tuning run's method and the sizes that set its memory. */
export interface FineTuningSetup {
  /** Train every weight, LoRA adapters on a bf16 base, or LoRA adapters on an NF4 base (QLoRA). */
  method: 'full' | 'lora' | 'qlora'
  /** The adapter rank $r$ (LoRA and QLoRA; default 16). */
  rank?: number
  /** The linear layers that carry adapters (default all seven). */
  targets?: readonly AdapterTarget[]
  /** Sequences per micro-batch $b$, and tokens per sequence $s$ (the longest the data holds). */
  microBatch: number
  sequence: number
  /** Gradient checkpointing: store each layer's input only and recompute the layer in the backward pass. */
  checkpointing: boolean
  /** Materialise all $bsV$ logits in 32 bits, or compute the loss in chunks, which holds a negligible part of them. */
  logits: 'full' | 'chunked'
  /**
   * Adam's moments in 32 bits (`adam32`, 8 bytes per trained parameter), in 8 bits (`adam8`, 2 bytes), or in 32 bits
   * in paged memory that can spill to the host (`paged`). Default `adam32`.
   */
  optimiser?: 'adam32' | 'adam8' | 'paged'
  /** QLoRA: quantise the NF4 block scales too (default true): 4.127 bits per weight instead of 4.5. */
  doubleQuant?: boolean
  /**
   * Without checkpointing: `fused` attention kernels recompute the $s \times s$ attention matrix in the backward pass
   * and store none of it (default); `stored` keeps it, Korthikanti et al.'s $5as/h$ term.
   */
  attention?: 'fused' | 'stored'
}

/** The memory of a fine-tuning run, in bytes. */
export interface FineTuningMemory {
  /** The base weights: bf16 for full fine-tuning and LoRA; NF4 linear layers and bf16 embeddings for QLoRA. */
  weights: number
  /** The adapters' 32-bit weights (0 for full fine-tuning). */
  adapterState: number
  /** Adam's moments, and for full fine-tuning the 32-bit master copy of the weights. */
  optimiserState: number
  /** One gradient per trained parameter: bf16 for full fine-tuning, 32-bit for adapters. */
  gradients: number
  /** The activations the backward pass needs, for one micro-batch. */
  activations: number
  /** The 32-bit logits of one micro-batch (0 with a chunked loss). */
  logits: number
  total: number
  /** Whether the run fits the device: the total, or with `paged` the total less the optimiser state. */
  fits: boolean
}

/** The compute and price of a fine-tuning run. */
export interface FineTuningCompute {
  /** FLOPs per training token. */
  flopsPerToken: number
  /** FLOPs of the whole run. */
  flops: number
  /** Wall-clock seconds at the run's model FLOP utilisation. */
  seconds: number
  /** The price of those seconds. */
  cost: number
}

/** A fine-tuning run's data and hardware, for its compute and price. */
export interface FineTuningRun {
  /** Tokens per epoch (prompt and completion), and epochs. */
  tokens: number
  epochs: number
  /** Model FLOP utilisation, the fraction of peak reached (e.g. 0.3). */
  mfu: number
  /** Peak dense FLOP/s of the hardware at the training precision (e.g. $9.9 \times 10^{14}$ for an H100 in bf16). */
  peakFlops: number
  /** Price per hour of the hardware. */
  pricePerHour: number
}

/**
 * Llama 3.1 8B (Grattafiori et al., 2024): 32 blocks of width 4096, feed-forward width 14,336, 32 query heads and 8
 * key–value heads of size 128, a vocabulary of 128,256 and separate input and output embeddings;
 * $\Psi = 8.03 \times 10^9$ without the normalisation weights.
 */
export const LLAMA_3_1_8B: TransformerTrainShape = {
  parameters: 8_029_995_008,
  layers: 32,
  width: 4096,
  ffnWidth: 14_336,
  vocab: 128_256,
  queryHeads: 32,
  kvHeads: 8,
  headDim: 128,
  tiedEmbeddings: false,
}

/**
 * Llama 3.1 70B (Grattafiori et al., 2024): 80 blocks of width 8192, feed-forward width 28,672, 64 query heads and 8
 * key–value heads of size 128, a vocabulary of 128,256 and separate input and output embeddings;
 * $\Psi = 70.55 \times 10^9$ without the normalisation weights.
 */
export const LLAMA_3_1_70B: TransformerTrainShape = {
  parameters: 70_552_387_584,
  layers: 80,
  width: 8192,
  ffnWidth: 28_672,
  vocab: 128_256,
  queryHeads: 64,
  kvHeads: 8,
  headDim: 128,
  tiedEmbeddings: false,
}

/**
 * The input and output sizes of each linear layer of a block, under grouped-query attention: the query and output
 * projections map between $h$ and $a \cdot \text{headDim}$, the key and value projections from $h$ to
 * $\text{kvHeads} \cdot \text{headDim}$, and the feed-forward projections between $h$ and $f$.
 *
 * @param shape The model.
 * @returns For each linear layer, its input size and output size.
 */
function projections(shape: TransformerTrainShape): Record<AdapterTarget, [number, number]> {
  const h = shape.width
  const q = shape.queryHeads * shape.headDim
  const kv = shape.kvHeads * shape.headDim
  const f = shape.ffnWidth
  return { q: [h, q], k: [h, kv], v: [h, kv], o: [q, h], gate: [h, f], up: [h, f], down: [f, h] }
}

/**
 * The weights of the linear layers of all blocks, $\Psi_{\text{lin}} = L(2h \cdot a\,d_h + 2h \cdot d_{kv} + 3hf)$:
 * the weights QLoRA quantises. The embeddings and normalisation weights are the rest of $\Psi$.
 *
 * @param shape The model.
 * @returns $\Psi_{\text{lin}}$.
 *
 * @example What QLoRA quantises in Llama 3.1 8B and 70B
 * for (const [name, shape] of [['8B', LLAMA_3_1_8B], ['70B', LLAMA_3_1_70B]]) {
 *   const linear = linearParameters(shape)
 *   print(`${name}: ${(linear / 1e9).toFixed(2)}e9 linear weights, ${((shape.parameters - linear) / 1e9).toFixed(2)}e9 embeddings`)
 * }
 */
export function linearParameters(shape: TransformerTrainShape): number {
  const sizes = Object.values(projections(shape))
  return shape.layers * sizes.reduce((total, [input, output]) => total + input * output, 0)
}

/**
 * The trainable parameters $\Psi_a$ of rank-$r$ LoRA adapters: a $d_{\text{out}} \times d_{\text{in}}$ layer gets
 * $\Bmat \in \reals^{d_{\text{out}} \times r}$ and $\Amat \in \reals^{r \times d_{\text{in}}}$, so $r(d_{\text{in}} +
 * d_{\text{out}})$ parameters, in every block (the bootstrapping note's "adapter size").
 *
 * @param shape The model.
 * @param rank The adapter rank $r$.
 * @param targets The linear layers that carry adapters (default all seven).
 * @returns $\Psi_a$.
 *
 * @example Rank 16 on every linear layer of Llama 3.1 8B
 * const all = adapterParameters(LLAMA_3_1_8B, 16)
 * print('all seven layers:', all, `(${((100 * all) / LLAMA_3_1_8B.parameters).toFixed(2)}% of the model)`)
 * print('query and value only:', adapterParameters(LLAMA_3_1_8B, 16, ['q', 'v']))
 */
export function adapterParameters(
  shape: TransformerTrainShape,
  rank: number,
  targets: readonly AdapterTarget[] = ALL_TARGETS,
): number {
  const sizes = projections(shape)
  return shape.layers * rank * targets.reduce((total, t) => total + sizes[t][0] + sizes[t][1], 0)
}

/**
 * The bytes a fine-tuning run holds on the device, by part.
 *
 * - **Model state.** Full fine-tuning keeps, per parameter, a bf16 weight and gradient, a 32-bit master copy and
 *   Adam's two 32-bit moments: $16\Psi$ (Rajbhandari et al., 2020; QLoRA note, "Model state"). LoRA keeps the base in
 *   bf16, $2\Psi$, and per adapter parameter a 32-bit weight, gradient and two moments, $16\Psi_a$. QLoRA stores the
 *   linear layers in NF4 at $(4 + 0.127)/8 = 0.516$ bytes per weight with double quantisation ($4.5/8$ without) and the
 *   rest in bf16 (QLoRA note, "Double quantisation" and "Worked example"). 8-bit Adam stores each moment in one byte.
 * - **Activations**, for $b$ sequences of $s$ tokens. Without checkpointing every layer stores
 *   $sbh(34 + 5as/h)$ bytes (Korthikanti et al., 2022), $34\,sbh$ with fused attention. With checkpointing each layer
 *   stores its input, $2sbh$, and the layer being recomputed holds its queries, keys and values, attention output,
 *   residual and normalised inputs and four feed-forward tensors, $(7.5h + 4f) \cdot 2$ bytes per token
 *   (bootstrapping note, step 3).
 * - **Logits**, $4bsV$ bytes in 32 bits, unless the loss is computed in chunks.
 *
 * @param shape The model.
 * @param setup The method, the adapters, the micro-batch, the sequence length, checkpointing, the logits, the
 *   optimiser, double quantisation and the attention kernel.
 * @param device The device whose memory (GB) `fits` is checked against; without it `fits` is true.
 * @returns The bytes of each part, their total, and whether the run fits.
 *
 * @example Full fine-tuning, LoRA and QLoRA of Llama 3.1 8B
 * for (const method of ['full', 'lora', 'qlora']) {
 *   const m = fineTuningMemory(LLAMA_3_1_8B, { method, microBatch: 8, sequence: 1024, checkpointing: true, logits: 'chunked' }, { memoryGb: 24 })
 *   print(`${method}: ${(m.total / 1e9).toFixed(1)} GB, fits a 24 GB device: ${m.fits}`)
 * }
 *
 * @example Where the memory of a LoRA run goes
 * const m = fineTuningMemory(LLAMA_3_1_8B, { method: 'lora', microBatch: 8, sequence: 1024, checkpointing: true, logits: 'full' })
 * for (const part of ['weights', 'adapterState', 'optimiserState', 'gradients', 'activations', 'logits'])
 *   print(`${part}: ${(m[part] / 1e9).toFixed(2)} GB`)
 */
export function fineTuningMemory(
  shape: TransformerTrainShape,
  setup: FineTuningSetup,
  device?: Pick<Device, 'memoryGb'>,
): FineTuningMemory {
  const { method, rank = 16, targets = ALL_TARGETS, optimiser = 'adam32', doubleQuant = true } = setup
  const psi = shape.parameters
  const moments = optimiser === 'adam8' ? 2 : 8
  let weights: number
  let adapterState = 0
  let optimiserState: number
  let gradients: number
  if (method === 'full') {
    weights = 2 * psi
    gradients = 2 * psi
    optimiserState = (4 + moments) * psi
  } else {
    const adapters = adapterParameters(shape, rank, targets)
    const linear = linearParameters(shape)
    const nf4 = (4 + (doubleQuant ? 8 / 64 + 32 / (64 * 256) : 32 / 64)) / 8
    weights = method === 'lora' ? 2 * psi : nf4 * linear + 2 * (psi - linear)
    adapterState = 4 * adapters
    gradients = 4 * adapters
    optimiserState = moments * adapters
  }
  const tokens = setup.microBatch * setup.sequence
  const h = shape.width
  let activations: number
  if (setup.checkpointing) activations = 2 * tokens * h * shape.layers + (7.5 * h + 4 * shape.ffnWidth) * 2 * tokens
  else {
    const attention = setup.attention === 'stored' ? (5 * shape.queryHeads * setup.sequence) / h : 0
    activations = tokens * h * (34 + attention) * shape.layers
  }
  const logits = setup.logits === 'full' ? 4 * tokens * shape.vocab : 0
  const total = weights + adapterState + optimiserState + gradients + activations + logits
  const resident = optimiser === 'paged' ? total - optimiserState : total
  return {
    weights,
    adapterState,
    optimiserState,
    gradients,
    activations,
    logits,
    total,
    fits: device ? resident <= device.memoryGb * 1e9 : true,
  }
}

/**
 * The FLOPs, time and price of a fine-tuning run. Per token: $2\Psi$ forward, $2\Psi$ for the gradients with respect to
 * activations, the weight gradients ($2\Psi$ for full fine-tuning, $2\Psi_a$ for adapters, which skip the frozen
 * weights), and $2\Psi$ to recompute the forward pass under checkpointing (bootstrapping note, step 3; Kaplan et al.,
 * 2020). So $6\Psi$ for full fine-tuning, a third more with checkpointing, and about $6\Psi$ for LoRA with
 * checkpointing. QLoRA counts as LoRA: its dequantisation slows each step but adds no matrix FLOPs. Counting the
 * embedding lookup in $\Psi$ overstates the arithmetic by a few per cent, on the safe side.
 *
 * @param shape The model.
 * @param setup The method, the adapters and checkpointing (the other fields do not change the FLOPs).
 * @param run The tokens per epoch, the epochs, the model FLOP utilisation, the hardware's peak FLOP/s and its price per
 *   hour.
 * @returns The FLOPs per token and in all, the seconds at that utilisation, and the price.
 *
 * @example The bootstrapping note's LoRA run
 * // 4,000 tickets of 630 tokens, 3 epochs, on an H100 (990 TFLOP/s dense bf16) at 30% utilisation and $3 an hour.
 * const setup = { method: 'lora', microBatch: 8, sequence: 1024, checkpointing: true, logits: 'chunked' }
 * const c = fineTuningCompute(LLAMA_3_1_8B, setup, { tokens: 4000 * 630, epochs: 3, mfu: 0.3, peakFlops: 990e12, pricePerHour: 3 })
 * print('FLOPs per token:', c.flopsPerToken, ' total:', c.flops)
 * print(`${(c.seconds / 60).toFixed(1)} minutes, $${c.cost.toFixed(2)}`)
 */
export function fineTuningCompute(
  shape: TransformerTrainShape,
  setup: Pick<FineTuningSetup, 'method' | 'rank' | 'targets' | 'checkpointing'>,
  run: FineTuningRun,
): FineTuningCompute {
  const psi = shape.parameters
  const weightGradients =
    setup.method === 'full' ? 2 * psi : 2 * adapterParameters(shape, setup.rank ?? 16, setup.targets ?? ALL_TARGETS)
  const flopsPerToken = 2 * psi + 2 * psi + weightGradients + (setup.checkpointing ? 2 * psi : 0)
  const flops = flopsPerToken * run.tokens * run.epochs
  const seconds = flops / (run.mfu * run.peakFlops)
  return { flopsPerToken, flops, seconds, cost: (seconds / 3600) * run.pricePerHour }
}
