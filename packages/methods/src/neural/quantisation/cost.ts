/**
 * Back-of-envelope cost of serving a transformer: the memory its weights and key–value cache take at a given precision,
 * and the decoding throughput a device's memory bandwidth and arithmetic allow (the roofline argument: generating one
 * token reads every weight once, so small-batch decoding is bound by bandwidth, not FLOPs). Kaplan et al. (2020) count
 * 2N FLOPs per token for N parameters; the KV cache holds 2 · layers · heads · head size values per token.
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
  /** Memory in GB (10⁹ bytes), bandwidth in GB/s and peak dense throughput in TFLOP/s at the compute precision. */
  memoryGb: number
  bandwidthGbs: number
  tflops: number
}

/** The memory of a served model. */
export interface ServingMemory {
  /** Bytes of weights: N · bits / 8. */
  weights: number
  /** Bytes of KV cache: 2 · layers · kvHeads · headDim · context · batch · kvBits / 8. */
  kvCache: number
  total: number
  /** Whether the total fits the device. */
  fits: boolean
}

/** The memory the weights and key–value cache take (bytes). */
export function servingMemory(model: TransformerShape, setup: ServingSetup, device?: Device): ServingMemory {
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

/** The roofline estimate of decoding throughput for a model, a serving setup and a device. */
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
