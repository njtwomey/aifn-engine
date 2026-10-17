/**
 * Receptive fields of convolutional networks: the theoretical field of a chain of convolutions and poolings along one
 * axis, the effective field of one output unit of any differentiable network, and the share of a dilated kernel's taps
 * that land inside a zero-padded map.
 *
 * Along one axis, a layer with kernel $k$, stride $s$, dilation $d$ and padding $p$ has the dilated extent
 * $e = d(k - 1) + 1$. With jump $j$ (input pixels between adjacent units), size $r$ (input pixels a unit can see) and
 * start $c$ (the input coordinate of the centre of the first unit), from $j_0 = r_0 = 1$ and $c_0 = 0$ at the input,
 *
 * $$
 * j_l = j_{l-1} s_l, \qquad r_l = r_{l-1} + (e_l - 1) j_{l-1}, \qquad
 * c_l = c_{l-1} + \left(\tfrac{e_l - 1}{2} - p_l\right) j_{l-1}
 * $$
 *
 * (Araujo, Norris and Sim, 2019, "Computing Receptive Fields of Convolutional Neural Networks", Distill; Dumoulin and
 * Visin, 2016, for the dilated extent). `networkCost` grows its per-axis fields with the same step, `growField`. The
 * effective receptive field is the gradient of one output unit with respect to the input (Luo, Li, Urtasun and Zemel,
 * 2016, "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks", §2). The tap coverage
 * follows Chen, Papandreou, Schroff and Adam (2017), "Rethinking Atrous Convolution for Semantic Image Segmentation",
 * §3.3 and figure 4.
 */

import { grad } from 'aifn-compute/foundation/autodiff'
import { DomainError } from 'aifn-compute/foundation/errors'
import { get, type Value } from 'aifn-compute/foundation/tensor'

/** One layer along one axis, for `receptiveField`. A pooling window counts as a kernel. */
export type ReceptiveFieldLayer = {
  /** Kernel taps $k$. */
  kernel: number
  /** Step $s$ between outputs. Default 1. */
  stride?: number
  /** Spacing $d$ between taps. Default 1. */
  dilation?: number
  /** Zeros $p$ added on each side. Default 0. */
  padding?: number
}

/** The receptive field after a layer, along one axis, in input pixels. */
export type ReceptiveFieldStep = {
  /** Input pixels that can change one unit: $r$. */
  size: number
  /** Input pixels between adjacent units: $j$, the product of the strides so far. */
  jump: number
  /** Input coordinate of the centre of the first unit (input pixel $i$ sits at $i$). */
  start: number
}

/**
 * One step of the receptive-field recurrence along one axis: the size and jump after a window of dilated extent $e$
 * and stride $s$, $r + (e - 1)j$ and $js$. Shared by `receptiveField` and `networkCost`, so the two cannot drift.
 *
 * @param size The size $r$ of the field before the layer, in input pixels (Infinity stays Infinity).
 * @param jump The jump $j$ before the layer, in input pixels.
 * @param extent The layer's dilated extent $e = d(k - 1) + 1$, in taps of its input.
 * @param stride The layer's stride $s$.
 * @returns The size and jump after the layer.
 */
export function growField(size: number, jump: number, extent: number, stride: number): { size: number; jump: number } {
  return { size: size + (extent - 1) * jump, jump: jump * stride }
}

/**
 * The receptive field after each layer of a chain, along one axis (Araujo et al., 2019). Call it once per axis for a
 * 2-D network.
 *
 * @param layers The layers in order, input first, each with its kernel taps and optional stride, dilation and
 *   padding.
 * @returns One entry per layer, preceded by the input's own (size 1, jump 1, start 0): `layers.length + 1` entries.
 *
 * @example Two 3-tap layers, then a 2-wide stride-2 pool
 * const steps = receptiveField([{ kernel: 3 }, { kernel: 3 }, { kernel: 2, stride: 2 }])
 * print('sizes =', steps.map((r) => r.size))
 * print('jumps =', steps.map((r) => r.jump))
 *
 * @example Dilations 1, 2, 4, 8 double the field at every layer
 * print('sizes =', receptiveField([1, 2, 4, 8].map((d) => ({ kernel: 3, dilation: d }))).map((r) => r.size))
 */
export function receptiveField(layers: readonly ReceptiveFieldLayer[]): ReceptiveFieldStep[] {
  const out: ReceptiveFieldStep[] = [{ size: 1, jump: 1, start: 0 }]
  for (const layer of layers) {
    const prev = out[out.length - 1]
    const extent = (layer.dilation ?? 1) * (layer.kernel - 1) + 1
    const { size, jump } = growField(prev.size, prev.jump, extent, layer.stride ?? 1)
    out.push({ size, jump, start: prev.start + ((extent - 1) / 2 - (layer.padding ?? 0)) * prev.jump })
  }
  return out
}

/**
 * The effective receptive field of one output unit: the gradient of `f(x)` at `index` with respect to `x`, by reverse
 * mode (Luo et al., 2016). For a network linear in its input it does not depend on `x`; otherwise it is the field at
 * that input.
 *
 * @param f The network, from an input tensor to an output tensor; differentiable.
 * @param x The input at which to take the gradient.
 * @param index The full multi-index of the output unit, e.g. `[0, 0, i]` for an `[N, C, L]` output.
 * @returns The gradient of the unit with respect to every input value, shaped like `x`.
 *
 * @example Two uniform 3-tap layers: the kernel convolved with itself
 * const w = tensor([[[1 / 3, 1 / 3, 1 / 3]]])
 * const net = (x) => conv1d(conv1d(x, w, { padding: 1 }), w, { padding: 1 })
 * print('9 x field =', mul(9, effectiveReceptiveField(net, zeros([1, 1, 9]), [0, 0, 4])))
 */
export function effectiveReceptiveField(f: (x: Value) => Value, x: Value, index: readonly number[]): Value {
  return grad((v: Value) => get(f(v), ...index))(x) as Value
}

/**
 * The share of positions of an $n \times n$ zero-padded map at which exactly $c$ of the $k^2$ taps of a $k \times k$
 * kernel with dilation $d$, centred on the position, land inside the map, for $c = 0, \dots, k^2$. Along one axis tap
 * $u \in \{0, \dots, k - 1\}$ reads $i + d(u - (k - 1)/2)$; the count at $(i, j)$ is the product of the two axis
 * counts, so the cost is $O(nk + n^2)$. With a rate near the map size only the centre tap stays inside, which is why
 * ASPP adds an image-level pooling branch (Chen et al., 2017). Throws `DomainError` for an even kernel side.
 *
 * @param n The side of the square feature map, in positions.
 * @param kernel The kernel side $k$; odd, so the kernel has a centre tap.
 * @param dilation The dilation (atrous rate) $d \ge 1$.
 * @returns `shares[c]` for $c = 0, \dots, k^2$; the shares sum to 1.
 *
 * @example A 3 x 3 kernel on a 65 x 65 map at rates 1 and 64
 * const r1 = dilatedTapCoverage(65, 3, 1)
 * const r64 = dilatedTapCoverage(65, 3, 64)
 * print('rate 1: share with 9 taps =', r1[9])
 * print('rate 64: share with 9 taps =', r64[9], ', with 1 tap =', r64[1])
 */
export function dilatedTapCoverage(n: number, kernel: number, dilation: number): number[] {
  if (kernel % 2 !== 1) throw new DomainError('dilatedTapCoverage', 'dilatedTapCoverage: the kernel side must be odd')
  const half = (kernel - 1) / 2
  const inside = Array.from({ length: n }, (_, i) => {
    let c = 0
    for (let u = 0; u < kernel; u++) {
      const q = i + dilation * (u - half)
      if (q >= 0 && q < n) c++
    }
    return c
  })
  const counts = Array.from({ length: kernel * kernel + 1 }, () => 0)
  for (const a of inside) for (const b of inside) counts[a * b]++
  return counts.map((c) => c / (n * n))
}

/**
 * The mean number of taps inside the map per position, $\sum_c c \cdot \text{shares}_c$ over `dilatedTapCoverage`.
 *
 * @param n The side of the square feature map.
 * @param kernel The kernel side $k$ (odd).
 * @param dilation The dilation $d \ge 1$.
 * @returns The mean count of taps inside the map, between 1 and $k^2$.
 *
 * @example Fewer taps fall inside as the rate grows
 * for (const d of [1, 2, 6, 12]) print(`rate ${d}:`, meanValidTaps(33, 3, d))
 */
export function meanValidTaps(n: number, kernel: number, dilation: number): number {
  return dilatedTapCoverage(n, kernel, dilation).reduce((a, s, c) => a + s * c, 0)
}
