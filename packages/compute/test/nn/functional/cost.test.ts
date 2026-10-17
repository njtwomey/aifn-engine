/**
 * aifn-compute/nn/functional convolution arithmetic: `convolutionCost` and `convolutionStackCost` against published
 * layer counts (AlexNet, VGG-16, ResNet-50, MobileNet, ResNeXt, C3D, the transposed sizes of Dumoulin and Visin), and
 * `networkCost` against torchvision's published parameter counts and GFLOPs for whole networks, the ResNet-50 to
 * ConvNeXt-T ladder of Liu et al. (2022) and EfficientNet B0 to B7 (the site's self-checks).
 */
import { describe, expect, it } from 'vitest'
import { AifnError } from 'aifn-compute/foundation/errors'
import {
  convolutionCost,
  convolutionStackCost,
  layerCost,
  networkCost,
  type ChainedLayerSpec,
  type LayerSpec,
} from 'aifn-compute/nn/functional'
import {
  convnext,
  densenet,
  EFFICIENTNET_SCALES,
  efficientnet,
  googlenet,
  mobilenetV2,
  modernisationLadder,
  resnet,
  vgg16,
  type Model,
} from './architectures'

describe('convolutionCost', () => {
  it('matches published layer counts', () => {
    const alex = convolutionCost({ inChannels: 3, outChannels: 96, kernel: 11, stride: 4, input: [227, 227] })
    expect(alex.outputShape).toEqual([55, 55])
    expect(alex.params).toBe(34944)
    const vgg: ChainedLayerSpec[] = []
    for (const [n, c] of [
      [2, 64],
      [2, 128],
      [3, 256],
      [3, 512],
      [3, 512],
    ] as const)
      for (let i = 0; i < n; i++) vgg.push({ kernel: 3, outChannels: c, padding: 1 })
    expect(convolutionStackCost({ channels: 3, shape: [224, 224] }, vgg).params).toBe(14714688)
    expect(
      convolutionCost({ inChannels: 3, outChannels: 64, kernel: 3, padding: 1, input: [16, 112, 112] }).params,
    ).toBe(5248)
  })

  it('counts a bottleneck, a depthwise separable pair and a grouped layer', () => {
    const bottleneck = convolutionStackCost({ channels: 256, shape: [56, 56] }, [
      { kernel: 1, outChannels: 64, bias: false },
      { kernel: 3, outChannels: 64, padding: 1, bias: false },
      { kernel: 1, outChannels: 256, bias: false },
    ])
    expect(bottleneck.params).toBe(69632)
    expect(bottleneck.multiplies).toBe(69632 * 56 * 56)
    expect(bottleneck.layers.map((l) => l.name)).toEqual(['layer 1', 'layer 2', 'layer 3'])
    const wide = { inChannels: 256, outChannels: 256, bias: false }
    const full = convolutionCost({ ...wide, kernel: 3, padding: 1, input: [56, 56] })
    expect(full.weights).toBe(589824)
    const dw = convolutionCost({ ...wide, kind: 'depthwise', kernel: 3, padding: 1, input: [14, 14] })
    const pw = convolutionCost({ ...wide, kernel: 1, input: [14, 14] })
    expect(dw.weights).toBe(2304)
    expect(Math.round((589824 / (dw.weights + pw.weights)) * 100) / 100).toBe(8.69)
    const grouped = convolutionCost({
      inChannels: 128,
      outChannels: 128,
      kernel: 3,
      groups: 32,
      padding: 1,
      input: [56, 56],
    })
    expect(grouped.weights).toBe(4608)
    expect(grouped.groups).toBe(32)
  })

  it('gives the docstring example: a ResNet 3 x 3 layer at 56 x 56', () => {
    const c = convolutionCost({ inChannels: 64, outChannels: 64, kernel: 3, padding: 1, input: [56, 56] })
    expect([c.params, c.weights, c.biases, c.multiplies]).toEqual([36928, 36864, 64, 115605504])
    expect(c.outputShape).toEqual([56, 56])
  })

  it('sizes transposed and dilated layers', () => {
    const t2 = convolutionCost({
      kind: 'transposed',
      inChannels: 8,
      outChannels: 4,
      kernel: 4,
      stride: 2,
      padding: 1,
      input: [7, 7],
    })
    expect(t2.outputShape).toEqual([14, 14])
    const t1 = convolutionCost({ kind: 'transposed', inChannels: 1, outChannels: 1, kernel: 3, stride: 2, input: [4] })
    expect(t1.outputShape).toEqual([9])
    expect(t1.multiplies).toBe(12)
    const dilated = convolutionCost({ inChannels: 1, kernel: 3, dilation: 2, padding: 'same', input: [32] })
    expect(dilated.outputShape).toEqual([32])
  })

  it("pads 'same' as conv does: ceil((e - 1)/2) before, floor after, keeping ceil(n / s)", () => {
    const out = (kernel: number, stride: number, padding: number | 'same', n: number) =>
      convolutionCost({ inChannels: 1, kernel, stride, padding, input: [n] }).outputShape[0]
    expect(out(4, 1, 'same', 32)).toBe(32)
    expect(out(4, 2, 'same', 33)).toBe(17)
    expect(out(5, 2, 'same', 112)).toBe(56)
    expect(out(3, 2, 'same', 112)).toBe(56)
    // The candidate's symmetric rule, floor((e - 1)/2) on both sides, is the explicit padding.
    expect(out(4, 1, 1, 32)).toBe(31)
  })

  it('throws for groups that do not divide the channels and for a kernel that does not fit', () => {
    const indivisible = { inChannels: 6, outChannels: 4, kernel: 3, groups: 4, input: [8] }
    expect(() => convolutionCost(indivisible)).toThrow(AifnError)
    expect(() => convolutionCost({ inChannels: 1, kernel: 5, input: [3] })).toThrow(AifnError)
    expect(() => convolutionCost({ inChannels: 1, kernel: [3, 3], input: [8] })).toThrow(AifnError)
  })
})

describe('layerCost and networkCost', () => {
  it('cost single layers by hand', () => {
    const c = layerCost({ kind: 'conv', out: 128, kernel: 3, padding: 1 }, [64, 56, 56])
    expect(c.params).toBe(73856)
    expect(c.multiplies).toBe(9 * 64 * 128 * 56 * 56)
    const a = layerCost({ kind: 'conv', out: 96, kernel: 11, stride: 4 }, [3, 227, 227])
    expect(a.params).toBe(34944)
    expect(a.outputShape[1]).toBe(55)
    const stem = layerCost({ kind: 'conv', out: 64, kernel: 7, stride: 2, padding: 3 }, [3, 224, 224])
    expect(stem.outputShape[1]).toBe(112)
    expect(layerCost({ kind: 'depthwise', kernel: 3, bias: false, padding: 1 }, [256, 14, 14]).params).toBe(2304)
    expect(layerCost({ kind: 'transposed', out: 1, kernel: 3, stride: 2 }, [1, 4, 4]).outputShape[1]).toBe(9)
    expect(layerCost({ kind: 'conv', out: 64, kernel: 3, groups: 4, bias: false }, [64, 8, 8]).params).toBe(
      (9 * 64 * 64) / 4,
    )
  })

  it('grows the receptive field as Araujo et al.', () => {
    const three: LayerSpec[] = [1, 1, 1].map(() => ({ kind: 'conv', out: 1, kernel: 3, padding: 1 }))
    expect(networkCost(three, [1, 32, 32]).receptiveField.size[0]).toBe(7)
    const strided: LayerSpec[] = [{ kind: 'conv', out: 1, kernel: 3, stride: 2, padding: 1 }, ...three.slice(1)]
    expect(networkCost(strided, [1, 32, 32]).receptiveField.size[0]).toBe(11)
    const vgg = networkCost(vgg16().stages.slice(0, 5), [3, 224, 224]).receptiveField
    expect(vgg.size[0]).toBe(212)
    expect(vgg.jump[0]).toBe(32)
    expect(networkCost(vgg16().stages, [3, 224, 224]).receptiveField.size[0]).toBe(Infinity)
  })

  /** torchvision's AlexNet: one tower, 64-192-384-256-256 channels. */
  const tvAlexnet: Model = {
    id: 'tv-alexnet',
    name: 'AlexNet (torchvision)',
    year: 2014,
    family: 'plain',
    input: [3, 224, 224],
    stages: [
      {
        kind: 'sequence',
        layers: [
          { kind: 'conv', out: 64, kernel: 11, stride: 4, padding: 2 },
          { kind: 'pool', kernel: 3, stride: 2 },
          { kind: 'conv', out: 192, kernel: 5, padding: 2 },
          { kind: 'pool', kernel: 3, stride: 2 },
          { kind: 'conv', out: 384, kernel: 3, padding: 1 },
          { kind: 'conv', out: 256, kernel: 3, padding: 1 },
          { kind: 'conv', out: 256, kernel: 3, padding: 1 },
          { kind: 'pool', kernel: 3, stride: 2 },
          { kind: 'dense', out: 4096 },
          { kind: 'dense', out: 4096 },
          { kind: 'dense', out: 1000 },
        ],
      },
    ],
  }

  it.each([
    [tvAlexnet, 61_100_840, 0.71e9, undefined],
    [vgg16(), 138_357_544, 15.47e9, 16],
    [googlenet({ torchvision: true }), 6_624_904, 1.5e9, 22],
    [resnet(18), 11_689_512, 1.81e9, 18],
    [resnet(50), 25_557_032, 4.09e9, 50],
    [resnet(152), 60_192_808, 11.51e9, 152],
    [densenet(), 7_978_856, 2.83e9, 121],
    [mobilenetV2(), 3_504_872, 0.3e9, 53],
    [efficientnet(), 5_288_548, 0.39e9, 50],
    [convnext(), 28_589_128, 4.46e9, 59],
  ] as [Model, number, number, number | undefined][])(
    '%# matches torchvision: parameters exactly, multiply-accumulates within 2%',
    (m, params, macs, depth) => {
      const r = networkCost(m.stages, m.input)
      expect(r.params, m.name).toBe(params)
      expect(Math.abs(r.multiplies - macs) / macs, m.name).toBeLessThan(0.02)
      if (depth !== undefined) expect(r.depth, m.name).toBe(depth)
      expect(r.outputShape).toEqual([1000])
      expect(r.layers.length).toBe(m.stages.length)
    },
  )

  it('gives ResNet-50 4.089e9 multiply-accumulates', () => {
    expect(networkCost(resnet(50).stages, [3, 224, 224]).multiplies).toBe(4_089_184_256)
  })

  it('climbs the ResNet-50 to ConvNeXt-T ladder of Liu et al. (2022)', () => {
    const reported = [4.1, 4.5, 4.4, NaN, 5.3, 4.6, 4.1, 4.2, 4.2, 4.5]
    const steps = modernisationLadder()
    steps.forEach((step, i) => {
      const r = networkCost(step.model.stages, step.model.input)
      if (Number.isNaN(reported[i])) return
      expect(Math.abs(r.multiplies / 1e9 - reported[i]) / reported[i], step.label).toBeLessThan(0.03)
    })
    const last = steps.at(-1)!.model
    expect(networkCost(last.stages, last.input).params).toBe(28_589_128)
  })

  it('scales EfficientNet B0 to B7 as Tan and Le (2019), Table 2', () => {
    // Millions of parameters and billions of FLOPs (multiply-accumulates) as the paper prints them.
    const table: Record<string, [number, number]> = {
      B0: [5.3, 0.39],
      B1: [7.8, 0.7],
      B2: [9.2, 1.0],
      B3: [12, 1.8],
      B4: [19, 4.2],
      B5: [30, 9.9],
      B6: [43, 19],
      B7: [66, 37],
    }
    for (const [name, [w, d, res]] of Object.entries(EFFICIENTNET_SCALES)) {
      const r = networkCost(efficientnet(w, d, res, name).stages, [3, res, res])
      const [params, flops] = table[name]
      expect(Math.abs(r.params / 1e6 - params) / params, name).toBeLessThan(0.03)
      expect(Math.abs(r.multiplies / 1e9 - flops) / flops, name).toBeLessThan(0.05)
    }
    const b3 = networkCost(efficientnet(1.2, 1.4, 300, 'B3').stages, [3, 300, 300])
    expect(b3.params).toBe(12_233_232)
  })

  it('throws when branch shapes disagree', () => {
    const bad: LayerSpec = {
      kind: 'add',
      branches: [[{ kind: 'conv', out: 8, kernel: 3 }], []],
    }
    expect(() => layerCost(bad, [8, 16, 16])).toThrow(AifnError)
  })
})
