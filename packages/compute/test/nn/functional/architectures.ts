/**
 * Layer specs of well-known image classifiers for the `networkCost` tests: a copy of the site's
 * `convolutional-network-architectures/_shared/architectures.ts` (AI Field Notes), which its figures and the
 * self-check use. Each model is a list of labelled stages walked from a 224 x 224 RGB input; where torchvision differs
 * from the paper, the torchvision variant is an option, so parameter counts can be compared with torchvision's.
 */
import type { LayerSpec, SequenceLayer } from 'aifn-compute/nn/functional'

export type Model = {
  id: string
  name: string
  year: number
  family: string
  input: [number, number, number]
  stages: SequenceLayer[]
}

const stage = (label: string, layers: LayerSpec[]): SequenceLayer => ({ kind: 'sequence', label, layers })
const conv = (out: number, kernel: number, stride = 1, padding: number | 'same' = 'same', bias = false): LayerSpec => ({
  kind: 'conv',
  out,
  kernel,
  stride,
  padding,
  bias,
})
const bn: LayerSpec = { kind: 'norm' }
const act: LayerSpec = { kind: 'activation' }
const head = (out = 1000): LayerSpec[] => [{ kind: 'globalPool' }, { kind: 'dense', out }]

/** Conv, batch norm, activation. */
const cba = (out: number, kernel: number, stride = 1): LayerSpec[] => [conv(out, kernel, stride), bn, act]

export function lenet5(): Model {
  return {
    id: 'lenet-5',
    name: 'LeNet-5',
    year: 1998,
    family: 'plain stack',
    input: [1, 32, 32],
    stages: [
      stage('C1, S2', [conv(6, 5, 1, 0, true), act, { kind: 'pool', kernel: 2 }]),
      stage('C3, S4', [conv(16, 5, 1, 0, true), act, { kind: 'pool', kernel: 2 }]),
      stage('C5', [conv(120, 5, 1, 0, true), act]),
      stage('F6, output', [{ kind: 'dense', out: 84 }, act, { kind: 'dense', out: 10 }]),
    ],
  }
}

/** AlexNet as in the paper: 227-pixel effective input, layers 2, 4 and 5 split into two groups (two GPUs). */
export function alexnet(): Model {
  const pool: LayerSpec = { kind: 'pool', kernel: 3, stride: 2 }
  const c = (out: number, k: number, s: number, p: number, groups = 1): LayerSpec => ({
    kind: 'conv',
    out,
    kernel: k,
    stride: s,
    padding: p,
    groups,
  })
  return {
    id: 'alexnet',
    name: 'AlexNet',
    year: 2012,
    family: 'plain stack',
    input: [3, 227, 227],
    stages: [
      stage('conv1', [c(96, 11, 4, 0), act, pool]),
      stage('conv2', [c(256, 5, 1, 2, 2), act, pool]),
      stage('conv3-5', [c(384, 3, 1, 1), act, c(384, 3, 1, 1, 2), act, c(256, 3, 1, 1, 2), act, pool]),
      stage('fc6-8', [
        { kind: 'dense', out: 4096 },
        act,
        { kind: 'dense', out: 4096 },
        act,
        { kind: 'dense', out: 1000 },
      ]),
    ],
  }
}

export function vgg16(): Model {
  const block = (out: number, n: number): LayerSpec[] => [
    ...Array.from({ length: n }, () => [conv(out, 3, 1, 1, true), act]).flat(),
    { kind: 'pool', kernel: 2 },
  ]
  return {
    id: 'vgg-16',
    name: 'VGG-16',
    year: 2014,
    family: 'plain stack',
    input: [3, 224, 224],
    stages: [
      stage('stage 1 (64)', block(64, 2)),
      stage('stage 2 (128)', block(128, 2)),
      stage('stage 3 (256)', block(256, 3)),
      stage('stage 4 (512)', block(512, 3)),
      stage('stage 5 (512)', block(512, 3)),
      stage('fc6-8', [
        { kind: 'dense', out: 4096 },
        act,
        { kind: 'dense', out: 4096 },
        act,
        { kind: 'dense', out: 1000 },
      ]),
    ],
  }
}

/** Widths of one Inception module: 1×1, 3×3 reduce, 3×3, 5×5 reduce, 5×5, pool projection. */
export type InceptionWidths = [number, number, number, number, number, number]

/**
 * One Inception module. `reduce: false` drops the 1×1 reductions (the naive module); `torchvision` adds batch norm
 * and uses 3×3 in the "5×5" branch, as torchvision's GoogLeNet does.
 */
export function inceptionModule(
  w: InceptionWidths,
  { reduce = true, torchvision = false }: { reduce?: boolean; torchvision?: boolean } = {},
): LayerSpec {
  const unit = (out: number, k: number): LayerSpec[] =>
    torchvision ? [conv(out, k), bn, act] : [conv(out, k, 1, 'same', true), act]
  const big = torchvision ? 3 : 5
  return {
    kind: 'concat',
    label: 'inception',
    branches: [
      unit(w[0], 1),
      reduce ? [...unit(w[1], 1), ...unit(w[2], 3)] : unit(w[2], 3),
      reduce ? [...unit(w[3], 1), ...unit(w[4], big)] : unit(w[4], big),
      [{ kind: 'pool', kernel: 3, stride: 1, padding: 1, ceil: true }, ...(reduce ? unit(w[5], 1) : [])],
    ],
  }
}

export const GOOGLENET_MODULES: Record<string, InceptionWidths> = {
  '3a': [64, 96, 128, 16, 32, 32],
  '3b': [128, 128, 192, 32, 96, 64],
  '4a': [192, 96, 208, 16, 48, 64],
  '4b': [160, 112, 224, 24, 64, 64],
  '4c': [128, 128, 256, 24, 64, 64],
  '4d': [112, 144, 288, 32, 64, 64],
  '4e': [256, 160, 320, 32, 128, 128],
  '5a': [256, 160, 320, 32, 128, 128],
  '5b': [384, 192, 384, 48, 128, 128],
}

/** GoogLeNet (Inception v1) without its auxiliary classifiers. */
export function googlenet({ torchvision = false }: { torchvision?: boolean } = {}): Model {
  const unit = (out: number, k: number, s = 1): LayerSpec[] =>
    torchvision ? [conv(out, k, s), bn, act] : [conv(out, k, s, 'same', true), act]
  const pool = (k = 3): LayerSpec => ({ kind: 'pool', kernel: k, stride: 2, ceil: true })
  const m = (id: string) => inceptionModule(GOOGLENET_MODULES[id], { torchvision })
  return {
    id: 'googlenet',
    name: 'GoogLeNet',
    year: 2014,
    family: 'multi-branch',
    input: [3, 224, 224],
    stages: [
      stage('stem', [...unit(64, 7, 2), pool(), ...unit(64, 1), ...unit(192, 3), pool()]),
      stage('inception 3a-3b', [m('3a'), m('3b'), pool()]),
      stage('inception 4a-4e', [m('4a'), m('4b'), m('4c'), m('4d'), m('4e'), pool(torchvision ? 2 : 3)]),
      stage('inception 5a-5b', [m('5a'), m('5b')]),
      stage('classifier', head()),
    ],
  }
}

/** A ResNet bottleneck (1×1, 3×3, 1×1 with expansion 4) or basic (two 3×3) block, with a projection when needed. */
export function resnetBlock(inChannels: number, width: number, stride: number, bottleneck: boolean): LayerSpec {
  const out = bottleneck ? 4 * width : width
  const body: LayerSpec[] = bottleneck
    ? [...cba(width, 1), ...cba(width, 3, stride), conv(out, 1), bn]
    : [...cba(width, 3, stride), conv(out, 3), bn]
  const shortcut: LayerSpec[] = stride !== 1 || inChannels !== out ? [conv(out, 1, stride, 0), bn] : []
  return { kind: 'add', label: 'residual block', branches: [body, shortcut] }
}

export function resnet(depth: 18 | 34 | 50 | 101 | 152): Model {
  const blocks = { 18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3] }[depth]
  const bottleneck = depth >= 50
  const widths = [64, 128, 256, 512]
  let channels = 64
  const stages: SequenceLayer[] = [
    stage('stem', [...cba(64, 7, 2), { kind: 'pool', kernel: 3, stride: 2, padding: 1 }]),
  ]
  widths.forEach((w, s) => {
    const layers: LayerSpec[] = []
    for (let b = 0; b < blocks[s]; b++) {
      layers.push(resnetBlock(channels, w, b === 0 && s > 0 ? 2 : 1, bottleneck))
      channels = bottleneck ? 4 * w : w
    }
    stages.push(stage(`stage ${s + 1} (${blocks[s]} blocks)`, layers))
  })
  stages.push(stage('classifier', head()))
  return {
    id: `resnet-${depth}`,
    name: `ResNet-${depth}`,
    year: 2015,
    family: 'residual',
    input: [3, 224, 224],
    stages,
  }
}

/** One DenseNet layer: BN, ReLU, 1×1 to `bottleneck`·k channels, BN, ReLU, 3×3 to k; its output is concatenated. */
export function denseLayer(growth: number, bottleneck = 4): LayerSpec {
  return {
    kind: 'concat',
    label: 'dense layer',
    branches: [[], [bn, act, conv(bottleneck * growth, 1), bn, act, conv(growth, 3)]],
  }
}

/** DenseNet-BC: blocks of dense layers with growth rate k, transitions halving channels and resolution. */
export function densenet(blocks: number[] = [6, 12, 24, 16], growth = 32, name = 'DenseNet-121'): Model {
  let channels = 2 * growth
  const stages: SequenceLayer[] = [
    stage('stem', [...cba(channels, 7, 2), { kind: 'pool', kernel: 3, stride: 2, padding: 1 }]),
  ]
  blocks.forEach((n, i) => {
    const layers: LayerSpec[] = Array.from({ length: n }, () => denseLayer(growth))
    channels += n * growth
    if (i < blocks.length - 1) {
      channels = Math.floor(channels / 2)
      layers.push(bn, act, conv(channels, 1), { kind: 'pool', kernel: 2 })
    } else layers.push(bn, act)
    stages.push(stage(`dense block ${i + 1} (${n} layers)`, layers))
  })
  stages.push(stage('classifier', head()))
  return { id: name.toLowerCase(), name, year: 2017, family: 'dense', input: [3, 224, 224], stages }
}

/** Channels rounded to a multiple of 8, never below 90% of the request (MobileNet and EfficientNet code). */
export function makeDivisible(v: number, divisor = 8): number {
  const r = Math.max(divisor, Math.floor((v + divisor / 2) / divisor) * divisor)
  return r < 0.9 * v ? r + divisor : r
}

/**
 * An inverted residual (MBConv) block: 1×1 expansion by `expand`, k×k depthwise, optional squeeze-and-excitation
 * with `seChannels` hidden units, linear 1×1 projection; identity shortcut when the shapes agree.
 */
export function invertedResidual(
  inChannels: number,
  out: number,
  {
    expand,
    kernel = 3,
    stride = 1,
    seChannels = 0,
  }: { expand: number; kernel?: number; stride?: number; seChannels?: number },
): LayerSpec {
  const hidden = inChannels * expand
  const body: LayerSpec[] = [
    ...(expand !== 1 ? cba(hidden, 1) : []),
    { kind: 'depthwise', kernel, stride, padding: 'same', bias: false },
    bn,
    act,
    ...(seChannels > 0
      ? [
          {
            kind: 'gate',
            label: 'squeeze-and-excitation',
            branch: [{ kind: 'globalPool' }, conv(seChannels, 1, 1, 0, true), act, conv(hidden, 1, 1, 0, true)],
          } satisfies LayerSpec,
        ]
      : []),
    conv(out, 1),
    bn,
  ]
  return stride === 1 && inChannels === out
    ? { kind: 'add', label: 'inverted residual', branches: [body, []] }
    : { kind: 'sequence', label: 'inverted residual', layers: body }
}

export function mobilenetV2(): Model {
  const settings: [number, number, number, number][] = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
  ]
  let channels = 32
  const stages: SequenceLayer[] = [stage('stem', cba(32, 3, 2))]
  settings.forEach(([t, c, n, s], i) => {
    const layers: LayerSpec[] = []
    for (let b = 0; b < n; b++) {
      layers.push(invertedResidual(channels, c, { expand: t, stride: b === 0 ? s : 1 }))
      channels = c
    }
    stages.push(stage(`stage ${i + 1} (${n} × t=${t}, ${c} ch)`, layers))
  })
  stages.push(stage('head', [...cba(1280, 1), ...head()]))
  return { id: 'mobilenet-v2', name: 'MobileNetV2', year: 2018, family: 'efficient', input: [3, 224, 224], stages }
}

/** EfficientNet-B0 to B7: width, depth and resolution coefficients of the reference implementation. */
export const EFFICIENTNET_SCALES: Record<string, [number, number, number]> = {
  B0: [1.0, 1.0, 224],
  B1: [1.0, 1.1, 240],
  B2: [1.1, 1.2, 260],
  B3: [1.2, 1.4, 300],
  B4: [1.4, 1.8, 380],
  B5: [1.6, 2.2, 456],
  B6: [1.8, 2.6, 528],
  B7: [2.0, 3.1, 600],
}

/** EfficientNet from the B0 baseline, with width multiplier, depth multiplier and input resolution. */
export function efficientnet(width = 1, depth = 1, resolution = 224, name = 'EfficientNet-B0'): Model {
  // expand, kernel, stride, in, out, repeats (B0)
  const settings: [number, number, number, number, number, number][] = [
    [1, 3, 1, 32, 16, 1],
    [6, 3, 2, 16, 24, 2],
    [6, 5, 2, 24, 40, 2],
    [6, 3, 2, 40, 80, 3],
    [6, 5, 1, 80, 112, 3],
    [6, 5, 2, 112, 192, 4],
    [6, 3, 1, 192, 320, 1],
  ]
  const ch = (c: number) => makeDivisible(c * width)
  let channels = ch(32)
  const stages: SequenceLayer[] = [stage('stem', cba(channels, 3, 2))]
  settings.forEach(([e, k, s, , c, n], i) => {
    const repeats = Math.ceil(n * depth)
    const out = ch(c)
    const layers: LayerSpec[] = []
    for (let b = 0; b < repeats; b++) {
      layers.push(
        invertedResidual(channels, out, {
          expand: e,
          kernel: k,
          stride: b === 0 ? s : 1,
          seChannels: Math.max(1, Math.floor(channels / 4)),
        }),
      )
      channels = out
    }
    stages.push(stage(`stage ${i + 1} (${repeats} × MBConv${e} ${k}×${k})`, layers))
  })
  stages.push(stage('head', [...cba(ch(1280), 1), ...head()]))
  return { id: name.toLowerCase(), name, year: 2019, family: 'efficient', input: [3, resolution, resolution], stages }
}

/** One ConvNeXt block: 7×7 depthwise, layer norm, 1×1 to 4C, GELU, 1×1 back to C, layer scale, identity shortcut. */
export function convnextBlock(dim: number, kernel = 7): LayerSpec {
  return {
    kind: 'add',
    label: 'ConvNeXt block',
    branches: [
      [
        { kind: 'depthwise', kernel, padding: 'same', bias: true },
        { kind: 'norm' },
        conv(4 * dim, 1, 1, 0, true),
        act,
        conv(dim, 1, 1, 0, true),
        { kind: 'channelScale' },
      ],
      [],
    ],
  }
}

export function convnext(depths = [3, 3, 9, 3], dims = [96, 192, 384, 768], name = 'ConvNeXt-T'): Model {
  const stages: SequenceLayer[] = [stage('stem (4×4 patchify)', [conv(dims[0], 4, 4, 0, true), { kind: 'norm' }])]
  dims.forEach((d, i) => {
    const down: LayerSpec[] = i > 0 ? [{ kind: 'norm' }, conv(d, 2, 2, 0, true)] : []
    stages.push(
      stage(`stage ${i + 1} (${depths[i]} blocks)`, [
        ...down,
        ...Array.from({ length: depths[i] }, () => convnextBlock(d)),
      ]),
    )
  })
  stages.push(stage('classifier', [{ kind: 'globalPool' }, { kind: 'norm' }, { kind: 'dense', out: 1000 }]))
  return { id: name.toLowerCase(), name, year: 2022, family: 'modern', input: [3, 224, 224], stages }
}

/** The models of the architecture explorer, in time order. */
export function modelZoo(): Model[] {
  return [
    lenet5(),
    alexnet(),
    vgg16(),
    googlenet(),
    resnet(18),
    resnet(50),
    resnet(152),
    densenet(),
    mobilenetV2(),
    efficientnet(),
    convnext(),
  ]
}

/** One rung of the ResNet-50 to ConvNeXt-T modernisation (Liu et al. 2022, section 2): a label and the network. */
export type LadderStep = { label: string; change: string; model: Model }

type LadderOptions = {
  depths: number[]
  stem: 'resnet' | 'patchify'
  /** ResNet-style bottleneck (wide outside), inverted (wide inside), or inverted with the depthwise layer first. */
  block: 'bottleneck' | 'inverted' | 'inverted-dw-first'
  depthwise: boolean
  /** Bottleneck width (inner for `bottleneck`, outer for the inverted blocks) of stage 1. */
  width: number
  kernel: number
  /** One norm per block instead of one per convolution. */
  fewerNorms: boolean
  separateDownsampling: boolean
}

function ladderModel(o: LadderOptions, name: string): Model {
  const widths = [o.width, 2 * o.width, 4 * o.width, 8 * o.width]
  const outer = (w: number) => (o.block === 'bottleneck' ? 4 * w : w)
  const norm = (keep: boolean): LayerSpec[] => (o.fewerNorms && !keep ? [] : [bn])
  const spatial = (channels: number, stride: number): LayerSpec =>
    o.depthwise
      ? { kind: 'depthwise', kernel: o.kernel, stride, padding: 'same', bias: o.separateDownsampling }
      : conv(channels, o.kernel, stride)
  const block = (inC: number, w: number, stride: number): LayerSpec => {
    const out = outer(w)
    let body: LayerSpec[]
    if (o.block === 'bottleneck')
      body = [conv(w, 1), ...norm(false), act, spatial(w, stride), ...norm(false), act, conv(out, 1), ...norm(true)]
    else if (o.block === 'inverted')
      body = [
        conv(4 * w, 1),
        ...norm(false),
        act,
        spatial(4 * w, stride),
        ...norm(false),
        act,
        conv(out, 1),
        ...norm(true),
      ]
    else
      body = [
        spatial(inC, stride),
        ...norm(true),
        conv(4 * w, 1, 1, 'same', o.fewerNorms),
        act,
        conv(out, 1, 1, 'same', o.fewerNorms),
        ...(o.separateDownsampling ? [{ kind: 'channelScale' } as LayerSpec] : norm(false)),
      ]
    const shortcut: LayerSpec[] = stride !== 1 || inC !== out ? [conv(out, 1, stride, 0), bn] : []
    return { kind: 'add', label: 'block', branches: [body, shortcut] }
  }
  const stemOut = o.block === 'bottleneck' && !o.depthwise ? 64 : o.width
  const stages: SequenceLayer[] = [
    stage(
      'stem',
      o.stem === 'resnet'
        ? [...cba(stemOut, 7, 2), { kind: 'pool', kernel: 3, stride: 2, padding: 1 }]
        : [conv(stemOut, 4, 4, 0, o.separateDownsampling), bn],
    ),
  ]
  let channels = stemOut
  widths.forEach((w, s) => {
    const layers: LayerSpec[] = []
    if (o.separateDownsampling && s > 0) {
      layers.push(bn, conv(outer(w), 2, 2, 0, true))
      channels = outer(w)
    }
    for (let b = 0; b < o.depths[s]; b++) {
      const stride = !o.separateDownsampling && b === 0 && s > 0 ? 2 : 1
      layers.push(block(channels, w, stride))
      channels = outer(w)
    }
    stages.push(stage(`stage ${s + 1} (${o.depths[s]} blocks)`, layers))
  })
  stages.push(
    stage('classifier', [
      { kind: 'globalPool' },
      ...(o.separateDownsampling ? [bn] : []),
      { kind: 'dense', out: 1000 },
    ]),
  )
  return { id: name, name, year: 2022, family: 'modern', input: [3, 224, 224], stages }
}

/** The modernisation steps from ResNet-50 to ConvNeXt-T, each a full network whose cost `networkCost` gives. */
export function modernisationLadder(): LadderStep[] {
  const base: LadderOptions = {
    depths: [3, 4, 6, 3],
    stem: 'resnet',
    block: 'bottleneck',
    depthwise: false,
    width: 64,
    kernel: 3,
    fewerNorms: false,
    separateDownsampling: false,
  }
  const steps: [string, string, Partial<LadderOptions>][] = [
    ['ResNet-50', 'bottleneck blocks (3, 4, 6, 3), 7×7 stride-2 stem and max pool', {}],
    ['stage ratio', 'blocks per stage (3, 4, 6, 3) → (3, 3, 9, 3)', { depths: [3, 3, 9, 3] }],
    ['patchify stem', '7×7 stride-2 convolution and max pool → one 4×4 stride-4 convolution', { stem: 'patchify' }],
    ['depthwise', 'the 3×3 convolution becomes depthwise (one filter per channel)', { depthwise: true }],
    ['wider', 'bottleneck width 64 → 96 per stage-1 block', { width: 96 }],
    ['inverted bottleneck', 'narrow-wide-narrow: 96 → 384 → 96 instead of 384 → 96 → 384', { block: 'inverted' }],
    ['depthwise first', 'the depthwise layer moves before the 1×1 expansion', { block: 'inverted-dw-first' }],
    ['7×7 kernel', 'depthwise kernel 3×3 → 7×7', { kernel: 7 }],
    [
      'micro design',
      'ReLU → GELU, one activation and one norm per block, batch norm → layer norm',
      { fewerNorms: true },
    ],
    [
      'ConvNeXt-T',
      'separate 2×2 stride-2 downsampling layers between stages, layer scale',
      { separateDownsampling: true },
    ],
  ]
  let options = base
  return steps.map(([label, change, patch]) => {
    options = { ...options, ...patch }
    return { label, change, model: ladderModel(options, label) }
  })
}
