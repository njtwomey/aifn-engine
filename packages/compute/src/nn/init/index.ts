/**
 * `aifn-compute/nn/init`: initialisers (after torch.nn.init): Xavier, He, LeCun, normal and zeros.
 */

export {
  heNormal,
  heUniform,
  lecunUniform,
  normalInit,
  xavierNormal,
  xavierUniform,
  zerosInit,
  type Fans,
  type HeOptions,
  type Initialiser,
} from './init'
