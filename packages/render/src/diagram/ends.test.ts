import { describe, expect, it } from 'vitest'
import { parseEnd } from './ends'

describe('parseEnd', () => {
  it('reads a final port suffix', () => {
    expect(parseEnd('a:n')).toEqual({ id: 'a', side: 'n' })
    expect(parseEnd('a')).toEqual({ id: 'a' })
  })
  it('keeps a colon that is not a port in the id', () => {
    expect(parseEnd('x:1')).toEqual({ id: 'x:1' })
    expect(parseEnd('y:t:e')).toEqual({ id: 'y:t', side: 'e' })
    expect(parseEnd('f:east')).toEqual({ id: 'f:east' })
  })
})
