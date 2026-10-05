import type { CartPoleRender } from 'aifn-compute/foundation/contracts'
import { Curve, Plot, Points, useAxis } from '../viz'

/** Props of `CartPoleView`: one step's environment state, drawn by the environment's `cartpole` render spec. */
export type CartPoleViewProps<S> = {
  render: CartPoleRender<S>
  state: S
  title?: string
  /** An ending's tone: the cart, the pole and the track ends in the destructive red (a failure) or success green. */
  tone?: 'destructive' | 'success'
}

const CART_W = 0.5
const CART_H = 0.3

/**
 * A cart on its track with the pole hinged on top, in equal units, fitted to the track and the pole's height; the track
 * ends mark where the episode ends.
 */
export function CartPoleView<S>({ render, state, title, tone }: CartPoleViewProps<S>) {
  const L = render.trackLimit
  const x = useAxis({ label: 'x (m)', range: [-L - 0.3, L + 0.3] })
  const y = useAxis({ label: 'height (m)', range: [-0.1, CART_H + render.poleLength + 0.1], equal: x })
  const cx = render.cart(state)
  const th = render.angle(state)
  const hinge: [number, number] = [cx, CART_H]
  const tip: [number, number] = [cx + render.poleLength * Math.sin(th), CART_H + render.poleLength * Math.cos(th)]
  const box = {
    x: [cx - CART_W / 2, cx + CART_W / 2, cx + CART_W / 2, cx - CART_W / 2, cx - CART_W / 2],
    y: [0, 0, CART_H, CART_H, 0],
  }
  return (
    <Plot x={x} y={y} title={title}>
      <Curve name="track" x={[-L, L]} y={[0, 0]} muted silent />
      <Points
        name="track ends"
        x={[-L, L]}
        y={[0, 0]}
        muted={tone !== 'destructive'}
        tone={tone === 'destructive' ? tone : undefined}
      />
      <Curve name="cart" x={box.x} y={box.y} slot={0} tone={tone} width={3} silent />
      <Curve name="pole" x={[hinge[0], tip[0]]} y={[hinge[1], tip[1]]} emphasis={!tone} tone={tone} width={5} silent />
      <Points name="hinge" x={[hinge[0]]} y={[hinge[1]]} emphasis={!tone} tone={tone} />
    </Plot>
  )
}
