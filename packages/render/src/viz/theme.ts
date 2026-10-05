import { categorical, chrome, type Mode } from '../design/palette'

const FONT = "'Geist Variable', system-ui, sans-serif"

export const LINE_WIDTH = 2
export const MARKER_SIZE = 8

/** Plot margins around the grid, in pixels. Charts that size themselves (equal aspect) read these. */
export const GRID = { left: 56, right: 20, top: 32, bottom: 44 }

/**
 * Base ECharts option under every chart. Chart components add series and axes on top. Every chart styling decision
 * lives here so every lab figure looks the same.
 */
export function baseOption(mode: Mode) {
  const c = chrome(mode)
  const axis = {
    axisLine: { lineStyle: { color: c.axis } },
    axisTick: { show: false },
    axisLabel: { color: c.muted, fontSize: 10, fontFamily: FONT, hideOverlap: true },
    splitLine: { lineStyle: { color: c.grid, width: 1 } },
    nameTextStyle: { color: c.inkSecondary, fontSize: 11, fontFamily: FONT },
    nameLocation: 'middle' as const,
    nameGap: 28,
    axisPointer: {
      lineStyle: { color: c.muted, width: 1, type: 'dashed' as const },
      label: { backgroundColor: c.inkSecondary, color: c.surface, fontSize: 10, fontFamily: FONT, padding: [2, 4] },
    },
  }
  return {
    backgroundColor: 'transparent',
    color: [...categorical(mode)],
    textStyle: { fontFamily: FONT, color: c.ink, fontSize: 11 },
    // No animation at all. Entrance animation: a resize during ECharts' clip-path animation can leave lines hidden until
    // a repaint. Update tweening: when frames change faster than the tween (a Player at 120 steps/s, a drag), marks
    // glide towards targets that have already moved and lag the rest of the scene (the cart-pole's hinge behind its pole).
    animation: false,
    animationDuration: 0,
    animationDurationUpdate: 0,
    grid: { ...GRID, containLabel: false },
    legend: {
      top: 0,
      right: 0,
      // Plain: a long legend wraps onto more rows (the Plot keeps room for them) rather than paging.
      type: 'plain',
      icon: 'circle',
      itemWidth: 7,
      itemHeight: 7,
      itemGap: 8,
      padding: [4, 0, 4, 0],
      textStyle: { color: c.inkSecondary, fontSize: 11, fontFamily: FONT },
    },
    tooltip: {
      confine: true,
      backgroundColor: mode === 'dark' ? '#262624' : '#ffffff',
      borderColor: mode === 'dark' ? 'rgba(255,255,255,0.10)' : 'rgba(11,11,11,0.10)',
      textStyle: { color: c.ink, fontSize: 11, fontFamily: FONT },
      extraCssText: 'border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.08);',
    },
    xAxis: axis,
    yAxis: axis,
  }
}
