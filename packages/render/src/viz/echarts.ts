/** Tree-shaken ECharts build. Register new chart types or components here, once. Only EChart.tsx imports this. */
import { BarChart, CustomChart, LineChart, ParallelChart, ScatterChart } from 'echarts/charts'
import {
  AxisPointerComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  ParallelComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer, SVGRenderer } from 'echarts/renderers'

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  CustomChart,
  ParallelChart,
  GridComponent,
  ParallelComponent,
  TooltipComponent,
  AxisPointerComponent,
  LegendComponent,
  VisualMapComponent,
  MarkLineComponent,
  MarkAreaComponent,
  SVGRenderer,
  CanvasRenderer,
])

export { echarts }
export type { EChartsCoreOption as EChartsOption } from 'echarts/core'
