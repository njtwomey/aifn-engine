import type { ReactNode } from 'react'
import type { ThemePreference } from '../design/theme'
import { ThemeProvider } from '../design/ThemeProvider'
import { TooltipProvider } from '../ui/tooltip'

/** Everything render components expect around them: the theme and the tooltip provider. */
export function Providers({ children, theme }: { children: ReactNode; theme?: ThemePreference }) {
  return (
    <ThemeProvider initial={theme}>
      <TooltipProvider delay={300}>{children}</TooltipProvider>
    </ThemeProvider>
  )
}
