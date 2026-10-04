import type { ReactNode } from 'react'
import type { ThemePreference } from '@render/design/theme'
import { ThemeProvider } from '@render/design/ThemeProvider'
import { TooltipProvider } from '@render/ui/tooltip'

/** Everything render components expect around them: the theme and the tooltip provider. */
export function Providers({ children, theme }: { children: ReactNode; theme?: ThemePreference }) {
  return (
    <ThemeProvider initial={theme}>
      <TooltipProvider delay={300}>{children}</TooltipProvider>
    </ThemeProvider>
  )
}
