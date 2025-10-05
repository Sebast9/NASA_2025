"use client"

import { Loader2, Satellite } from "lucide-react"

export function LoadingPage() {
  return (
    <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-4">
      <div className="space-y-8 text-center">
        <div className="relative">
          <div className="glow-accent absolute inset-0 animate-pulse rounded-full" />
          <Satellite className="relative h-24 w-24 animate-spin text-accent" style={{ animationDuration: "3s" }} />
        </div>

        <div className="space-y-4">
          <h2 className="text-3xl font-bold">Analyzing Exoplanets</h2>
          <p className="text-lg text-muted-foreground">Our AI is processing your data...</p>
        </div>

        <div className="flex items-center justify-center gap-2">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
          <span className="text-sm text-muted-foreground">This may take a few moments</span>
        </div>
      </div>
    </div>
  )
}
