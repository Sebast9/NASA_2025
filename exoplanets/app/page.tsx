"use client"

import { useState } from "react"
import { Starfield } from "@/components/starfield"
import { LandingPage } from "@/components/landing-page"
import { ResultsPage } from "@/components/results-page"

export type PredictionResult = {
  id: string
  prediction: string
  confidence: number
}

export type APIResponse = {
  catalog_type: string
  priority: string
  total_rows: number
  features_in: number
  features_out: number
  results: PredictionResult[]
}

export type AppState = "landing" | "results"

export default function Home() {
  const [state, setState] = useState<AppState>("landing")
  const [results, setResults] = useState<PredictionResult[]>([])
  const [apiMetadata, setApiMetadata] = useState<Omit<APIResponse, "results"> | null>(null)

  const handleApiResponse = (apiResponse: APIResponse) => {
    setResults(apiResponse.results)
    setApiMetadata({
      catalog_type: apiResponse.catalog_type,
      priority: apiResponse.priority,
      total_rows: apiResponse.total_rows,
      features_in: apiResponse.features_in,
      features_out: apiResponse.features_out,
    })
    setState("results")
  }

  const handleBack = () => {
    setState("landing")
    setResults([])
    setApiMetadata(null)
  }

  return (
    <main className="relative min-h-screen overflow-hidden">
      <Starfield />

      {state === "landing" && <LandingPage onApiResponse={handleApiResponse} />}

      {state === "results" && <ResultsPage results={results} metadata={apiMetadata} onBack={handleBack} />}
    </main>
  )
}
