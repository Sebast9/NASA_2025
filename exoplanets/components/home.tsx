"use client"

import { Button } from "@/components/ui/button"
import { Sparkles, Telescope, TrendingUp, MessageSquare } from "lucide-react"

interface HomePageProps {
  onStart: () => void
}

export function HomePage({ onStart }: HomePageProps) {
  return (
    <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-4">
      <div className="mx-auto max-w-4xl text-center">
        {/* Hero Section */}

        <h1 className="mb-6 bg-gradient-to-b from-white to-gray-400 bg-clip-text text-6xl font-bold tracking-tight text-transparent md:text-7xl">
          ExoClassifier
        </h1>

        <p className="mb-12 text-xl text-gray-400 md:text-2xl">
          Analiza y clasifica exoplanetas usando modelos de Machine Learning avanzados
        </p>

        {/* Features Grid */}
        <div className="mb-12 grid gap-6 md:grid-cols-3">
          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm transition-all hover:border-blue-500/30 hover:bg-white/10">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-blue-500/20">
              <Telescope className="h-6 w-6 text-blue-400" />
            </div>
            <h3 className="mb-2 text-lg font-semibold text-white">Análisis Preciso</h3>
            <p className="text-sm text-gray-400">
              Clasificación de alta precisión usando algoritmos de ML entrenados con datos reales
            </p>
          </div>

          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm transition-all hover:border-green-500/30 hover:bg-white/10">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-green-500/20">
              <TrendingUp className="h-6 w-6 text-green-400" />
            </div>
            <h3 className="mb-2 text-lg font-semibold text-white">Visualización de Datos</h3>
            <p className="text-sm text-gray-400">Gráficas interactivas y estadísticas detalladas de tus resultados</p>
          </div>

          <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm transition-all hover:border-purple-500/30 hover:bg-white/10">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-purple-500/20">
              <MessageSquare className="h-6 w-6 text-purple-400" />
            </div>
            <h3 className="mb-2 text-lg font-semibold text-white">Asistente IA</h3>
            <p className="text-sm text-gray-400">Chatbot con Gemini para responder tus preguntas sobre exoplanetas</p>
          </div>
        </div>

        {/* CTA Button */}
        <Button
          onClick={onStart}
          size="lg"
          className="group relative overflow-hidden bg-gradient-to-r from-blue-600 to-purple-600 px-8 py-6 text-lg font-semibold text-white shadow-lg shadow-blue-500/25 transition-all hover:shadow-xl hover:shadow-blue-500/40"
        >
          <span className="relative z-10 flex items-center gap-2">
            Comenzar Análisis
            <Sparkles className="h-5 w-5 transition-transform group-hover:rotate-12" />
          </span>
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 opacity-0 transition-opacity group-hover:opacity-100" />
        </Button>

        {/* Info Text */}
      </div>
    </div>
  )
}
