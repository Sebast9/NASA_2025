"use client"

import type React from "react"

import { useCallback, useState } from "react"
import { Upload, FileText, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import type { APIResponse } from "@/app/page"

interface LandingPageProps {
  onApiResponse: (data: APIResponse) => void
}

export function LandingPage({ onApiResponse }: LandingPageProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string>("")
  const [file, setFile] = useState<File | null>(null)
  const [datasetType, setDatasetType] = useState<"TOI" | "KOI" | "K2">("TOI")
  const [isLoading, setIsLoading] = useState(false)

  const handleFileSelect = useCallback((file: File) => {
    setError("")
    if (!file.name.endsWith(".csv") && !file.name.endsWith(".xlsx")) {
      setError("Solo se soportan archivos .csv o .xlsx")
      return
    }
    setFile(file)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile) handleFileSelect(droppedFile)
    },
    [handleFileSelect],
  )

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFile = e.target.files?.[0]
      if (selectedFile) handleFileSelect(selectedFile)
    },
    [handleFileSelect],
  )

  const handleSubmit = async () => {
    if (!file) {
      setError("Por favor selecciona un archivo primero.")
      return
    }

    const formData = new FormData()
    formData.append("file", file)

    setIsLoading(true)
    setError("")

    try {
      const res = await fetch(`http://localhost:8000/predict-excel?tipo=${datasetType}`, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        throw new Error("El servidor retornó un error.")
      }

      const data = await res.json()
      onApiResponse(data)
    } catch (err) {
      setError("Error al subir el archivo. Verifica el servidor y el formato del archivo.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-4">
      <div className="w-full max-w-4xl space-y-8 text-center">
        <div className="space-y-4">
          <div className="flex items-center justify-center gap-3">
            <Sparkles className="h-10 w-10 text-primary" />
            <h1 className="text-5xl font-bold tracking-tight text-balance">Predicción de Exoplanetas</h1>
          </div>
          <p className="text-xl text-muted-foreground text-balance">
            Sube un archivo CSV o Excel para predecir exoplanetas usando IA.
          </p>
        </div>

        <Card
          className={`glow border-2 transition-all duration-300 ${
            isDragging ? "border-primary bg-primary/5" : "border-border"
          }`}
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center gap-6 p-12">
            <div className="rounded-full bg-primary/10 p-6">
              <Upload className="h-12 w-12 text-primary" />
            </div>

            <div className="space-y-2">
              <h2 className="text-2xl font-semibold">Subir Archivo</h2>
              <p className="text-muted-foreground">Arrastra tu archivo aquí o haz clic para elegirlo (.csv o .xlsx)</p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-muted-foreground">Tipo de Dataset</label>
              <select
                value={datasetType}
                onChange={(e) => setDatasetType(e.target.value as "TOI" | "KOI" | "K2")}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option value="TOI">TOI</option>
                <option value="KOI">KOI</option>
                <option value="K2">K2</option>
              </select>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row">
              <Button size="lg" className="glow" asChild disabled={isLoading}>
                <label className="cursor-pointer">
                  <FileText className="mr-2 h-5 w-5" />
                  Seleccionar archivo
                  <input
                    type="file"
                    accept=".csv,.xlsx"
                    className="hidden"
                    onChange={handleFileInput}
                    disabled={isLoading}
                  />
                </label>
              </Button>

              <Button size="lg" variant="outline" onClick={handleSubmit} disabled={!file || isLoading}>
                {isLoading ? "Procesando..." : "Analizar"}
              </Button>
            </div>

            {file && (
              <p className="text-sm text-foreground">
                Archivo seleccionado: <strong>{file.name}</strong>
              </p>
            )}
            {error && <p className="text-sm text-destructive">{error}</p>}
          </div>
        </Card>

        <div className="grid gap-6 sm:grid-cols-3">
          <Card className="border-border/50 bg-card/50 p-6">
            <h3 className="mb-2 font-semibold">Paso 1</h3>
            <p className="text-sm text-muted-foreground">Sube tu archivo CSV o Excel con datos de exoplanetas</p>
          </Card>
          <Card className="border-border/50 bg-card/50 p-6">
            <h3 className="mb-2 font-semibold">Paso 2</h3>
            <p className="text-sm text-muted-foreground">Selecciona el tipo de dataset (TOI, KOI, K2)</p>
          </Card>
          <Card className="border-border/50 bg-card/50 p-6">
            <h3 className="mb-2 font-semibold">Paso 3</h3>
            <p className="text-sm text-muted-foreground">Obtén predicciones impulsadas por IA</p>
          </Card>
        </div>
      </div>
    </div>
  )
}
