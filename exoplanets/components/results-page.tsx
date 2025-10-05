"use client"

import { useState, useMemo } from "react"
import { ArrowLeft, Download, Search, ArrowUpDown, Sparkles, TrendingDown, Target, CheckCircle2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import Papa from "papaparse"
import type { PredictionResult, APIResponse } from "@/app/page"

interface ResultsPageProps {
  results: PredictionResult[]
  metadata: Omit<APIResponse, "results"> | null
  onBack: () => void
}

export function ResultsPage({ results, metadata, onBack }: ResultsPageProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [filterStatus, setFilterStatus] = useState<string>("all")
  const [sortBy, setSortBy] = useState<string>("confidence")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc")

  const filteredAndSorted = useMemo(() => {
    const filtered = results.filter((row) => {
      const matchesSearch =
        row.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        row.prediction.toLowerCase().includes(searchTerm.toLowerCase()) ||
        String((row.confidence * 100).toFixed(2)).includes(searchTerm)
      const matchesFilter = filterStatus === "all" || row.prediction === filterStatus
      return matchesSearch && matchesFilter
    })

    filtered.sort((a, b) => {
      const aVal = a[sortBy as keyof PredictionResult]
      const bVal = b[sortBy as keyof PredictionResult]

      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortOrder === "asc" ? aVal - bVal : bVal - aVal
      }

      return sortOrder === "asc" ? String(aVal).localeCompare(String(bVal)) : String(bVal).localeCompare(String(aVal))
    })

    return filtered
  }, [results, searchTerm, filterStatus, sortBy, sortOrder])

  const stats = useMemo(() => {
    const falsePositive = results.filter((r) => r.prediction === "FALSE POSITIVE").length
    const candidate = results.filter((r) => r.prediction === "CANDIDATE").length
    const confirmed = results.filter((r) => r.prediction === "CONFIRMED").length
    const avgConfidence = (results.reduce((sum, r) => sum + r.confidence, 0) / results.length) * 100

    return { falsePositive, candidate, confirmed, avgConfidence: avgConfidence.toFixed(1) }
  }, [results])

  const handleExport = () => {
    const exportData = filteredAndSorted.map((row) => ({
      id: row.id,
      prediction: row.prediction,
      confidence: `${(row.confidence * 100).toFixed(2)}%`,
    }))
    const csv = Papa.unparse(exportData)
    const blob = new Blob([csv], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "exoplanet-predictions.csv"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="relative z-10 min-h-screen px-4 py-8">
      <div className="mx-auto max-w-7xl space-y-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-primary/10 p-2.5">
              <Sparkles className="h-7 w-7 text-primary" />
            </div>
            <h1 className="text-4xl font-bold tracking-tight text-balance">Resultados de Predicción</h1>
          </div>

          <div className="flex gap-3">
            <Button
              variant="outline"
              onClick={onBack}
              className="border-border/50 hover:border-primary/50 transition-all bg-transparent"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Nuevo Análisis
            </Button>
            <Button className="glow bg-primary hover:bg-primary/90 transition-all" onClick={handleExport}>
              <Download className="mr-2 h-4 w-4" />
              Exportar CSV
            </Button>
          </div>
        </div>

        {metadata && (
          <Card className="glow border-2 border-primary/20 bg-card/95 backdrop-blur-sm p-6 transition-all hover:border-primary/30">
            <div className="flex items-center gap-2 mb-6">
              <div className="rounded-lg bg-primary/10 p-2">
                <Target className="h-5 w-5 text-primary" />
              </div>
              <h2 className="text-xl font-semibold">Información del Dataset</h2>
            </div>
            <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
              <div className="space-y-1">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Tipo de Catálogo</p>
                <p className="text-2xl font-bold text-primary">{metadata.catalog_type}</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Prioridad</p>
                <p className="text-2xl font-bold">{metadata.priority}</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Total de Filas</p>
                <p className="text-2xl font-bold">{metadata.total_rows.toLocaleString()}</p>
              </div>
            </div>
          </Card>
        )}

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <Card className="glow border-2 border-red-500/30 bg-gradient-to-br from-card/95 to-red-950/20 backdrop-blur-sm p-6 transition-all hover:border-red-500/50 hover:shadow-[0_0_30px_rgba(239,68,68,0.3)]">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium text-red-200/70">Falsos Positivos</p>
                <p className="text-4xl font-bold text-red-500 tabular-nums">{stats.falsePositive}</p>
              </div>
              <div className="rounded-xl bg-red-500/10 p-3 ring-1 ring-red-500/20">
                <TrendingDown className="h-7 w-7 text-red-500" />
              </div>
            </div>
          </Card>

          <Card className="glow border-2 border-green-500/30 bg-gradient-to-br from-card/95 to-green-950/20 backdrop-blur-sm p-6 transition-all hover:border-green-500/50 hover:shadow-[0_0_30px_rgba(34,197,94,0.3)]">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium text-green-200/70">Candidatos</p>
                <p className="text-4xl font-bold text-green-500 tabular-nums">{stats.candidate}</p>
              </div>
              <div className="rounded-xl bg-green-500/10 p-3 ring-1 ring-green-500/20">
                <Sparkles className="h-7 w-7 text-green-500" />
              </div>
            </div>
          </Card>

          <Card className="glow border-2 border-yellow-500/30 bg-gradient-to-br from-card/95 to-yellow-950/20 backdrop-blur-sm p-6 transition-all hover:border-yellow-500/50 hover:shadow-[0_0_30px_rgba(234,179,8,0.3)]">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium text-yellow-200/70">Confirmados</p>
                <p className="text-4xl font-bold text-yellow-500 tabular-nums">{stats.confirmed}</p>
              </div>
              <div className="rounded-xl bg-yellow-500/10 p-3 ring-1 ring-yellow-500/20">
                <CheckCircle2 className="h-7 w-7 text-yellow-500" />
              </div>
            </div>
          </Card>

          <Card className="glow border-2 border-primary/30 bg-gradient-to-br from-card/95 to-primary/10 backdrop-blur-sm p-6 transition-all hover:border-primary/50 hover:shadow-[0_0_30px_rgba(59,130,246,0.3)]">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium text-primary-foreground/70">Confianza Promedio</p>
                <p className="text-4xl font-bold text-primary tabular-nums">{stats.avgConfidence}%</p>
              </div>
              <div className="rounded-xl bg-primary/10 p-3 ring-1 ring-primary/20">
                <Target className="h-7 w-7 text-primary" />
              </div>
            </div>
          </Card>
        </div>

        <Card className="glow border-2 border-border/50 bg-card/95 backdrop-blur-sm transition-all hover:border-border/70">
          <div className="p-6">
            <div className="mb-6 flex flex-col gap-4 sm:flex-row">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Buscar por ID o predicción..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 border-border/50 bg-background/50 focus:border-primary/50 transition-all"
                />
              </div>

              <Select value={filterStatus} onValueChange={setFilterStatus}>
                <SelectTrigger className="w-full sm:w-[200px] border-border/50 bg-background/50">
                  <SelectValue placeholder="Filtrar por estado" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Todos</SelectItem>
                  <SelectItem value="CANDIDATE">Candidato</SelectItem>
                  <SelectItem value="FALSE POSITIVE">Falso Positivo</SelectItem>
                  <SelectItem value="CONFIRMED">Confirmado</SelectItem>
                </SelectContent>
              </Select>

              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-full sm:w-[180px] border-border/50 bg-background/50">
                  <SelectValue placeholder="Ordenar por" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="confidence">Confianza</SelectItem>
                  <SelectItem value="prediction">Predicción</SelectItem>
                  <SelectItem value="id">ID</SelectItem>
                </SelectContent>
              </Select>

              <Button
                variant="outline"
                size="icon"
                onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
                className="border-border/50 hover:border-primary/50 transition-all"
              >
                <ArrowUpDown className="h-4 w-4" />
              </Button>
            </div>

            <div className="overflow-x-auto rounded-lg border border-border/50">
              <Table className="table-auto">
                <TableHeader>
                  <TableRow className="border-border/50 hover:bg-muted/30">
                    <TableHead className="font-semibold text-foreground text-center px-4 py-3 whitespace-nowrap">
                      ID
                    </TableHead>
                    <TableHead className="font-semibold text-foreground text-center px-4 py-3 whitespace-nowrap">
                      Predicción
                    </TableHead>
                    <TableHead className="font-semibold text-foreground text-center px-4 py-3 whitespace-nowrap">
                      Confianza
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredAndSorted.map((row, idx) => (
                    <TableRow key={idx} className="border-border/30 hover:bg-muted/20 transition-colors">
                      <TableCell className="font-mono text-sm text-center px-4 py-3 whitespace-nowrap">
                        {row.id}
                      </TableCell>
                      <TableCell className="text-center px-4 py-3">
                        <div className="flex justify-center">
                          <Badge
                            variant={row.prediction === "CANDIDATE" ? "default" : "secondary"}
                            className={
                              row.prediction === "CANDIDATE"
                                ? "bg-green-500/90 hover:bg-green-500 text-white font-medium px-3 py-1"
                                : row.prediction === "CONFIRMED"
                                  ? "bg-yellow-500/90 hover:bg-yellow-500 text-white font-medium px-3 py-1"
                                  : "bg-red-500/90 hover:bg-red-500 text-white font-medium px-3 py-1"
                            }
                          >
                            {row.prediction}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell className="px-4 py-3">
                        <div className="flex items-center justify-center gap-3">
                          <div className="h-2 w-32 overflow-hidden rounded-full bg-muted/50 ring-1 ring-border/30">
                            <div
                              className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-300"
                              style={{ width: `${row.confidence * 100}%` }}
                            />
                          </div>
                          <span className="font-mono text-sm font-medium tabular-nums">
                            {(row.confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            <p className="mt-6 text-center text-sm text-muted-foreground">
              Mostrando <span className="font-semibold text-foreground">{filteredAndSorted.length}</span> de{" "}
              <span className="font-semibold text-foreground">{results.length}</span> resultados
            </p>
          </div>
        </Card>
      </div>
    </div>
  )
}
