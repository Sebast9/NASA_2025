"use client"

import { useState, useMemo, useEffect } from "react"
import {
  ArrowLeft,
  Download,
  Search,
  ArrowUpDown,
  Sparkles,
  TrendingDown,
  Target,
  CheckCircle2,
  BarChart3,
  X,
  Trophy,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import Papa from "papaparse"
import type { PredictionResult, APIResponse } from "@/app/page"
import { ChatBot } from "@/components/chatbot"

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
  const [showCharts, setShowCharts] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 50

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

  useEffect(() => {
    setCurrentPage(1)
  }, [searchTerm, filterStatus, sortBy, sortOrder])

  const paginatedResults = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage
    const endIndex = startIndex + itemsPerPage
    return filteredAndSorted.slice(startIndex, endIndex)
  }, [filteredAndSorted, currentPage, itemsPerPage])

  const totalPages = Math.ceil(filteredAndSorted.length / itemsPerPage)

  const getPageNumbers = () => {
    const pages: (number | string)[] = []
    const maxVisible = 5

    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i)
      }
    } else {
      if (currentPage <= 3) {
        for (let i = 1; i <= 4; i++) {
          pages.push(i)
        }
        pages.push("...")
        pages.push(totalPages)
      } else if (currentPage >= totalPages - 2) {
        pages.push(1)
        pages.push("...")
        for (let i = totalPages - 3; i <= totalPages; i++) {
          pages.push(i)
        }
      } else {
        pages.push(1)
        pages.push("...")
        pages.push(currentPage - 1)
        pages.push(currentPage)
        pages.push(currentPage + 1)
        pages.push("...")
        pages.push(totalPages)
      }
    }

    return pages
  }

  const stats = useMemo(() => {
    const falsePositive = results.filter((r) => r.prediction === "FALSE POSITIVE").length
    const candidate = results.filter((r) => r.prediction === "CANDIDATE").length
    const confirmed = results.filter((r) => r.prediction === "CONFIRMED").length
    const avgConfidence = (results.reduce((sum, r) => sum + r.confidence, 0) / results.length) * 100

    return { falsePositive, candidate, confirmed, avgConfidence: avgConfidence.toFixed(1) }
  }, [results])

  const topTypes = useMemo(() => {
    const typeCounts = new Map<string, number>()

    results.forEach((r) => {
      const count = typeCounts.get(r.prediction) || 0
      typeCounts.set(r.prediction, count + 1)
    })

    const sortedTypes = Array.from(typeCounts.entries())
      .map(([type, count]) => ({
        type,
        count,
        percentage: ((count / results.length) * 100).toFixed(1),
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5)

    return sortedTypes
  }, [results])

  const chartData = useMemo(() => {
    const pieData = [
      { name: "Falsos Positivos", value: stats.falsePositive, color: "#ef4444" },
      { name: "Candidatos", value: stats.candidate, color: "#22c55e" },
      { name: "Confirmados", value: stats.confirmed, color: "#eab308" },
    ]

    const barData = [
      { name: "Falsos Positivos", count: stats.falsePositive, fill: "#ef4444" },
      { name: "Candidatos", count: stats.candidate, fill: "#22c55e" },
      { name: "Confirmados", count: stats.confirmed, fill: "#eab308" },
    ]

    const confidenceBuckets = [
      { range: "0-20%", count: 0 },
      { range: "20-40%", count: 0 },
      { range: "40-60%", count: 0 },
      { range: "60-80%", count: 0 },
      { range: "80-100%", count: 0 },
    ]

    results.forEach((r) => {
      const conf = r.confidence * 100
      if (conf <= 20) confidenceBuckets[0].count++
      else if (conf <= 40) confidenceBuckets[1].count++
      else if (conf <= 60) confidenceBuckets[2].count++
      else if (conf <= 80) confidenceBuckets[3].count++
      else confidenceBuckets[4].count++
    })

    return { pieData, barData, confidenceBuckets }
  }, [results, stats])

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
            <Button
              variant="outline"
              onClick={() => setShowCharts(true)}
              className="border-border/50 hover:border-primary/50 transition-all bg-transparent"
            >
              <BarChart3 className="mr-2 h-4 w-4" />
              Visualizar Gráficas
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
                  <TableRow className="border-border/30 hover:bg-muted/30">
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
                  {paginatedResults.map((row, idx) => (
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

            <div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
              <p className="text-sm text-muted-foreground">
                Mostrando <span className="font-semibold text-foreground">{(currentPage - 1) * itemsPerPage + 1}</span>{" "}
                -{" "}
                <span className="font-semibold text-foreground">
                  {Math.min(currentPage * itemsPerPage, filteredAndSorted.length)}
                </span>{" "}
                de <span className="font-semibold text-foreground">{filteredAndSorted.length}</span> resultados
              </p>

              {totalPages > 1 && (
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
                    disabled={currentPage === 1}
                    className="border-border/50 hover:border-primary/50 transition-all disabled:opacity-50"
                  >
                    Anterior
                  </Button>

                  <div className="flex items-center gap-1">
                    {getPageNumbers().map((page, idx) => (
                      <Button
                        key={idx}
                        variant={page === currentPage ? "default" : "outline"}
                        size="sm"
                        onClick={() => typeof page === "number" && setCurrentPage(page)}
                        disabled={typeof page === "string"}
                        className={
                          page === currentPage
                            ? "bg-primary hover:bg-primary/90 min-w-[40px]"
                            : "border-border/50 hover:border-primary/50 transition-all min-w-[40px]"
                        }
                      >
                        {page}
                      </Button>
                    ))}
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
                    disabled={currentPage === totalPages}
                    className="border-border/50 hover:border-primary/50 transition-all disabled:opacity-50"
                  >
                    Siguiente
                  </Button>
                </div>
              )}
            </div>
          </div>
        </Card>

        {showCharts && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
            <div className="relative w-full max-w-5xl max-h-[90vh] overflow-y-auto bg-background border-2 border-primary/30 rounded-xl shadow-2xl">
              <div className="sticky top-0 z-10 flex items-center justify-between p-6 bg-background/95 backdrop-blur-sm border-b border-border/50">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <BarChart3 className="h-6 w-6 text-primary" />
                  Visualización de Resultados
                </h2>
                <Button variant="ghost" size="icon" onClick={() => setShowCharts(false)} className="hover:bg-muted/50">
                  <X className="h-5 w-5" />
                </Button>
              </div>

              <div className="space-y-8 p-6">
                <Card className="p-6 border-border/50">
                  <h3 className="text-lg font-semibold mb-4">Distribución de Predicciones</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={chartData.pieData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }: { name: string; percent: number }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {chartData.pieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </Card>

                <Card className="p-6 border-border/50">
                  <h3 className="text-lg font-semibold mb-4">Comparación de Conteos</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData.barData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="name" stroke="#888" />
                      <YAxis stroke="#888" />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #333" }}
                        labelStyle={{ color: "#fff" }}
                      />
                      <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                        {chartData.barData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                <Card className="p-6 border-border/50">
                  <h3 className="text-lg font-semibold mb-4">Distribución de Confianza</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData.confidenceBuckets}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="range" stroke="#888" />
                      <YAxis stroke="#888" />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #333" }}
                        labelStyle={{ color: "#fff" }}
                      />
                      <Bar dataKey="count" fill="#3b82f6" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            </div>
          </div>
        )}

        <ChatBot />
      </div>
    </div>
  )
}
