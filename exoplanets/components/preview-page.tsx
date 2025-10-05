"use client"

import { ArrowLeft, Play } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { ExoplanetData } from "@/app/page"

interface PreviewPageProps {
  data: ExoplanetData[]
  onConfirm: () => void
  onBack: () => void
}

export function PreviewPage({ data, onConfirm, onBack }: PreviewPageProps) {
  const columns = data.length > 0 ? Object.keys(data[0]) : []

  return (
    <div className="relative z-10 min-h-screen px-4 py-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="flex items-center justify-between">
          <Button variant="ghost" onClick={onBack}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>

          <h1 className="text-3xl font-bold">Preview Data</h1>

          <Button className="glow" onClick={onConfirm}>
            <Play className="mr-2 h-4 w-4" />
            Run Prediction
          </Button>
        </div>

        <Card className="border-border/50">
          <div className="p-6">
            <div className="mb-4 flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                {data.length} rows • {columns.length} columns
              </p>
            </div>

            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    {columns.map((col) => (
                      <TableHead key={col} className="font-semibold">
                        {col}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.slice(0, 10).map((row, idx) => (
                    <TableRow key={idx}>
                      {columns.map((col) => (
                        <TableCell key={col}>{String(row[col])}</TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {data.length > 10 && (
              <p className="mt-4 text-center text-sm text-muted-foreground">Showing first 10 of {data.length} rows</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
