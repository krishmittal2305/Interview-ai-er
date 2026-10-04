"use client"

import React, { useState, useEffect, useCallback } from "react"
import { apiClient, ObservabilityStats, InferenceLogEvent } from "@/lib/api-client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Activity,
  Cpu,
  Zap,
  AlertCircle,
  CheckCircle2,
  RefreshCw,
  Search,
  DollarSign,
  Layers,
  Filter,
  ArrowUpRight,
  Database,
  Sparkles,
  Clock,
  Terminal,
  Server,
  Play,
  ChevronDown,
  ChevronUp,
} from "lucide-react"

export default function ObservabilityDashboard() {
  const [stats, setStats] = useState<ObservabilityStats | null>(null)
  const [events, setEvents] = useState<InferenceLogEvent[]>([])
  const [totalEvents, setTotalEvents] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isDiagnosticRunning, setIsDiagnosticRunning] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Filters
  const [timeWindow, setTimeWindow] = useState("24h")
  const [activeTab, setActiveTab] = useState<"overview" | "expensive" | "events">("overview")
  const [typeFilter, setTypeFilter] = useState("all")
  const [statusFilter, setStatusFilter] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")
  const [expandedSpanId, setExpandedSpanId] = useState<string | null>(null)

  const fetchData = useCallback(async (silent = false) => {
    if (!silent) setIsLoading(true)
    setIsRefreshing(true)
    try {
      const [statsData, eventsData] = await Promise.all([
        apiClient.getObservabilityStats(timeWindow, typeFilter),
        apiClient.getObservabilityEvents({
          limit: 50,
          offset: 0,
          type: typeFilter !== "all" ? typeFilter : undefined,
          status: statusFilter !== "all" ? statusFilter : undefined,
          search: searchQuery || undefined,
        }),
      ])

      setStats(statsData)
      setEvents(eventsData?.events || [])
      setTotalEvents(eventsData?.total || 0)
    } catch (err) {
      console.error("Failed to load observability data:", err)
    } finally {
      setIsLoading(false)
      setIsRefreshing(false)
    }
  }, [timeWindow, typeFilter, statusFilter, searchQuery])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Auto-refresh interval (10s)
  useEffect(() => {
    if (!autoRefresh) return
    const timer = setInterval(() => {
      fetchData(true)
    }, 10000)
    return () => clearInterval(timer)
  }, [autoRefresh, fetchData])

  const handleDiagnosticPing = async () => {
    setIsDiagnosticRunning(true)
    try {
      await apiClient.triggerDiagnosticPing("ml")
      await fetchData(true)
    } catch (err) {
      console.error("Diagnostic ping failed:", err)
    } finally {
      setIsDiagnosticRunning(false)
    }
  }

  const toggleExpand = (id: string) => {
    setExpandedSpanId((prev) => (prev === id ? null : id))
  }

  const totalVol = stats?.total_volume || 0
  const mlVol = stats?.ml_volume || 0
  const llmVol = stats?.llm_volume || 0
  const mlPercent = totalVol > 0 ? Math.round((mlVol / totalVol) * 100) : 0
  const llmPercent = totalVol > 0 ? Math.round((llmVol / totalVol) * 100) : 0

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header & Controls */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div>
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-indigo-500/10 text-indigo-400 rounded-lg border border-indigo-500/20">
                <Activity className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                  AI & ML Inference Observability
                  <Badge variant="outline" className="bg-indigo-950/60 border-indigo-500/30 text-indigo-300 text-xs py-0.5">
                    Live Telemetry
                  </Badge>
                </h1>
                <p className="text-sm text-slate-400">
                  Real-time latency, throughput, failure rates, and hardware allocation across local ML and LLM providers.
                </p>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Time Window Pills */}
            <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
              {["1h", "24h", "7d", "all"].map((w) => (
                <button
                  key={w}
                  onClick={() => setTimeWindow(w)}
                  className={`px-3 py-1 rounded-md font-medium transition-all ${
                    timeWindow === w
                      ? "bg-indigo-600 text-white shadow"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {w === "1h" ? "1 Hour" : w === "24h" ? "24 Hours" : w === "7d" ? "7 Days" : "All Time"}
                </button>
              ))}
            </div>

            {/* Auto-Refresh Toggle */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`border-slate-800 text-xs ${
                autoRefresh
                  ? "bg-emerald-950/40 text-emerald-300 border-emerald-800/50 hover:bg-emerald-950/60"
                  : "bg-slate-900 text-slate-400 hover:bg-slate-800"
              }`}
            >
              <span className={`w-2 h-2 rounded-full mr-2 ${autoRefresh ? "bg-emerald-400 animate-pulse" : "bg-slate-600"}`} />
              {autoRefresh ? "Live 10s" : "Paused"}
            </Button>

            {/* Manual Refresh */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => fetchData(false)}
              disabled={isRefreshing}
              className="bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800 text-xs"
            >
              <RefreshCw className={`h-3.5 w-3.5 mr-1.5 ${isRefreshing ? "animate-spin text-indigo-400" : ""}`} />
              Refresh
            </Button>

            {/* Diagnostic Inference Ping */}
            <Button
              size="sm"
              onClick={handleDiagnosticPing}
              disabled={isDiagnosticRunning}
              className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs shadow-md shadow-indigo-900/30"
            >
              <Play className={`h-3.5 w-3.5 mr-1.5 ${isDiagnosticRunning ? "animate-spin" : ""}`} />
              {isDiagnosticRunning ? "Benchmarking..." : "Run Diagnostic Ping"}
            </Button>
          </div>
        </div>

        {/* Top Metric KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          
          {/* Card 1: Volume */}
          <Card className="bg-slate-900/90 border-slate-800">
            <CardHeader className="pb-2">
              <CardDescription className="text-slate-400 text-xs flex items-center justify-between">
                <span>Inference Volume</span>
                <Layers className="h-4 w-4 text-indigo-400" />
              </CardDescription>
              <CardTitle className="text-2xl font-bold text-white">
                {totalVol.toLocaleString()}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between text-slate-400">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block" /> ML: {mlVol} ({mlPercent}%)
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-purple-400 inline-block" /> LLM: {llmVol} ({llmPercent}%)
                  </span>
                </div>
                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
                  <div style={{ width: `${mlPercent}%` }} className="bg-cyan-400 h-full" />
                  <div style={{ width: `${llmPercent}%` }} className="bg-purple-400 h-full" />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Card 2: Average Latency */}
          <Card className="bg-slate-900/90 border-slate-800">
            <CardHeader className="pb-2">
              <CardDescription className="text-slate-400 text-xs flex items-center justify-between">
                <span>Average Latency</span>
                <Clock className="h-4 w-4 text-amber-400" />
              </CardDescription>
              <CardTitle className="text-2xl font-bold text-white">
                {stats?.average_latency_ms ? `${stats.average_latency_ms.toFixed(1)} ms` : "0.0 ms"}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0 text-xs text-slate-400">
              Mean response duration across all completed executions.
            </CardContent>
          </Card>

          {/* Card 3: P95 Latency */}
          <Card className="bg-slate-900/90 border-slate-800">
            <CardHeader className="pb-2">
              <CardDescription className="text-slate-400 text-xs flex items-center justify-between">
                <span>P95 Tail Latency</span>
                <Zap className="h-4 w-4 text-rose-400" />
              </CardDescription>
              <CardTitle className="text-2xl font-bold text-white">
                {stats?.p95_latency_ms ? `${stats.p95_latency_ms.toFixed(1)} ms` : "0.0 ms"}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0 text-xs text-slate-400">
              95th percentile watermark for SLA adherence and timeout risk.
            </CardContent>
          </Card>

          {/* Card 4: Failure Rate */}
          <Card className="bg-slate-900/90 border-slate-800">
            <CardHeader className="pb-2">
              <CardDescription className="text-slate-400 text-xs flex items-center justify-between">
                <span>Failure Rate</span>
                <AlertCircle className={`h-4 w-4 ${(stats?.failure_rate_pct || 0) > 0 ? "text-rose-400" : "text-emerald-400"}`} />
              </CardDescription>
              <CardTitle className={`text-2xl font-bold ${(stats?.failure_rate_pct || 0) > 0 ? "text-rose-300" : "text-emerald-300"}`}>
                {stats?.failure_rate_pct !== undefined ? `${stats.failure_rate_pct.toFixed(1)}%` : "0.0%"}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0 text-xs text-slate-400">
              {stats?.failure_rate_pct === 0 ? "100% operational success rate" : "Exceptions caught & logged"}
            </CardContent>
          </Card>

          {/* Card 5: Fallback Rate */}
          <Card className="bg-slate-900/90 border-slate-800">
            <CardHeader className="pb-2">
              <CardDescription className="text-slate-400 text-xs flex items-center justify-between">
                <span>Fallback Rate</span>
                <Server className="h-4 w-4 text-blue-400" />
              </CardDescription>
              <CardTitle className="text-2xl font-bold text-white">
                {stats?.fallback_rate_pct !== undefined ? `${stats.fallback_rate_pct.toFixed(1)}%` : "0.0%"}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0 text-xs text-slate-400">
              Rerouted to heuristics or backup models on provider degradation.
            </CardContent>
          </Card>

        </div>

        {/* View Navigation Tabs */}
        <div className="flex border-b border-slate-800 space-x-6 text-sm font-medium">
          <button
            onClick={() => setActiveTab("overview")}
            className={`pb-3 transition-colors flex items-center gap-2 ${
              activeTab === "overview"
                ? "border-b-2 border-indigo-500 text-indigo-400 font-semibold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Database className="h-4 w-4" />
            Model & LLM Breakdown
          </button>
          <button
            onClick={() => setActiveTab("expensive")}
            className={`pb-3 transition-colors flex items-center gap-2 ${
              activeTab === "expensive"
                ? "border-b-2 border-indigo-500 text-indigo-400 font-semibold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Sparkles className="h-4 w-4" />
            Most Expensive Operations
            {stats?.most_expensive_operations && stats.most_expensive_operations.length > 0 && (
              <Badge variant="secondary" className="bg-slate-800 text-slate-300 text-xs px-1.5 py-0">
                {stats.most_expensive_operations.length}
              </Badge>
            )}
          </button>
          <button
            onClick={() => setActiveTab("events")}
            className={`pb-3 transition-colors flex items-center gap-2 ${
              activeTab === "events"
                ? "border-b-2 border-indigo-500 text-indigo-400 font-semibold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Terminal className="h-4 w-4" />
            Live Telemetry Stream
            <Badge variant="secondary" className="bg-slate-800 text-slate-300 text-xs px-1.5 py-0">
              {totalEvents}
            </Badge>
          </button>
        </div>

        {/* ZERO-STATE ALERT IF NO RUNS RECORDED */}
        {totalVol === 0 && !isLoading && (
          <div className="p-8 rounded-xl bg-slate-900/60 border border-slate-800 text-center space-y-4">
            <div className="w-12 h-12 bg-slate-800 rounded-full flex items-center justify-center mx-auto text-slate-400">
              <Activity className="h-6 w-6" />
            </div>
            <div className="max-w-md mx-auto">
              <h3 className="text-base font-semibold text-white">No inference telemetry recorded in this window</h3>
              <p className="text-sm text-slate-400 mt-1">
                This dashboard strictly reflects genuine recorded telemetry without fabricated metrics.
                Run a diagnostic ping or perform an interview practice to generate real telemetry.
              </p>
            </div>
            <Button
              onClick={handleDiagnosticPing}
              disabled={isDiagnosticRunning}
              className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs"
            >
              <Play className="h-3.5 w-3.5 mr-1.5" />
              Run Immediate Diagnostic Ping
            </Button>
          </div>
        )}

        {/* TAB 1: OVERVIEW & BREAKDOWN */}
        {activeTab === "overview" && totalVol > 0 && (
          <div className="space-y-6">
            
            {/* Model Usage Table */}
            <Card className="bg-slate-900/90 border-slate-800">
              <CardHeader className="pb-3 border-b border-slate-800/60">
                <CardTitle className="text-base font-semibold text-white flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-indigo-400" />
                  Model Usage & Latency Distribution
                </CardTitle>
                <CardDescription className="text-slate-400 text-xs">
                  Active execution share, mean latency, p95 tail, and failure rate broken down by model.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0 overflow-x-auto">
                <table className="w-full text-left text-sm text-slate-300">
                  <thead className="bg-slate-950/60 text-xs uppercase text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="px-4 py-3">Model</th>
                      <th className="px-4 py-3">Type</th>
                      <th className="px-4 py-3 text-right">Calls</th>
                      <th className="px-4 py-3 text-right">Traffic Share</th>
                      <th className="px-4 py-3 text-right">Avg Latency</th>
                      <th className="px-4 py-3 text-right">P95 Latency</th>
                      <th className="px-4 py-3 text-right">Failure Rate</th>
                      <th className="px-4 py-3 text-right">Fallback Rate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {stats?.model_usage.map((m) => (
                      <tr key={m.model} className="hover:bg-slate-800/40 transition-colors">
                        <td className="px-4 py-3 font-mono text-xs text-white">
                          {m.model}
                        </td>
                        <td className="px-4 py-3">
                          <Badge
                            variant="outline"
                            className={`text-[10px] uppercase font-bold py-0.5 ${
                              m.inference_type.toLowerCase() === "llm"
                                ? "bg-purple-950/50 text-purple-300 border-purple-800/50"
                                : "bg-cyan-950/50 text-cyan-300 border-cyan-800/50"
                            }`}
                          >
                            {m.inference_type}
                          </Badge>
                        </td>
                        <td className="px-4 py-3 text-right font-semibold text-white">
                          {m.total_calls.toLocaleString()}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <div className="flex items-center justify-end gap-2">
                            <span>{m.share_pct}%</span>
                            <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                              <div
                                style={{ width: `${m.share_pct}%` }}
                                className={`h-full ${
                                  m.inference_type.toLowerCase() === "llm" ? "bg-purple-500" : "bg-cyan-500"
                                }`}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-xs">
                          {m.avg_latency_ms.toFixed(1)} ms
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-xs text-amber-300">
                          {m.p95_latency_ms.toFixed(1)} ms
                        </td>
                        <td className="px-4 py-3 text-right">
                          <span
                            className={
                              m.failure_rate_pct > 0
                                ? "text-rose-400 font-medium"
                                : "text-emerald-400 font-medium"
                            }
                          >
                            {m.failure_rate_pct.toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right text-slate-400">
                          {m.fallback_rate_pct.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>

            {/* LLM Token & Cost Breakdown (if any LLM calls exist) */}
            {stats?.llm_usage && stats.llm_usage.length > 0 && (
              <Card className="bg-slate-900/90 border-slate-800">
                <CardHeader className="pb-3 border-b border-slate-800/60">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-base font-semibold text-white flex items-center gap-2">
                        <DollarSign className="h-4 w-4 text-emerald-400" />
                        LLM Token & Cost Consumption
                      </CardTitle>
                      <CardDescription className="text-slate-400 text-xs">
                        Token generation footprints and calculated operational costs based on provider rates.
                      </CardDescription>
                    </div>
                    <Badge variant="outline" className="bg-emerald-950/40 border-emerald-800/40 text-emerald-300">
                      Est. Total: ${stats.total_estimated_cost_usd.toFixed(4)}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="p-0 overflow-x-auto">
                  <table className="w-full text-left text-sm text-slate-300">
                    <thead className="bg-slate-950/60 text-xs uppercase text-slate-400 border-b border-slate-800">
                      <tr>
                        <th className="px-4 py-3">LLM Model</th>
                        <th className="px-4 py-3 text-right">Requests</th>
                        <th className="px-4 py-3 text-right">Input Tokens</th>
                        <th className="px-4 py-3 text-right">Output Tokens</th>
                        <th className="px-4 py-3 text-right">Total Tokens</th>
                        <th className="px-4 py-3 text-right">Estimated Cost (USD)</th>
                        <th className="px-4 py-3 text-right">Avg Latency</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {stats.llm_usage.map((l) => (
                        <tr key={l.model} className="hover:bg-slate-800/40 transition-colors">
                          <td className="px-4 py-3 font-mono text-xs text-white">
                            {l.model}
                          </td>
                          <td className="px-4 py-3 text-right font-semibold text-white">
                            {l.total_calls.toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-xs text-slate-400">
                            {l.input_tokens.toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-xs text-slate-400">
                            {l.output_tokens.toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-xs text-indigo-300 font-medium">
                            {l.total_tokens.toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-xs text-emerald-400 font-semibold">
                            ${l.estimated_cost_usd.toFixed(4)}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-xs">
                            {l.avg_latency_ms.toFixed(1)} ms
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
            )}

            {/* Task Usage Breakdown */}
            {stats?.task_usage && stats.task_usage.length > 0 && (
              <Card className="bg-slate-900/90 border-slate-800">
                <CardHeader className="pb-3 border-b border-slate-800/60">
                  <CardTitle className="text-base font-semibold text-white flex items-center gap-2">
                    <Terminal className="h-4 w-4 text-cyan-400" />
                    Inference Tasks & Functional Pipelines
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {stats.task_usage.map((t) => (
                    <div
                      key={t.task}
                      className="p-3 bg-slate-950/60 border border-slate-800/80 rounded-lg flex flex-col justify-between"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs text-indigo-300 font-medium truncate max-w-[180px]">
                          {t.task}
                        </span>
                        <Badge
                          variant="outline"
                          className="text-[10px] px-1.5 py-0 uppercase bg-slate-900 border-slate-700"
                        >
                          {t.inference_type}
                        </Badge>
                      </div>
                      <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
                        <span>{t.total_calls} calls</span>
                        <span className="font-mono">{t.avg_latency_ms.toFixed(1)} ms avg</span>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

          </div>
        )}

        {/* TAB 2: MOST EXPENSIVE OPERATIONS */}
        {activeTab === "expensive" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-xs text-slate-400">
                Top executions ordered by latency duration and computational / token expense.
              </p>
            </div>

            {(!stats?.most_expensive_operations || stats.most_expensive_operations.length === 0) ? (
              <div className="p-8 text-center bg-slate-900/50 rounded-xl border border-slate-800 text-slate-400 text-sm">
                No recorded operations yet.
              </div>
            ) : (
              <div className="space-y-2">
                {stats.most_expensive_operations.map((op, idx) => (
                  <div
                    key={op.id || idx}
                    className="p-4 bg-slate-900/90 border border-slate-800 rounded-lg hover:border-slate-700 transition-all"
                  >
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                      <div className="flex items-start gap-3">
                        <span className="w-6 h-6 rounded-full bg-slate-800 flex items-center justify-center text-xs font-mono text-slate-300 flex-shrink-0">
                          {idx + 1}
                        </span>
                        <div>
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-semibold text-white text-sm">{op.task}</span>
                            <Badge
                              variant="outline"
                              className={`text-[10px] py-0 ${
                                op.inference_type === "llm"
                                  ? "bg-purple-950/60 text-purple-300 border-purple-800"
                                  : "bg-cyan-950/60 text-cyan-300 border-cyan-800"
                              }`}
                            >
                              {op.inference_type.toUpperCase()}
                            </Badge>
                            <span className="text-xs font-mono text-slate-400">
                              model: <span className="text-slate-200">{op.model}</span>
                            </span>
                            {op.fallback_usage && (
                              <Badge className="bg-amber-950/80 text-amber-300 border-amber-800 text-[10px] py-0">
                                Fallback Invoked
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-4 text-xs text-slate-400 mt-1">
                            <span>Device: <span className="text-slate-300 font-mono">{op.memory_device || "cpu"}</span></span>
                            <span>Input Size: <span className="text-slate-300 font-mono">{op.input_size || 0}</span></span>
                            {op.output_size !== null && op.output_size !== undefined && (
                              <span>Output Size: <span className="text-slate-300 font-mono">{op.output_size}</span></span>
                            )}
                            {op.retry_count > 0 && (
                              <span className="text-amber-400 font-medium">Retries: {op.retry_count}</span>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-4 self-end md:self-center">
                        <div className="text-right">
                          <div className="text-base font-mono font-bold text-amber-400">
                            {op.latency_ms.toFixed(1)} ms
                          </div>
                          {op.cost_estimate_usd !== undefined && op.cost_estimate_usd > 0 && (
                            <div className="text-[11px] font-mono text-emerald-400">
                              ${op.cost_estimate_usd.toFixed(6)}
                            </div>
                          )}
                        </div>

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleExpand(op.id)}
                          className="text-slate-400 hover:text-white"
                        >
                          {expandedSpanId === op.id ? (
                            <ChevronUp className="h-4 w-4" />
                          ) : (
                            <ChevronDown className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </div>

                    {/* Expanded details */}
                    {expandedSpanId === op.id && (
                      <div className="mt-3 pt-3 border-t border-slate-800 text-xs">
                        <div className="bg-slate-950 p-3 rounded font-mono text-slate-300 overflow-x-auto">
                          <pre>{JSON.stringify(op, null, 2)}</pre>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: LIVE TELEMETRY STREAM */}
        {activeTab === "events" && (
          <div className="space-y-4">
            
            {/* Filter Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-900/90 border border-slate-800 rounded-lg">
              <div className="flex items-center gap-2 flex-1 min-w-[200px]">
                <Search className="h-4 w-4 text-slate-500" />
                <Input
                  placeholder="Filter by model, task, error, or device..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-slate-950 border-slate-800 text-xs h-8 text-slate-200 placeholder:text-slate-500"
                />
              </div>

              <div className="flex items-center gap-2">
                {/* Type Filter */}
                <select
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value)}
                  className="bg-slate-950 border border-slate-800 text-xs text-slate-300 rounded px-2.5 py-1.5 focus:outline-none"
                >
                  <option value="all">All Types (ML & LLM)</option>
                  <option value="ml">ML Only</option>
                  <option value="llm">LLM Only</option>
                </select>

                {/* Status Filter */}
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="bg-slate-950 border border-slate-800 text-xs text-slate-300 rounded px-2.5 py-1.5 focus:outline-none"
                >
                  <option value="all">All Statuses</option>
                  <option value="SUCCESS">Success</option>
                  <option value="FAILED">Failed</option>
                  <option value="FALLBACK">Fallback</option>
                </select>
              </div>
            </div>

            {/* Events List */}
            {events.length === 0 ? (
              <div className="p-8 text-center bg-slate-900/50 rounded-xl border border-slate-800 text-slate-400 text-sm">
                No telemetry spans match the current filters.
              </div>
            ) : (
              <div className="bg-slate-900/90 border border-slate-800 rounded-lg overflow-x-auto">
                <table className="w-full text-left text-sm text-slate-300">
                  <thead className="bg-slate-950/70 text-xs uppercase text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="px-4 py-3">Timestamp</th>
                      <th className="px-4 py-3">Type</th>
                      <th className="px-4 py-3">Task</th>
                      <th className="px-4 py-3">Model</th>
                      <th className="px-4 py-3">Status</th>
                      <th className="px-4 py-3 text-right">Latency</th>
                      <th className="px-4 py-3 text-right">Input/Output</th>
                      <th className="px-4 py-3">Device</th>
                      <th className="px-4 py-3 text-center">Inspect</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {events.map((ev) => (
                      <React.Fragment key={ev.id}>
                        <tr className="hover:bg-slate-800/40 transition-colors">
                          <td className="px-4 py-2.5 text-xs font-mono text-slate-400 whitespace-nowrap">
                            {ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : "--"}
                          </td>
                          <td className="px-4 py-2.5">
                            <Badge
                              variant="outline"
                              className={`text-[10px] py-0 uppercase ${
                                ev.inference_type === "llm"
                                  ? "bg-purple-950/50 text-purple-300 border-purple-800/50"
                                  : "bg-cyan-950/50 text-cyan-300 border-cyan-800/50"
                              }`}
                            >
                              {ev.inference_type}
                            </Badge>
                          </td>
                          <td className="px-4 py-2.5 text-xs font-medium text-white max-w-[180px] truncate">
                            {ev.task}
                          </td>
                          <td className="px-4 py-2.5 text-xs font-mono text-slate-300 max-w-[160px] truncate">
                            {ev.model}
                          </td>
                          <td className="px-4 py-2.5">
                            <Badge
                              className={`text-[10px] py-0 ${
                                ev.status === "SUCCESS"
                                  ? "bg-emerald-950/80 text-emerald-300 border-emerald-800"
                                  : ev.status === "FALLBACK"
                                  ? "bg-amber-950/80 text-amber-300 border-amber-800"
                                  : "bg-rose-950/80 text-rose-300 border-rose-800"
                              }`}
                            >
                              {ev.status}
                            </Badge>
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono text-xs font-semibold text-slate-200">
                            {ev.latency_ms.toFixed(1)} ms
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono text-xs text-slate-400">
                            {ev.input_size} / {ev.output_size ?? "--"}
                          </td>
                          <td className="px-4 py-2.5 text-xs font-mono text-slate-400">
                            {ev.memory_device || "cpu"}
                          </td>
                          <td className="px-4 py-2.5 text-center">
                            <button
                              onClick={() => toggleExpand(ev.id)}
                              className="text-xs text-indigo-400 hover:text-indigo-300 font-mono"
                            >
                              {expandedSpanId === ev.id ? "Close" : "JSON"}
                            </button>
                          </td>
                        </tr>

                        {expandedSpanId === ev.id && (
                          <tr className="bg-slate-950">
                            <td colSpan={9} className="p-4">
                              <div className="bg-slate-900 border border-slate-800 p-3 rounded font-mono text-xs text-slate-300 overflow-x-auto">
                                <pre>{JSON.stringify(ev, null, 2)}</pre>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  )
}
