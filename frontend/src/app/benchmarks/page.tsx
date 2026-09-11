"use client";

import { apiFetch } from "@/lib/api";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  Gauge,
  Layers3,
  Medal,
  RefreshCw,
  Search,
  Server,
  Target,
  Trophy,
  Zap,
} from "lucide-react";


// ============================================================
// CONFIGURATION
// ============================================================

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

const fetch = apiFetch;


// ============================================================
// TYPES
// ============================================================

type BenchmarkRecord = {
  model: string;
  model_id: string;
  task: string;
  dataset: string | null;
  split: string | null;
  samples: number;
  embedding_dimension: number | null;

  accuracy: number | null;
  auroc: number | null;
  f1: number | null;
  precision: number | null;
  recall: number | null;

  recall_at_1: number | null;
  recall_at_5: number | null;
  recall_at_10: number | null;

  latency_ms: number | null;
  throughput: number | null;
  gpu_memory_mb: number | null;

  status: string;
};

type BenchmarkResponse = {
  total_records: number;
  records: BenchmarkRecord[];
};


// ============================================================
// HELPERS
// ============================================================

function formatPercent(value: number | null, digits = 2) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function formatNumber(value: number | null, digits = 2) {
  if (value === null || value === undefined) return "—";

  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatMs(value: number | null) {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(2)} ms`;
}

function formatThroughput(value: number | null) {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(1)} /s`;
}

function formatMemory(value: number | null) {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(1)} MB`;
}

function shortModelName(model: string) {
  if (model === "GigaPath-Flash") return "GigaPath";
  return model;
}

function taskRecords(records: BenchmarkRecord[], task: string) {
  return records.filter((record) => record.task === task);
}

function winnerBy(
  records: BenchmarkRecord[],
  selector: (record: BenchmarkRecord) => number | null,
  higherIsBetter = true
) {
  const valid = records.filter((record) => {
    const value = selector(record);
    return value !== null && value !== undefined;
  });

  if (!valid.length) return null;

  return valid.reduce((best, current) => {
    const bestValue = selector(best);
    const currentValue = selector(current);

    if (bestValue === null || bestValue === undefined) return current;
    if (currentValue === null || currentValue === undefined) return best;

    if (higherIsBetter) {
      return currentValue > bestValue ? current : best;
    }

    return currentValue < bestValue ? current : best;
  });
}


// ============================================================
// SMALL UI COMPONENTS
// ============================================================

function SectionLabel({
  number,
  children,
}: {
  number: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mb-2 flex items-center gap-2">
      <span className="text-[10px] font-semibold tracking-[0.22em] text-cyan-400">
        {number}
      </span>

      <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
        {children}
      </span>
    </div>
  );
}


function MetricCard({
  label,
  value,
  description,
  icon: Icon,
  accent = false,
}: {
  label: string;
  value: string;
  description: string;
  icon: React.ElementType;
  accent?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border p-5 ${
        accent
          ? "border-cyan-500/40 bg-cyan-500/[0.06]"
          : "border-slate-800 bg-slate-950/40"
      }`}
    >
      <div className="mb-4 flex items-center justify-between">
        <span className="text-[10px] font-medium uppercase tracking-[0.18em] text-slate-500">
          {label}
        </span>

        <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
          <Icon size={16} className="text-cyan-400" />
        </div>
      </div>

      <div className="text-2xl font-semibold tracking-tight text-white">
        {value}
      </div>

      <div className="mt-1 text-xs text-slate-500">{description}</div>
    </div>
  );
}


function WinnerBadge() {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/20 bg-emerald-500/[0.08] px-2 py-1 text-[9px] font-semibold uppercase tracking-wider text-emerald-400">
      <CheckCircle2 size={11} />
      Best
    </span>
  );
}


function BenchmarkBar({
  value,
  max,
  label,
  valueLabel,
  winner,
}: {
  value: number | null;
  max: number;
  label: string;
  valueLabel: string;
  winner?: boolean;
}) {
  const percentage =
    value === null || max <= 0 ? 0 : Math.min((value / max) * 100, 100);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-4">
        <span className="text-xs font-medium text-slate-300">{label}</span>

        <div className="flex items-center gap-2">
          {winner && <WinnerBadge />}
          <span className="text-xs font-semibold text-white">
            {valueLabel}
          </span>
        </div>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-cyan-400 transition-all duration-500"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}


// ============================================================
// PAGE
// ============================================================

export default function BenchmarksPage() {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // ----------------------------------------------------------
  // FETCH BENCHMARKS
  // ----------------------------------------------------------

  async function loadBenchmarks(showRefresh = false) {
    try {
      if (showRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError(null);

      const response = await fetch(`${API_BASE}/api/benchmarks`, {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`Benchmark API returned ${response.status}`);
      }

      const result: BenchmarkResponse = await response.json();

      setData(result);
    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to load benchmark data."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadBenchmarks();
  }, []);

  // ----------------------------------------------------------
  // GROUP RECORDS
  // ----------------------------------------------------------

  const records = data?.records ?? [];

  const efficiency = useMemo(
    () => taskRecords(records, "foundation_efficiency"),
    [records]
  );

  const retrieval = useMemo(
    () => taskRecords(records, "image_retrieval"),
    [records]
  );

  const classification = useMemo(
    () => taskRecords(records, "classification"),
    [records]
  );

  const mil = useMemo(
    () => taskRecords(records, "wsi_mil_prototype"),
    [records]
  );

  // ----------------------------------------------------------
  // WINNERS
  // ----------------------------------------------------------

  const classificationWinner = useMemo(
    () => winnerBy(classification, (r) => r.auroc),
    [classification]
  );

  const efficiencyWinner = useMemo(
    () => winnerBy(efficiency, (r) => r.throughput),
    [efficiency]
  );

  const retrievalR1Winner = useMemo(
    () => winnerBy(retrieval, (r) => r.recall_at_1),
    [retrieval]
  );

  const retrievalR5Winner = useMemo(
    () => winnerBy(retrieval, (r) => r.recall_at_5),
    [retrieval]
  );

  const retrievalR10Winner = useMemo(
    () => winnerBy(retrieval, (r) => r.recall_at_10),
    [retrieval]
  );

  // ----------------------------------------------------------
  // MAX VALUES FOR BARS
  // ----------------------------------------------------------

  const maxAccuracy = Math.max(
    ...classification
      .map((r) => r.accuracy ?? 0)
      .filter((v) => Number.isFinite(v)),
    1
  );

  const maxAuroc = Math.max(
    ...classification
      .map((r) => r.auroc ?? 0)
      .filter((v) => Number.isFinite(v)),
    1
  );

  const maxF1 = Math.max(
    ...classification
      .map((r) => r.f1 ?? 0)
      .filter((v) => Number.isFinite(v)),
    1
  );

  const maxThroughput = Math.max(
    ...efficiency
      .map((r) => r.throughput ?? 0)
      .filter((v) => Number.isFinite(v)),
    1
  );

  const maxR5 = Math.max(
    ...retrieval
      .map((r) => r.recall_at_5 ?? 0)
      .filter((v) => Number.isFinite(v)),
    1
  );

  // ----------------------------------------------------------
  // LOADING
  // ----------------------------------------------------------

  if (loading) {
    return (
      <main className="min-h-full px-6 py-8">
        <div className="mx-auto max-w-7xl">
          <div className="mb-8">
            <SectionLabel number="05">MODEL EVALUATION</SectionLabel>

            <h1 className="text-3xl font-semibold tracking-tight text-white">
              Benchmark Laboratory
            </h1>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Compare foundation models across classification quality,
              retrieval performance and inference efficiency.
            </p>
          </div>

          <div className="flex min-h-[420px] items-center justify-center rounded-2xl border border-slate-800 bg-slate-950/40">
            <div className="flex items-center gap-3 text-sm text-slate-400">
              <RefreshCw size={16} className="animate-spin text-cyan-400" />
              Loading benchmark laboratory...
            </div>
          </div>
        </div>
      </main>
    );
  }

  // ----------------------------------------------------------
  // ERROR
  // ----------------------------------------------------------

  if (error) {
    return (
      <main className="min-h-full px-6 py-8">
        <div className="mx-auto max-w-7xl">
          <div className="mb-8">
            <SectionLabel number="05">MODEL EVALUATION</SectionLabel>

            <h1 className="text-3xl font-semibold tracking-tight text-white">
              Benchmark Laboratory
            </h1>

            <p className="mt-2 text-sm text-slate-400">
              Foundation-model evaluation and performance comparison.
            </p>
          </div>

          <div className="rounded-2xl border border-red-500/20 bg-red-500/[0.04] p-8">
            <div className="flex items-center gap-3">
              <Activity className="text-red-400" size={20} />

              <div>
                <h2 className="font-semibold text-white">
                  Benchmark API unavailable
                </h2>

                <p className="mt-1 text-sm text-slate-400">{error}</p>
              </div>
            </div>

            <button
              onClick={() => loadBenchmarks(true)}
              className="mt-6 inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/[0.08] px-4 py-2 text-sm font-medium text-cyan-300 transition hover:bg-cyan-500/[0.14]"
            >
              <RefreshCw size={14} />
              Retry
            </button>
          </div>
        </div>
      </main>
    );
  }

  // ----------------------------------------------------------
  // MAIN PAGE
  // ----------------------------------------------------------

  return (
    <main className="min-h-full px-6 py-8">
      <div className="mx-auto max-w-7xl">

        {/* ================================================== */}
        {/* HEADER */}
        {/* ================================================== */}

        <div className="mb-8 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div>
            <SectionLabel number="05">MODEL EVALUATION</SectionLabel>

            <h1 className="text-3xl font-semibold tracking-tight text-white">
              Benchmark Laboratory
            </h1>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Compare registered pathology foundation models across downstream
              classification, image retrieval and inference efficiency.
            </p>
          </div>

          <button
            onClick={() => loadBenchmarks(true)}
            disabled={refreshing}
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-300 transition hover:border-cyan-500/30 hover:text-cyan-300 disabled:opacity-50"
          >
            <RefreshCw
              size={15}
              className={refreshing ? "animate-spin" : ""}
            />
            Refresh Benchmarks
          </button>
        </div>


        {/* ================================================== */}
        {/* OVERVIEW */}
        {/* ================================================== */}

        <section className="mb-8">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">

            <MetricCard
              label="Models Evaluated"
              value={String(
                new Set(records.map((record) => record.model_id)).size
              )}
              description="Registered foundation encoders"
              icon={Layers3}
            />

            <MetricCard
              label="Benchmark Runs"
              value={String(records.length)}
              description="Tracked evaluation records"
              icon={BarChart3}
            />

            <MetricCard
              label="Tasks Covered"
              value={String(
                new Set(records.map((record) => record.task)).size
              )}
              description="Evaluation categories"
              icon={Target}
            />

            <MetricCard
              label="Best AUROC"
              value={formatPercent(
                classificationWinner?.auroc ?? null
              )}
              description={
                classificationWinner
                  ? classificationWinner.model
                  : "Classification benchmark"
              }
              icon={Trophy}
              accent
            />

          </div>
        </section>


        {/* ================================================== */}
        {/* WINNER CARDS */}
        {/* ================================================== */}

        <section className="mb-8">
          <div className="mb-4">
            <SectionLabel number="01">BENCHMARK HIGHLIGHTS</SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              Leading Performance
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              Winners are calculated directly from the measured benchmark
              records.
            </p>
          </div>

          <div className="grid gap-4 lg:grid-cols-3">

            {/* Classification */}
            <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-5">
              <div className="flex items-start justify-between">
                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
                  <Target size={17} className="text-cyan-400" />
                </div>

                <WinnerBadge />
              </div>

              <div className="mt-5">
                <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
                  Classification
                </div>

                <div className="mt-1 text-lg font-semibold text-white">
                  {classificationWinner?.model ?? "—"}
                </div>

                <div className="mt-3 flex items-end justify-between">
                  <span className="text-xs text-slate-500">
                    AUROC
                  </span>

                  <span className="text-2xl font-semibold text-cyan-300">
                    {formatPercent(
                      classificationWinner?.auroc ?? null
                    )}
                  </span>
                </div>
              </div>
            </div>


            {/* Efficiency */}
            <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-5">
              <div className="flex items-start justify-between">
                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
                  <Gauge size={17} className="text-cyan-400" />
                </div>

                <WinnerBadge />
              </div>

              <div className="mt-5">
                <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
                  Inference Efficiency
                </div>

                <div className="mt-1 text-lg font-semibold text-white">
                  {efficiencyWinner?.model ?? "—"}
                </div>

                <div className="mt-3 flex items-end justify-between">
                  <span className="text-xs text-slate-500">
                    Throughput
                  </span>

                  <span className="text-2xl font-semibold text-cyan-300">
                    {formatThroughput(
                      efficiencyWinner?.throughput ?? null
                    )}
                  </span>
                </div>
              </div>
            </div>


            {/* Retrieval */}
            <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-5">
              <div className="flex items-start justify-between">
                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
                  <Search size={17} className="text-cyan-400" />
                </div>

                <WinnerBadge />
              </div>

              <div className="mt-5">
                <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
                  Retrieval · R@5
                </div>

                <div className="mt-1 text-lg font-semibold text-white">
                  {retrievalR5Winner?.model ?? "—"}
                </div>

                <div className="mt-3 flex items-end justify-between">
                  <span className="text-xs text-slate-500">
                    Recall@5
                  </span>

                  <span className="text-2xl font-semibold text-cyan-300">
                    {formatPercent(
                      retrievalR5Winner?.recall_at_5 ?? null
                    )}
                  </span>
                </div>
              </div>
            </div>

          </div>
        </section>


        {/* ================================================== */}
        {/* CLASSIFICATION */}
        {/* ================================================== */}

        <section className="mb-8 rounded-2xl border border-slate-800 bg-slate-950/40 p-6">

          <div className="mb-6">
            <SectionLabel number="02">DOWNSTREAM EVALUATION</SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              Classification Benchmark
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              PatchCamelyon prototype evaluation using foundation-model
              embeddings.
            </p>
          </div>


          {/* Header */}
          <div className="hidden grid-cols-[1.5fr_repeat(5,1fr)] gap-4 border-b border-slate-800 pb-3 text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500 md:grid">
            <span>Model</span>
            <span>Accuracy</span>
            <span>AUROC</span>
            <span>F1</span>
            <span>Precision</span>
            <span>Recall</span>
          </div>


          <div className="divide-y divide-slate-800/70">

            {classification.map((record) => {
              const isWinner =
                classificationWinner?.model_id === record.model_id;

              return (
                <div
                  key={`classification-${record.model_id}`}
                  className={`grid gap-4 py-5 md:grid-cols-[1.5fr_repeat(5,1fr)] md:items-center ${
                    isWinner
                      ? "rounded-xl bg-cyan-500/[0.025]"
                      : ""
                  }`}
                >

                  <div>
                    <div className="flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-cyan-400" />

                      <span className="text-sm font-semibold text-white">
                        {record.model}
                      </span>

                      {isWinner && <WinnerBadge />}
                    </div>

                    <div className="mt-1 text-[10px] text-slate-600">
                      {record.samples} evaluation samples
                    </div>
                  </div>


                  <div>
                    <div className="text-[9px] uppercase tracking-wider text-slate-600 md:hidden">
                      Accuracy
                    </div>

                    <span className="text-sm font-semibold text-slate-200">
                      {formatPercent(record.accuracy)}
                    </span>
                  </div>


                  <div>
                    <div className="text-[9px] uppercase tracking-wider text-slate-600 md:hidden">
                      AUROC
                    </div>

                    <span
                      className={`text-sm font-semibold ${
                        isWinner
                          ? "text-cyan-300"
                          : "text-slate-200"
                      }`}
                    >
                      {formatPercent(record.auroc)}
                    </span>
                  </div>


                  <div>
                    <div className="text-[9px] uppercase tracking-wider text-slate-600 md:hidden">
                      F1
                    </div>

                    <span className="text-sm font-semibold text-slate-200">
                      {formatPercent(record.f1)}
                    </span>
                  </div>


                  <div>
                    <div className="text-[9px] uppercase tracking-wider text-slate-600 md:hidden">
                      Precision
                    </div>

                    <span className="text-sm font-semibold text-slate-200">
                      {formatPercent(record.precision)}
                    </span>
                  </div>


                  <div>
                    <div className="text-[9px] uppercase tracking-wider text-slate-600 md:hidden">
                      Recall
                    </div>

                    <span className="text-sm font-semibold text-slate-200">
                      {formatPercent(record.recall)}
                    </span>
                  </div>

                </div>
              );
            })}

          </div>


          {/* Visual comparison */}
          <div className="mt-7 grid gap-6 border-t border-slate-800 pt-6 lg:grid-cols-3">

            <div>
              <div className="mb-4 flex items-center gap-2">
                <Target size={14} className="text-cyan-400" />

                <span className="text-xs font-semibold text-white">
                  Accuracy
                </span>
              </div>

              <div className="space-y-4">
                {classification.map((record) => (
                  <BenchmarkBar
                    key={`accuracy-${record.model_id}`}
                    label={shortModelName(record.model)}
                    value={record.accuracy}
                    max={maxAccuracy}
                    valueLabel={formatPercent(record.accuracy)}
                  />
                ))}
              </div>
            </div>


            <div>
              <div className="mb-4 flex items-center gap-2">
                <Activity size={14} className="text-cyan-400" />

                <span className="text-xs font-semibold text-white">
                  AUROC
                </span>
              </div>

              <div className="space-y-4">
                {classification.map((record) => (
                  <BenchmarkBar
                    key={`auroc-${record.model_id}`}
                    label={shortModelName(record.model)}
                    value={record.auroc}
                    max={maxAuroc}
                    valueLabel={formatPercent(record.auroc)}
                    winner={
                      classificationWinner?.model_id ===
                      record.model_id
                    }
                  />
                ))}
              </div>
            </div>


            <div>
              <div className="mb-4 flex items-center gap-2">
                <Medal size={14} className="text-cyan-400" />

                <span className="text-xs font-semibold text-white">
                  F1 Score
                </span>
              </div>

              <div className="space-y-4">
                {classification.map((record) => (
                  <BenchmarkBar
                    key={`f1-${record.model_id}`}
                    label={shortModelName(record.model)}
                    value={record.f1}
                    max={maxF1}
                    valueLabel={formatPercent(record.f1)}
                  />
                ))}
              </div>
            </div>

          </div>

        </section>


        {/* ================================================== */}
        {/* EFFICIENCY */}
        {/* ================================================== */}

        <section className="mb-8 rounded-2xl border border-slate-800 bg-slate-950/40 p-6">

          <div className="mb-6">
            <SectionLabel number="03">SYSTEM PERFORMANCE</SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              Foundation Model Efficiency
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              Measured single-process inference performance on tissue tiles.
            </p>
          </div>


          <div className="grid gap-4 lg:grid-cols-3">

            {efficiency.map((record) => {
              const isWinner =
                efficiencyWinner?.model_id === record.model_id;

              return (
                <div
                  key={`efficiency-${record.model_id}-${record.model}`}
                  className={`rounded-xl border p-5 ${
                    isWinner
                      ? "border-cyan-500/30 bg-cyan-500/[0.035]"
                      : "border-slate-800 bg-slate-950/30"
                  }`}
                >

                  <div className="flex items-start justify-between">
                    <div>
                      <div className="text-sm font-semibold text-white">
                        {record.model}
                      </div>

                      <div className="mt-1 text-[10px] text-slate-600">
                        {record.embedding_dimension}-D embedding
                      </div>
                    </div>

                    {isWinner && <WinnerBadge />}
                  </div>


                  <div className="mt-6 grid grid-cols-2 gap-3">

                    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                      <div className="flex items-center gap-2 text-[9px] uppercase tracking-wider text-slate-600">
                        <Zap size={11} />
                        Latency
                      </div>

                      <div className="mt-2 text-sm font-semibold text-white">
                        {formatMs(record.latency_ms)}
                      </div>
                    </div>


                    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                      <div className="flex items-center gap-2 text-[9px] uppercase tracking-wider text-slate-600">
                        <Gauge size={11} />
                        Throughput
                      </div>

                      <div className="mt-2 text-sm font-semibold text-white">
                        {formatThroughput(record.throughput)}
                      </div>
                    </div>


                    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                      <div className="flex items-center gap-2 text-[9px] uppercase tracking-wider text-slate-600">
                        <Server size={11} />
                        GPU Memory
                      </div>

                      <div className="mt-2 text-sm font-semibold text-white">
                        {formatMemory(record.gpu_memory_mb)}
                      </div>
                    </div>


                    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                      <div className="flex items-center gap-2 text-[9px] uppercase tracking-wider text-slate-600">
                        <Database size={11} />
                        Samples
                      </div>

                      <div className="mt-2 text-sm font-semibold text-white">
                        {formatNumber(record.samples, 0)}
                      </div>
                    </div>

                  </div>

                </div>
              );
            })}

          </div>


          {/* Throughput visual */}
          <div className="mt-7 rounded-xl border border-slate-800 bg-slate-950/30 p-5">

            <div className="mb-5 flex items-center gap-2">
              <Cpu size={15} className="text-cyan-400" />

              <div>
                <div className="text-sm font-semibold text-white">
                  Tile Throughput
                </div>

                <div className="text-[10px] text-slate-600">
                  Higher is better
                </div>
              </div>
            </div>

            <div className="space-y-5">
              {efficiency.map((record) => (
                <BenchmarkBar
                  key={`throughput-${record.model_id}-${record.model}`}
                  label={record.model}
                  value={record.throughput}
                  max={maxThroughput}
                  valueLabel={formatThroughput(record.throughput)}
                  winner={
                    efficiencyWinner?.model_id === record.model_id
                  }
                />
              ))}
            </div>

          </div>

        </section>


        {/* ================================================== */}
        {/* RETRIEVAL */}
        {/* ================================================== */}

        <section className="mb-8 rounded-2xl border border-slate-800 bg-slate-950/40 p-6">

          <div className="mb-6">
            <SectionLabel number="04">MULTIMODAL RETRIEVAL</SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              Image Retrieval Benchmark
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              Retrieval quality measured using Recall@K over the WSI tile
              benchmark.
            </p>
          </div>


          {/* Table */}
          <div className="overflow-x-auto">

            <table className="w-full min-w-[650px] text-left">

              <thead>
                <tr className="border-b border-slate-800">
                  <th className="px-3 py-3 text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    Model
                  </th>

                  <th className="px-3 py-3 text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    Embedding
                  </th>

                  <th className="px-3 py-3 text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    R@1
                  </th>

                  <th className="px-3 py-3 text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    R@5
                  </th>

                  <th className="px-3 py-3 text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    R@10
                  </th>
                </tr>
              </thead>


              <tbody className="divide-y divide-slate-800/70">

                {retrieval.map((record) => (
                  <tr
                    key={`retrieval-${record.model_id}-${record.dataset}`}
                    className="transition hover:bg-slate-900/50"
                  >

                    <td className="px-3 py-4">
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-cyan-400" />

                        <span className="text-sm font-semibold text-white">
                          {record.model}
                        </span>
                      </div>
                    </td>


                    <td className="px-3 py-4 text-sm text-slate-400">
                      {record.embedding_dimension}-D
                    </td>


                    <td className="px-3 py-4">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-slate-300">
                          {formatPercent(record.recall_at_1)}
                        </span>

                        {retrievalR1Winner?.model_id ===
                          record.model_id && (
                          <WinnerBadge />
                        )}
                      </div>
                    </td>


                    <td className="px-3 py-4">
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-sm font-semibold ${
                            retrievalR5Winner?.model_id ===
                            record.model_id
                              ? "text-cyan-300"
                              : "text-slate-300"
                          }`}
                        >
                          {formatPercent(record.recall_at_5)}
                        </span>

                        {retrievalR5Winner?.model_id ===
                          record.model_id && <WinnerBadge />}
                      </div>
                    </td>


                    <td className="px-3 py-4">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-slate-300">
                          {formatPercent(record.recall_at_10)}
                        </span>

                        {retrievalR10Winner?.model_id ===
                          record.model_id && <WinnerBadge />}
                      </div>
                    </td>

                  </tr>
                ))}

              </tbody>

            </table>

          </div>


          {/* R@5 visualization */}
          <div className="mt-7 rounded-xl border border-slate-800 bg-slate-950/30 p-5">

            <div className="mb-5 flex items-center gap-2">
              <Search size={15} className="text-cyan-400" />

              <div>
                <div className="text-sm font-semibold text-white">
                  Recall@5 Comparison
                </div>

                <div className="text-[10px] text-slate-600">
                  Higher is better
                </div>
              </div>
            </div>

            <div className="space-y-5">

              {retrieval.map((record) => (
                <BenchmarkBar
                  key={`retrieval-bar-${record.model_id}-${record.dataset}`}
                  label={record.model}
                  value={record.recall_at_5}
                  max={maxR5}
                  valueLabel={formatPercent(record.recall_at_5)}
                  winner={
                    retrievalR5Winner?.model_id === record.model_id
                  }
                />
              ))}

            </div>

          </div>

        </section>


        {/* ================================================== */}
        {/* TASK LEADERBOARD */}
        {/* ================================================== */}

        <section className="mb-8 rounded-2xl border border-slate-800 bg-slate-950/40 p-6">

          <div className="mb-6">
            <SectionLabel number="05">TASK LEADERBOARD</SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              Model Performance Leaders
            </h2>

            <p className="mt-1 max-w-3xl text-xs leading-5 text-slate-500">
              PathoVerse reports task-specific leaders instead of combining
              unrelated metrics into an arbitrary overall score.
            </p>
          </div>


          <div className="grid gap-4 lg:grid-cols-2">

            {/* Classification */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-5">

              <div className="flex items-center gap-3">
                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
                  <Trophy size={16} className="text-cyan-400" />
                </div>

                <div>
                  <div className="text-sm font-semibold text-white">
                    Classification
                  </div>

                  <div className="text-[10px] text-slate-600">
                    Highest AUROC
                  </div>
                </div>
              </div>

              <div className="mt-5 flex items-end justify-between">
                <div>
                  <div className="text-lg font-semibold text-white">
                    {classificationWinner?.model ?? "—"}
                  </div>

                  <div className="mt-1 text-xs text-slate-600">
                    PatchCamelyon
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-2xl font-semibold text-cyan-300">
                    {formatPercent(
                      classificationWinner?.auroc ?? null
                    )}
                  </div>

                  <div className="text-[9px] uppercase tracking-wider text-slate-600">
                    AUROC
                  </div>
                </div>
              </div>

            </div>


            {/* Efficiency */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-5">

              <div className="flex items-center gap-3">
                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
                  <Gauge size={16} className="text-cyan-400" />
                </div>

                <div>
                  <div className="text-sm font-semibold text-white">
                    Inference Efficiency
                  </div>

                  <div className="text-[10px] text-slate-600">
                    Highest throughput
                  </div>
                </div>
              </div>

              <div className="mt-5 flex items-end justify-between">
                <div>
                  <div className="text-lg font-semibold text-white">
                    {efficiencyWinner?.model ?? "—"}
                  </div>

                  <div className="mt-1 text-xs text-slate-600">
                    45 WSI tissue tiles
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-2xl font-semibold text-cyan-300">
                    {formatThroughput(
                      efficiencyWinner?.throughput ?? null
                    )}
                  </div>

                  <div className="text-[9px] uppercase tracking-wider text-slate-600">
                    Throughput
                  </div>
                </div>
              </div>

            </div>


            {/* Retrieval R@1 */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-5">

              <div className="flex items-center gap-3">
                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
                  <Search size={16} className="text-cyan-400" />
                </div>

                <div>
                  <div className="text-sm font-semibold text-white">
                    Retrieval · R@1
                  </div>

                  <div className="text-[10px] text-slate-600">
                    Highest top-1 recall
                  </div>
                </div>
              </div>

              <div className="mt-5 flex items-end justify-between">
                <div className="text-lg font-semibold text-white">
                  {retrievalR1Winner?.model ?? "—"}
                </div>

                <div className="text-right">
                  <div className="text-2xl font-semibold text-cyan-300">
                    {formatPercent(
                      retrievalR1Winner?.recall_at_1 ?? null
                    )}
                  </div>

                  <div className="text-[9px] uppercase tracking-wider text-slate-600">
                    Recall@1
                  </div>
                </div>
              </div>

            </div>


            {/* Retrieval R@10 */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-5">

              <div className="flex items-center gap-3">
                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
                  <Layers3 size={16} className="text-cyan-400" />
                </div>

                <div>
                  <div className="text-sm font-semibold text-white">
                    Retrieval · R@10
                  </div>

                  <div className="text-[10px] text-slate-600">
                    Highest top-10 recall
                  </div>
                </div>
              </div>

              <div className="mt-5 flex items-end justify-between">
                <div className="text-lg font-semibold text-white">
                  {retrievalR10Winner?.model ?? "—"}
                </div>

                <div className="text-right">
                  <div className="text-2xl font-semibold text-cyan-300">
                    {formatPercent(
                      retrievalR10Winner?.recall_at_10 ?? null
                    )}
                  </div>

                  <div className="text-[9px] uppercase tracking-wider text-slate-600">
                    Recall@10
                  </div>
                </div>
              </div>

            </div>

          </div>

        </section>


        {/* ================================================== */}
        {/* MIL STATUS */}
        {/* ================================================== */}

        <section className="mb-8 rounded-2xl border border-amber-500/15 bg-amber-500/[0.025] p-6">

          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">

            <div className="flex items-start gap-4">

              <div className="rounded-lg border border-amber-500/20 bg-amber-500/[0.06] p-2">
                <Activity size={17} className="text-amber-400" />
              </div>

              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold text-white">
                    Whole-Slide MIL
                  </h3>

                  <span className="rounded-full border border-amber-500/20 bg-amber-500/[0.06] px-2 py-1 text-[9px] font-semibold uppercase tracking-wider text-amber-400">
                    Prototype
                  </span>
                </div>

                <p className="mt-1 max-w-2xl text-xs leading-5 text-slate-500">
                  The current WSI attention-MIL result is retained as a
                  prototype benchmark record and is intentionally excluded
                  from model-performance rankings because the MIL head is
                  untrained.
                </p>
              </div>

            </div>


            <div className="grid grid-cols-3 gap-3">

              <div className="rounded-lg border border-slate-800 bg-slate-950/40 px-4 py-3 text-center">
                <div className="text-lg font-semibold text-white">
                  {mil.length}
                </div>

                <div className="text-[9px] uppercase tracking-wider text-slate-600">
                  Runs
                </div>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-950/40 px-4 py-3 text-center">
                <div className="text-lg font-semibold text-white">
                  {mil[0]?.samples ?? 0}
                </div>

                <div className="text-[9px] uppercase tracking-wider text-slate-600">
                  Tiles
                </div>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-950/40 px-4 py-3 text-center">
                <div className="text-lg font-semibold text-white">
                  {mil[0]?.embedding_dimension ?? "—"}
                </div>

                <div className="text-[9px] uppercase tracking-wider text-slate-600">
                  Embedding
                </div>
              </div>

            </div>

          </div>

        </section>


        {/* ================================================== */}
        {/* FOOTER */}
        {/* ================================================== */}

        <div className="pb-8 text-center text-[10px] text-slate-700">
          PathoVerse AI · Benchmark data served by the PathoVerse evaluation API
        </div>

      </div>
    </main>
  );
}