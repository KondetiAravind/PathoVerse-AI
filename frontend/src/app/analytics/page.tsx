"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Database,
  Gauge,
  Layers3,
  Microscope,
  Network,
  RefreshCw,
  Search,
  Server,
  Target,
  Workflow,
  Zap,
} from "lucide-react";


// ============================================================
// CONFIGURATION
// ============================================================

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";


// ============================================================
// TYPES
// ============================================================

type ModelInfo = {
  name: string;
  model_id: string;
  embedding_dimension: number;
  input_size: number;
  modality: string;
  description: string;
  status: string;
};

type SlideInfo = {
  slide_id: string;
  name?: string;
  width: number;
  height: number;
  levels: number;
  mpp_x?: number | null;
  mpp_y?: number | null;
  objective_power?: number | null;
  tissue_ratio?: number | null;
  tile_count?: number | null;
};

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

type AnalyticsData = {
  models: ModelInfo[];
  slides: SlideInfo[];
  benchmarks: BenchmarkRecord[];
};


// ============================================================
// HELPERS
// ============================================================

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(
  value: number | null | undefined,
  digits = 0
) {
  if (value === null || value === undefined) return "—";

  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatThroughput(value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(1)} tiles/s`;
}

function formatMemory(value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(1)} MB`;
}

function formatLatency(value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(2)} ms`;
}

function getWinner(
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

    if (bestValue === null || bestValue === undefined) {
      return current;
    }

    if (currentValue === null || currentValue === undefined) {
      return best;
    }

    if (higherIsBetter) {
      return currentValue > bestValue ? current : best;
    }

    return currentValue < bestValue ? current : best;
  });
}


// ============================================================
// SMALL COMPONENTS
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


function StatCard({
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
      className={`rounded-2xl border p-5 ${
        accent
          ? "border-cyan-500/35 bg-cyan-500/[0.05]"
          : "border-slate-800 bg-slate-950/40"
      }`}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[9px] font-medium uppercase tracking-[0.18em] text-slate-500">
            {label}
          </div>

          <div className="mt-3 text-2xl font-semibold tracking-tight text-white">
            {value}
          </div>

          <div className="mt-1 text-xs text-slate-600">
            {description}
          </div>
        </div>

        <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2.5">
          <Icon size={16} className="text-cyan-400" />
        </div>
      </div>
    </div>
  );
}


function StatusBadge({
  status,
  warning = false,
}: {
  status: string;
  warning?: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[9px] font-semibold uppercase tracking-wider ${
        warning
          ? "border-amber-500/20 bg-amber-500/[0.06] text-amber-400"
          : "border-emerald-500/20 bg-emerald-500/[0.06] text-emerald-400"
      }`}
    >
      <CheckCircle2 size={10} />
      {status}
    </span>
  );
}


// ============================================================
// PAGE
// ============================================================

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);


  // ==========================================================
  // LOAD PLATFORM DATA
  // ==========================================================

  async function loadAnalytics(showRefresh = false) {
    try {
      if (showRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError(null);

      const [modelsResponse, slidesResponse, benchmarksResponse] =
        await Promise.all([
          fetch(`${API_BASE}/api/models`, {
            cache: "no-store",
          }),

          fetch(`${API_BASE}/api/slides`, {
            cache: "no-store",
          }),

          fetch(`${API_BASE}/api/benchmarks`, {
            cache: "no-store",
          }),
        ]);

      if (!modelsResponse.ok) {
        throw new Error(
          `Models API returned ${modelsResponse.status}`
        );
      }

      if (!slidesResponse.ok) {
        throw new Error(
          `Slides API returned ${slidesResponse.status}`
        );
      }

      if (!benchmarksResponse.ok) {
        throw new Error(
          `Benchmark API returned ${benchmarksResponse.status}`
        );
      }

      const models: ModelInfo[] = await modelsResponse.json();
      const slides: SlideInfo[] = await slidesResponse.json();
      const benchmarkData = await benchmarksResponse.json();

      setData({
        models,
        slides,
        benchmarks: benchmarkData.records ?? [],
      });
    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to load analytics data."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }


  useEffect(() => {
    loadAnalytics();
  }, []);


  // ==========================================================
  // DERIVED DATA
  // ==========================================================

  const models = data?.models ?? [];
  const slides = data?.slides ?? [];
  const benchmarks = data?.benchmarks ?? [];


  const classification = useMemo(
    () =>
      benchmarks.filter(
        (record) => record.task === "classification"
      ),
    [benchmarks]
  );


  const efficiency = useMemo(
    () =>
      benchmarks.filter(
        (record) => record.task === "foundation_efficiency"
      ),
    [benchmarks]
  );


  const retrieval = useMemo(
    () =>
      benchmarks.filter(
        (record) => record.task === "image_retrieval"
      ),
    [benchmarks]
  );


  const mil = useMemo(
    () =>
      benchmarks.filter(
        (record) => record.task === "wsi_mil_prototype"
      ),
    [benchmarks]
  );


  const bestClassifier = useMemo(
    () => getWinner(classification, (r) => r.auroc),
    [classification]
  );


  const fastestModel = useMemo(
    () => getWinner(efficiency, (r) => r.latency_ms, false),
    [efficiency]
  );


  const highestThroughput = useMemo(
    () => getWinner(efficiency, (r) => r.throughput),
    [efficiency]
  );


  const bestRetrieval = useMemo(
    () => getWinner(retrieval, (r) => r.recall_at_5),
    [retrieval]
  );


  const totalSamples = useMemo(
    () =>
      benchmarks.reduce(
        (sum, record) => sum + (record.samples || 0),
        0
      ),
    [benchmarks]
  );


  const availableModels = models.filter(
    (model) => model.status === "available"
  ).length;


  const platformHealth =
    availableModels === models.length &&
    benchmarks.length > 0 &&
    slides.length > 0;


  // ==========================================================
  // LOADING
  // ==========================================================

  if (loading) {
    return (
      <main className="min-h-full px-6 py-8">
        <div className="mx-auto max-w-7xl">

          <SectionLabel number="06">
            PLATFORM INTELLIGENCE
          </SectionLabel>

          <h1 className="text-3xl font-semibold tracking-tight text-white">
            Analytics Center
          </h1>

          <p className="mt-2 text-sm text-slate-400">
            Aggregating foundation-model, WSI, retrieval and evaluation
            signals from the PathoVerse platform.
          </p>

          <div className="mt-8 flex min-h-[420px] items-center justify-center rounded-2xl border border-slate-800 bg-slate-950/40">
            <div className="flex items-center gap-3 text-sm text-slate-400">
              <RefreshCw
                size={16}
                className="animate-spin text-cyan-400"
              />

              Loading platform analytics...
            </div>
          </div>

        </div>
      </main>
    );
  }


  // ==========================================================
  // ERROR
  // ==========================================================

  if (error) {
    return (
      <main className="min-h-full px-6 py-8">
        <div className="mx-auto max-w-7xl">

          <SectionLabel number="06">
            PLATFORM INTELLIGENCE
          </SectionLabel>

          <h1 className="text-3xl font-semibold tracking-tight text-white">
            Analytics Center
          </h1>

          <div className="mt-8 rounded-2xl border border-red-500/20 bg-red-500/[0.04] p-8">

            <div className="flex items-start gap-4">
              <Activity
                size={20}
                className="mt-0.5 text-red-400"
              />

              <div>
                <h2 className="font-semibold text-white">
                  Analytics data unavailable
                </h2>

                <p className="mt-1 text-sm text-slate-400">
                  {error}
                </p>
              </div>
            </div>

            <button
              onClick={() => loadAnalytics(true)}
              className="mt-6 inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/[0.08] px-4 py-2 text-sm text-cyan-300 transition hover:bg-cyan-500/[0.14]"
            >
              <RefreshCw size={14} />
              Retry
            </button>

          </div>

        </div>
      </main>
    );
  }


  // ==========================================================
  // MAIN
  // ==========================================================

  return (
    <main className="min-h-full px-6 py-8">

      <div className="mx-auto max-w-7xl">


        {/* ================================================== */}
        {/* HEADER */}
        {/* ================================================== */}

        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">

          <div>

            <SectionLabel number="06">
              PLATFORM INTELLIGENCE
            </SectionLabel>

            <h1 className="text-3xl font-semibold tracking-tight text-white">
              Analytics Center
            </h1>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              System-level view of PathoVerse foundation models, whole-slide
              studies, evaluation workloads and multimodal retrieval.
            </p>

          </div>


          <div className="flex items-center gap-3">

            <div
              className={`inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-xs ${
                platformHealth
                  ? "border-emerald-500/20 bg-emerald-500/[0.05] text-emerald-400"
                  : "border-amber-500/20 bg-amber-500/[0.05] text-amber-400"
              }`}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              {platformHealth
                ? "Platform Operational"
                : "Partial Data"}
            </div>


            <button
              onClick={() => loadAnalytics(true)}
              disabled={refreshing}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-300 transition hover:border-cyan-500/30 hover:text-cyan-300 disabled:opacity-50"
            >
              <RefreshCw
                size={14}
                className={
                  refreshing ? "animate-spin" : ""
                }
              />

              Refresh
            </button>

          </div>

        </div>


        {/* ================================================== */}
        {/* PLATFORM SUMMARY */}
        {/* ================================================== */}

        <section className="mb-8">

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">

            <StatCard
              label="Foundation Models"
              value={String(models.length)}
              description={`${availableModels} currently available`}
              icon={BrainCircuit}
              accent
            />

            <StatCard
              label="Whole-Slide Studies"
              value={String(slides.length)}
              description="Registered WSI studies"
              icon={Microscope}
            />

            <StatCard
              label="Benchmark Records"
              value={String(benchmarks.length)}
              description={`${totalSamples.toLocaleString()} evaluated samples`}
              icon={BarChart3}
            />

            <StatCard
              label="Evaluation Tasks"
              value={String(
                new Set(benchmarks.map((r) => r.task)).size
              )}
              description="Classification · retrieval · efficiency · MIL"
              icon={Target}
            />

          </div>

        </section>


        {/* ================================================== */}
        {/* PLATFORM SIGNAL */}
        {/* ================================================== */}

        <section className="mb-8 rounded-2xl border border-slate-800 bg-slate-950/40 p-6">

          <div className="mb-6">

            <SectionLabel number="01">
              PLATFORM SIGNAL
            </SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              Current System Intelligence
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              Key signals derived from the registered model and benchmark
              metadata.
            </p>

          </div>


          <div className="grid gap-4 lg:grid-cols-4">

            {/* Classification */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-5">

              <div className="flex items-center justify-between">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
                  <Target
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <StatusBadge status="measured" />

              </div>


              <div className="mt-5">

                <div className="text-[9px] uppercase tracking-[0.18em] text-slate-600">
                  Best classification
                </div>

                <div className="mt-1 text-base font-semibold text-white">
                  {bestClassifier?.model ?? "—"}
                </div>

                <div className="mt-3 text-2xl font-semibold text-cyan-300">
                  {formatPercent(bestClassifier?.auroc)}
                </div>

                <div className="text-[9px] uppercase tracking-wider text-slate-600">
                  AUROC
                </div>

              </div>

            </div>


            {/* Throughput */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-5">

              <div className="flex items-center justify-between">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
                  <Gauge
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <StatusBadge status="measured" />

              </div>


              <div className="mt-5">

                <div className="text-[9px] uppercase tracking-[0.18em] text-slate-600">
                  Highest throughput
                </div>

                <div className="mt-1 text-base font-semibold text-white">
                  {highestThroughput?.model ?? "—"}
                </div>

                <div className="mt-3 text-2xl font-semibold text-cyan-300">
                  {formatThroughput(
                    highestThroughput?.throughput
                  )}
                </div>

              </div>

            </div>


            {/* Retrieval */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-5">

              <div className="flex items-center justify-between">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.08] p-2">
                  <Search
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <StatusBadge status="measured" />

              </div>


              <div className="mt-5">

                <div className="text-[9px] uppercase tracking-[0.18em] text-slate-600">
                  Best retrieval
                </div>

                <div className="mt-1 text-base font-semibold text-white">
                  {bestRetrieval?.model ?? "—"}
                </div>

                <div className="mt-3 text-2xl font-semibold text-cyan-300">
                  {formatPercent(
                    bestRetrieval?.recall_at_5
                  )}
                </div>

                <div className="text-[9px] uppercase tracking-wider text-slate-600">
                  Recall@5
                </div>

              </div>

            </div>


            {/* MIL */}
            <div className="rounded-xl border border-amber-500/15 bg-amber-500/[0.025] p-5">

              <div className="flex items-center justify-between">

                <div className="rounded-lg border border-amber-500/20 bg-amber-500/[0.06] p-2">
                  <Activity
                    size={16}
                    className="text-amber-400"
                  />
                </div>

                <StatusBadge
                  status="prototype"
                  warning
                />

              </div>


              <div className="mt-5">

                <div className="text-[9px] uppercase tracking-[0.18em] text-slate-600">
                  Whole-slide MIL
                </div>

                <div className="mt-1 text-base font-semibold text-white">
                  {mil.length > 0
                    ? "Attention MIL"
                    : "Not available"}
                </div>

                <div className="mt-3 text-2xl font-semibold text-amber-300">
                  {mil.length > 0
                    ? `${mil[0].samples} tiles`
                    : "—"}
                </div>

                <div className="text-[9px] uppercase tracking-wider text-slate-600">
                  Prototype analysis
                </div>

              </div>

            </div>

          </div>

        </section>


        {/* ================================================== */}
        {/* MODEL MATRIX */}
        {/* ================================================== */}

        <section className="mb-8 rounded-2xl border border-slate-800 bg-slate-950/40 p-6">

          <div className="mb-6">

            <SectionLabel number="02">
              FOUNDATION MODEL MATRIX
            </SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              Registered Model Capabilities
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              Model registry metadata combined with measured downstream
              performance.
            </p>

          </div>


          <div className="overflow-x-auto">

            <table className="w-full min-w-[850px]">

              <thead>

                <tr className="border-b border-slate-800">

                  <th className="px-3 py-3 text-left text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    Model
                  </th>

                  <th className="px-3 py-3 text-left text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    Modality
                  </th>

                  <th className="px-3 py-3 text-left text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    Embedding
                  </th>

                  <th className="px-3 py-3 text-left text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    Input
                  </th>

                  <th className="px-3 py-3 text-left text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    AUROC
                  </th>

                  <th className="px-3 py-3 text-left text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    Throughput
                  </th>

                  <th className="px-3 py-3 text-left text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500">
                    Status
                  </th>

                </tr>

              </thead>


              <tbody className="divide-y divide-slate-800/70">

                {models.map((model) => {

                  const classificationRecord =
                    classification.find(
                      (record) =>
                        record.model_id === model.model_id
                    );

                  const efficiencyRecord =
                    efficiency.find(
                      (record) =>
                        record.model_id === model.model_id
                    );

                  return (
                    <tr
                      key={model.model_id}
                      className="transition hover:bg-slate-900/50"
                    >

                      <td className="px-3 py-4">

                        <div className="flex items-center gap-2">

                          <span className="h-2 w-2 rounded-full bg-cyan-400" />

                          <span className="text-sm font-semibold text-white">
                            {model.name}
                          </span>

                        </div>

                      </td>


                      <td className="px-3 py-4">

                        <span className="rounded-full border border-slate-700 bg-slate-900 px-2 py-1 text-[9px] text-slate-400">
                          {model.modality}
                        </span>

                      </td>


                      <td className="px-3 py-4 text-sm text-slate-300">
                        {model.embedding_dimension}-D
                      </td>


                      <td className="px-3 py-4 text-sm text-slate-400">
                        {model.input_size}²
                      </td>


                      <td className="px-3 py-4">

                        <span
                          className={
                            classificationRecord
                              ? "text-sm font-semibold text-cyan-300"
                              : "text-sm text-slate-600"
                          }
                        >
                          {formatPercent(
                            classificationRecord?.auroc
                          )}
                        </span>

                      </td>


                      <td className="px-3 py-4 text-sm text-slate-300">
                        {formatThroughput(
                          efficiencyRecord?.throughput
                        )}
                      </td>


                      <td className="px-3 py-4">

                        <span className="inline-flex items-center gap-1.5 text-xs text-emerald-400">

                          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />

                          {model.status}

                        </span>

                      </td>

                    </tr>
                  );
                })}

              </tbody>

            </table>

          </div>

        </section>


        {/* ================================================== */}
        {/* QUALITY VS EFFICIENCY */}
        {/* ================================================== */}

        <section className="mb-8">

          <div className="mb-5">

            <SectionLabel number="03">
              QUALITY × EFFICIENCY
            </SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              Model Operating Profile
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              Compare measured classification quality against inference
              efficiency without collapsing the metrics into a single score.
            </p>

          </div>


          <div className="grid gap-4 lg:grid-cols-3">

            {models.map((model) => {

              const classificationRecord =
                classification.find(
                  (record) =>
                    record.model_id === model.model_id
                );

              const efficiencyRecord =
                efficiency.find(
                  (record) =>
                    record.model_id === model.model_id
                );

              const isClassificationWinner =
                bestClassifier?.model_id === model.model_id;

              const isEfficiencyWinner =
                highestThroughput?.model_id === model.model_id;

              return (
                <div
                  key={`profile-${model.model_id}`}
                  className={`rounded-2xl border p-5 ${
                    isClassificationWinner ||
                    isEfficiencyWinner
                      ? "border-cyan-500/30 bg-cyan-500/[0.025]"
                      : "border-slate-800 bg-slate-950/40"
                  }`}
                >

                  <div className="flex items-start justify-between">

                    <div>

                      <div className="text-sm font-semibold text-white">
                        {model.name}
                      </div>

                      <div className="mt-1 text-[10px] text-slate-600">
                        {model.modality}
                      </div>

                    </div>


                    <div className="flex gap-1">

                      {isClassificationWinner && (
                        <span className="rounded-full border border-cyan-500/20 bg-cyan-500/[0.07] px-2 py-1 text-[8px] font-semibold uppercase tracking-wider text-cyan-300">
                          Quality Leader
                        </span>
                      )}

                      {isEfficiencyWinner && (
                        <span className="rounded-full border border-emerald-500/20 bg-emerald-500/[0.07] px-2 py-1 text-[8px] font-semibold uppercase tracking-wider text-emerald-300">
                          Speed Leader
                        </span>
                      )}

                    </div>

                  </div>


                  <div className="mt-6 space-y-5">

                    <div>

                      <div className="mb-2 flex justify-between">

                        <span className="text-[9px] uppercase tracking-wider text-slate-600">
                          Classification AUROC
                        </span>

                        <span className="text-xs font-semibold text-white">
                          {formatPercent(
                            classificationRecord?.auroc
                          )}
                        </span>

                      </div>


                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">

                        <div
                          className="h-full rounded-full bg-cyan-400"
                          style={{
                            width: `${Math.min(
                              (classificationRecord?.auroc ??
                                0) * 100,
                              100
                            )}%`,
                          }}
                        />

                      </div>

                    </div>


                    <div>

                      <div className="mb-2 flex justify-between">

                        <span className="text-[9px] uppercase tracking-wider text-slate-600">
                          Throughput
                        </span>

                        <span className="text-xs font-semibold text-white">
                          {formatThroughput(
                            efficiencyRecord?.throughput
                          )}
                        </span>

                      </div>


                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">

                        <div
                          className="h-full rounded-full bg-emerald-400"
                          style={{
                            width: `${Math.min(
                              ((efficiencyRecord?.throughput ??
                                0) /
                                Math.max(
                                  highestThroughput?.throughput ??
                                    1,
                                  1
                                )) *
                                100,
                              100
                            )}%`,
                          }}
                        />

                      </div>

                    </div>


                    <div className="grid grid-cols-2 gap-3">

                      <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">

                        <div className="text-[9px] uppercase tracking-wider text-slate-600">
                          Latency
                        </div>

                        <div className="mt-1 text-sm font-semibold text-white">
                          {formatLatency(
                            efficiencyRecord?.latency_ms
                          )}
                        </div>

                      </div>


                      <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">

                        <div className="text-[9px] uppercase tracking-wider text-slate-600">
                          GPU Memory
                        </div>

                        <div className="mt-1 text-sm font-semibold text-white">
                          {formatMemory(
                            efficiencyRecord?.gpu_memory_mb
                          )}
                        </div>

                      </div>

                    </div>

                  </div>

                </div>
              );
            })}

          </div>

        </section>


        {/* ================================================== */}
        {/* RETRIEVAL INTELLIGENCE */}
        {/* ================================================== */}

        <section className="mb-8 rounded-2xl border border-slate-800 bg-slate-950/40 p-6">

          <div className="mb-6">

            <SectionLabel number="04">
              MULTIMODAL RETRIEVAL
            </SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              Retrieval Intelligence
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              Foundation-model retrieval performance across the current WSI
              tile benchmark.
            </p>

          </div>


          <div className="grid gap-4 lg:grid-cols-3">

            {retrieval.map((record) => {

              const bestR1 =
                getWinner(
                  retrieval,
                  (r) => r.recall_at_1
                )?.model_id === record.model_id;

              const bestR5 =
                getWinner(
                  retrieval,
                  (r) => r.recall_at_5
                )?.model_id === record.model_id;

              const bestR10 =
                getWinner(
                  retrieval,
                  (r) => r.recall_at_10
                )?.model_id === record.model_id;

              return (
                <div
                  key={`retrieval-${record.model_id}`}
                  className="rounded-xl border border-slate-800 bg-slate-950/30 p-5"
                >

                  <div className="flex items-center justify-between">

                    <div className="flex items-center gap-2">

                      <Search
                        size={15}
                        className="text-cyan-400"
                      />

                      <span className="text-sm font-semibold text-white">
                        {record.model}
                      </span>

                    </div>

                    <span className="text-[9px] text-slate-600">
                      {record.embedding_dimension}-D
                    </span>

                  </div>


                  <div className="mt-6 space-y-4">

                    <div>

                      <div className="flex justify-between">

                        <span className="text-[9px] uppercase tracking-wider text-slate-600">
                          Recall@1
                        </span>

                        <div className="flex items-center gap-2">

                          {bestR1 && (
                            <span className="text-[8px] font-semibold text-cyan-300">
                              BEST
                            </span>
                          )}

                          <span className="text-sm font-semibold text-white">
                            {formatPercent(
                              record.recall_at_1
                            )}
                          </span>

                        </div>

                      </div>

                    </div>


                    <div>

                      <div className="flex justify-between">

                        <span className="text-[9px] uppercase tracking-wider text-slate-600">
                          Recall@5
                        </span>

                        <div className="flex items-center gap-2">

                          {bestR5 && (
                            <span className="text-[8px] font-semibold text-cyan-300">
                              BEST
                            </span>
                          )}

                          <span className="text-sm font-semibold text-cyan-300">
                            {formatPercent(
                              record.recall_at_5
                            )}
                          </span>

                        </div>

                      </div>

                    </div>


                    <div>

                      <div className="flex justify-between">

                        <span className="text-[9px] uppercase tracking-wider text-slate-600">
                          Recall@10
                        </span>

                        <div className="flex items-center gap-2">

                          {bestR10 && (
                            <span className="text-[8px] font-semibold text-cyan-300">
                              BEST
                            </span>
                          )}

                          <span className="text-sm font-semibold text-white">
                            {formatPercent(
                              record.recall_at_10
                            )}
                          </span>

                        </div>

                      </div>

                    </div>

                  </div>

                </div>
              );
            })}

          </div>

        </section>


        {/* ================================================== */}
        {/* WSI PIPELINE */}
        {/* ================================================== */}

        <section className="mb-8">

          <div className="mb-5">

            <SectionLabel number="05">
              WHOLE-SLIDE PIPELINE
            </SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              PathoVerse Analysis Flow
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              Current platform architecture from WSI ingestion through
              foundation-model analysis and explainability.
            </p>

          </div>


          <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-6">

            <div className="grid gap-3 md:grid-cols-5">

              {/* WSI */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2 w-fit">
                  <Microscope
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <div className="mt-4 text-sm font-semibold text-white">
                  WSI
                </div>

                <div className="mt-1 text-[10px] leading-4 text-slate-600">
                  Whole-slide ingestion
                </div>

              </div>


              <div className="hidden items-center justify-center md:flex">
                <ArrowRight
                  size={17}
                  className="text-slate-700"
                />
              </div>


              {/* Tiling */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2 w-fit">
                  <Layers3
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <div className="mt-4 text-sm font-semibold text-white">
                  Tissue Tiling
                </div>

                <div className="mt-1 text-[10px] leading-4 text-slate-600">
                  Tissue-aware tile extraction
                </div>

              </div>


              <div className="hidden items-center justify-center md:flex">
                <ArrowRight
                  size={17}
                  className="text-slate-700"
                />
              </div>


              {/* Foundation */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2 w-fit">
                  <BrainCircuit
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <div className="mt-4 text-sm font-semibold text-white">
                  Foundation Models
                </div>

                <div className="mt-1 text-[10px] leading-4 text-slate-600">
                  ViT · GigaPath · CONCH
                </div>

              </div>


              <div className="hidden items-center justify-center md:flex">
                <ArrowRight
                  size={17}
                  className="text-slate-700"
                />
              </div>


              {/* Analysis */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2 w-fit">
                  <Network
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <div className="mt-4 text-sm font-semibold text-white">
                  Analysis
                </div>

                <div className="mt-1 text-[10px] leading-4 text-slate-600">
                  Classification · MIL · retrieval
                </div>

              </div>


              <div className="hidden items-center justify-center md:flex">
                <ArrowRight
                  size={17}
                  className="text-slate-700"
                />
              </div>


              {/* Explainability */}
              <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/[0.025] p-4">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2 w-fit">
                  <Workflow
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <div className="mt-4 text-sm font-semibold text-white">
                  Explainability
                </div>

                <div className="mt-1 text-[10px] leading-4 text-slate-600">
                  Attention regions · heatmaps
                </div>

              </div>

            </div>

          </div>

        </section>


        {/* ================================================== */}
        {/* WSI STUDY STATUS */}
        {/* ================================================== */}

        <section className="mb-8 rounded-2xl border border-slate-800 bg-slate-950/40 p-6">

          <div className="mb-5">

            <SectionLabel number="06">
              WSI INTELLIGENCE
            </SectionLabel>

            <h2 className="text-xl font-semibold text-white">
              Registered Whole-Slide Studies
            </h2>

          </div>


          <div className="space-y-3">

            {slides.map((slide) => (

              <div
                key={slide.slide_id}
                className="flex flex-col gap-4 rounded-xl border border-slate-800 bg-slate-950/30 p-5 lg:flex-row lg:items-center lg:justify-between"
              >

                <div className="flex items-center gap-4">

                  <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2.5">
                    <Database
                      size={17}
                      className="text-cyan-400"
                    />
                  </div>

                  <div>

                    <div className="text-sm font-semibold text-white">
                      {slide.slide_id}
                    </div>

                    <div className="mt-1 text-xs text-slate-600">
                      Aperio WSI · {slide.width} × {slide.height}px
                    </div>

                  </div>

                </div>


                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">

                  <div className="rounded-lg border border-slate-800 px-4 py-3 text-center">

                    <div className="text-sm font-semibold text-white">
                      {slide.levels}
                    </div>

                    <div className="text-[8px] uppercase tracking-wider text-slate-600">
                      Levels
                    </div>

                  </div>


                  <div className="rounded-lg border border-slate-800 px-4 py-3 text-center">

                    <div className="text-sm font-semibold text-white">
                      {slide.mpp_x
                        ? slide.mpp_x.toFixed(3)
                        : "—"}
                    </div>

                    <div className="text-[8px] uppercase tracking-wider text-slate-600">
                      MPP
                    </div>

                  </div>


                  <div className="rounded-lg border border-slate-800 px-4 py-3 text-center">

                    <div className="text-sm font-semibold text-white">
                      {slide.tile_count ?? "—"}
                    </div>

                    <div className="text-[8px] uppercase tracking-wider text-slate-600">
                      Tiles
                    </div>

                  </div>


                  <div className="rounded-lg border border-slate-800 px-4 py-3 text-center">

                    <div className="text-sm font-semibold text-emerald-400">
                      Registered
                    </div>

                    <div className="text-[8px] uppercase tracking-wider text-slate-600">
                      Status
                    </div>

                  </div>

                </div>

              </div>

            ))}

          </div>

        </section>


        {/* ================================================== */}
        {/* COMPUTE SIGNAL */}
        {/* ================================================== */}

        <section className="mb-8">

          <div className="grid gap-4 md:grid-cols-3">

            <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-5">

              <div className="flex items-center gap-3">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2">
                  <Cpu
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <div>

                  <div className="text-sm font-semibold text-white">
                    GPU Inference
                  </div>

                  <div className="text-[10px] text-slate-600">
                    Measured model execution
                  </div>

                </div>

              </div>


              <div className="mt-5 text-2xl font-semibold text-white">
                {formatThroughput(
                  highestThroughput?.throughput
                )}
              </div>

              <div className="mt-1 text-xs text-slate-600">
                Highest measured tile throughput
              </div>

            </div>


            <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-5">

              <div className="flex items-center gap-3">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2">
                  <Server
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <div>

                  <div className="text-sm font-semibold text-white">
                    Fastest Encoder
                  </div>

                  <div className="text-[10px] text-slate-600">
                    Lowest measured latency
                  </div>

                </div>

              </div>


              <div className="mt-5 text-2xl font-semibold text-white">
                {fastestModel?.model ?? "—"}
              </div>

              <div className="mt-1 text-xs text-slate-600">
                {formatLatency(
                  fastestModel?.latency_ms
                )}
                {" "}per tile
              </div>

            </div>


            <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-5">

              <div className="flex items-center gap-3">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2">
                  <Zap
                    size={16}
                    className="text-cyan-400"
                  />
                </div>

                <div>

                  <div className="text-sm font-semibold text-white">
                    Active Evaluation
                  </div>

                  <div className="text-[10px] text-slate-600">
                    Benchmark records available
                  </div>

                </div>

              </div>


              <div className="mt-5 text-2xl font-semibold text-white">
                {benchmarks.length}
              </div>

              <div className="mt-1 text-xs text-slate-600">
                Measured and prototype records
              </div>

            </div>

          </div>

        </section>


        {/* ================================================== */}
        {/* FOOTER */}
        {/* ================================================== */}

        <div className="pb-8 text-center text-[10px] text-slate-700">
          PathoVerse AI · Platform analytics derived from registered API data
        </div>

      </div>
    </main>
  );
}