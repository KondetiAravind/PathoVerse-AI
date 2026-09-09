"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Brain,
  CheckCircle2,
  Cpu,
  Gauge,
  HardDrive,
  Layers3,
  Search,
  Zap,
} from "lucide-react";

import AppShell from "@/components/layout/AppShell";
import Card from "@/components/ui/Card";

import {
  getBenchmarks,
  getModels,
} from "@/lib/api";

import type {
  BenchmarkRecord,
  ModelInfo,
} from "@/types/api";

const MODEL_META: Record<
  string,
  {
    role: string;
  }
> = {
  "vit-b-16": {
    role: "Vision Encoder",
  },

  "gigapath-flash": {
    role: "Pathology Encoder",
  },

  conch: {
    role: "Vision-Language",
  },
};

function formatMetric(
  value: number | null | undefined,
  digits = 2
) {
  if (value === null || value === undefined) {
    return "—";
  }

  return value.toFixed(digits);
}

function findBenchmark(
  benchmarks: BenchmarkRecord[],
  task: string,
  modelId: string
) {
  return benchmarks.find(
    (item) =>
      item.task === task &&
      item.model_id === modelId
  );
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [benchmarks, setBenchmarks] =
    useState<BenchmarkRecord[]>([]);

  const [selectedModel, setSelectedModel] =
    useState<string>("gigapath-flash");

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError(null);

        const [
          modelData,
          benchmarkData,
        ] = await Promise.all([
          getModels(),
          getBenchmarks(),
        ]);

        const modelList: ModelInfo[] =
          Array.isArray(modelData)
            ? modelData
            : modelData.models ?? [];

        const benchmarkList: BenchmarkRecord[] =
          Array.isArray(benchmarkData)
            ? benchmarkData
            : benchmarkData.records ?? [];

        setModels(modelList);
        setBenchmarks(benchmarkList);

        const giga = modelList.find(
          (model) =>
            model.model_id === "gigapath-flash"
        );

        if (giga) {
          setSelectedModel(giga.model_id);
        } else if (modelList.length > 0) {
          setSelectedModel(
            modelList[0].model_id
          );
        }
      } catch (err) {
        console.error(err);

        setError(
          "Unable to load foundation model data."
        );
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);

  const selected =
    models.find(
      (model) =>
        model.model_id === selectedModel
    ) ?? null;

  const efficiency = selected
    ? findBenchmark(
        benchmarks,
        "foundation_efficiency",
        selected.model_id
      )
    : undefined;

  const classification = selected
    ? findBenchmark(
        benchmarks,
        "classification",
        selected.model_id
      )
    : undefined;

  const retrieval = selected
    ? findBenchmark(
        benchmarks,
        "image_retrieval",
        selected.model_id
      )
    : undefined;

  const efficiencyRanking = useMemo(() => {
    return models
      .map((model) => {
        const benchmark =
          findBenchmark(
            benchmarks,
            "foundation_efficiency",
            model.model_id
          );

        return {
          model,
          benchmark,
        };
      })
      .filter(
        (item) => item.benchmark !== undefined
      )
      .sort(
        (a, b) =>
          (b.benchmark?.throughput ?? 0) -
          (a.benchmark?.throughput ?? 0)
      );
  }, [models, benchmarks]);

  if (loading) {
    return (
      <AppShell>
        <div className="flex min-h-[70vh] items-center justify-center">
          <div className="flex items-center gap-3 text-sm text-[#8fa3b5]">
            <Activity
              size={17}
              className="animate-pulse text-[#37d5c3]"
            />

            Loading foundation models...
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <div className="mx-auto max-w-[1600px] p-6">

        {/* HEADER */}
        <section className="mb-7">
          <div className="mb-2 flex items-center gap-2">
            <Brain
              size={18}
              className="text-[#37d5c3]"
            />

            <span className="text-xs uppercase tracking-[0.18em] text-[#37d5c3]">
              Foundation Models
            </span>
          </div>

          <h1 className="text-2xl font-semibold text-white">
            Model Explorer
          </h1>

          <p className="mt-1 max-w-3xl text-sm leading-6 text-[#8fa3b5]">
            Explore registered pathology foundation
            models, embedding characteristics and
            measured benchmark performance.
          </p>
        </section>

        {error && (
          <div className="mb-5 rounded-lg border border-red-900/50 bg-red-950/20 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* MODEL CARDS */}
        <section className="grid gap-4 lg:grid-cols-3">
          {models.map((model) => {
            const active =
              model.model_id === selectedModel;

            const meta =
              MODEL_META[model.model_id] ?? {
                role: "Foundation Model",
              };

            const benchmark =
              findBenchmark(
                benchmarks,
                "foundation_efficiency",
                model.model_id
              );

            return (
              <button
                key={model.model_id}
                onClick={() =>
                  setSelectedModel(
                    model.model_id
                  )
                }
                className="text-left"
              >
                <Card
                  className={`h-full p-5 transition ${
                    active
                      ? "border-[#37d5c3] ring-1 ring-[#37d5c3]/40"
                      : "hover:border-[#315061]"
                  }`}
                >

                  <div className="flex items-start justify-between">

                    <div className="flex items-center gap-3">

                      <div
                        className={`rounded-lg p-2.5 ${
                          active
                            ? "bg-[#37d5c3]/10"
                            : "bg-[#101e2b]"
                        }`}
                      >
                        <Brain
                          size={19}
                          className={
                            active
                              ? "text-[#37d5c3]"
                              : "text-[#8fa3b5]"
                          }
                        />
                      </div>

                      <div>
                        <p className="text-sm font-semibold text-white">
                          {model.name}
                        </p>

                        <p className="text-xs text-[#60788b]">
                          {meta.role}
                        </p>
                      </div>

                    </div>

                    {model.status ===
                      "available" && (
                      <CheckCircle2
                        size={17}
                        className="text-[#37d5c3]"
                      />
                    )}

                  </div>

                  <p className="mt-4 min-h-[60px] text-xs leading-5 text-[#8fa3b5]">
                    {model.description}
                  </p>

                  <div className="mt-5 grid grid-cols-3 gap-2">

                    <MiniMetric
                      label="Embedding"
                      value={`${model.embedding_dimension}-d`}
                    />

                    <MiniMetric
                      label="Input"
                      value={`${model.input_size}px`}
                    />

                    <MiniMetric
                      label="Throughput"
                      value={
                        benchmark?.throughput != null
                          ? `${formatMetric(
                              benchmark.throughput,
                              0
                            )}/s`
                          : "—"
                      }
                    />

                  </div>

                </Card>
              </button>
            );
          })}
        </section>

        {/* SELECTED MODEL */}
        {selected && (
          <section className="mt-6 grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">

            {/* PERFORMANCE PANEL */}
            <Card className="p-6">

              <div className="flex items-start justify-between">

                <div>
                  <div className="flex items-center gap-3">

                    <h2 className="text-lg font-semibold text-white">
                      {selected.name}
                    </h2>

                    <span className="rounded-full border border-[#37d5c3]/30 bg-[#37d5c3]/10 px-2.5 py-1 text-[10px] text-[#37d5c3]">
                      {selected.modality}
                    </span>

                  </div>

                  <p className="mt-1 text-xs text-[#60788b]">
                    {MODEL_META[
                      selected.model_id
                    ]?.role ?? "Foundation Model"}
                  </p>
                </div>

                <span className="rounded-full border border-[#1d3040] px-3 py-1 text-[10px] text-[#37d5c3]">
                  {selected.status.toUpperCase()}
                </span>

              </div>

              {/* EFFICIENCY */}
              <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">

                <MetricCard
                  icon={Zap}
                  label="Latency"
                  value={
                    efficiency?.latency_ms != null
                      ? `${formatMetric(
                          efficiency.latency_ms,
                          3
                        )} ms`
                      : "—"
                  }
                />

                <MetricCard
                  icon={Gauge}
                  label="Throughput"
                  value={
                    efficiency?.throughput != null
                      ? `${formatMetric(
                          efficiency.throughput,
                          1
                        )} tiles/s`
                      : "—"
                  }
                />

                <MetricCard
                  icon={HardDrive}
                  label="GPU Memory"
                  value={
                    efficiency?.gpu_memory_mb != null
                      ? `${formatMetric(
                          efficiency.gpu_memory_mb,
                          1
                        )} MB`
                      : "—"
                  }
                />

                <MetricCard
                  icon={Layers3}
                  label="Embedding"
                  value={`${selected.embedding_dimension}-D`}
                />

              </div>

              {/* DOWNSTREAM */}
              <div className="mt-7">

                <div className="mb-4">
                  <h3 className="text-sm font-medium text-white">
                    Downstream Evaluation
                  </h3>

                  <p className="mt-1 text-xs text-[#60788b]">
                    Prototype benchmark results from
                    the PathoVerse evaluation pipeline.
                  </p>
                </div>

                <div className="grid gap-4 md:grid-cols-2">

                  <EvaluationCard
                    title="Classification"
                    icon={Brain}
                    values={[
                      [
                        "Accuracy",
                        classification?.accuracy,
                      ],
                      [
                        "AUROC",
                        classification?.auroc,
                      ],
                      [
                        "F1",
                        classification?.f1,
                      ],
                      [
                        "Precision",
                        classification?.precision,
                      ],
                      [
                        "Recall",
                        classification?.recall,
                      ],
                    ]}
                  />

                  <EvaluationCard
                    title="Image Retrieval"
                    icon={Search}
                    values={[
                      [
                        "Recall@1",
                        retrieval?.recall_at_1,
                      ],
                      [
                        "Recall@5",
                        retrieval?.recall_at_5,
                      ],
                      [
                        "Recall@10",
                        retrieval?.recall_at_10,
                      ],
                    ]}
                    percentage
                  />

                </div>

              </div>

            </Card>

            {/* SPECIFICATION */}
            <Card className="h-fit p-6">

              <div className="flex items-center gap-3">

                <Cpu
                  size={18}
                  className="text-[#37d5c3]"
                />

                <div>
                  <h2 className="text-sm font-medium text-white">
                    Model Specification
                  </h2>

                  <p className="text-xs text-[#60788b]">
                    Registered model metadata
                  </p>
                </div>

              </div>

              <div className="mt-6 space-y-4">

                <SpecRow
                  label="Model ID"
                  value={selected.model_id}
                />

                <SpecRow
                  label="Architecture"
                  value={selected.name}
                />

                <SpecRow
                  label="Modality"
                  value={selected.modality}
                />

                <SpecRow
                  label="Embedding Dimension"
                  value={`${selected.embedding_dimension}`}
                />

                <SpecRow
                  label="Input Resolution"
                  value={`${selected.input_size} × ${selected.input_size}`}
                />

                <SpecRow
                  label="Availability"
                  value={selected.status}
                />

                <SpecRow
                  label="Dataset"
                  value={
                    classification?.dataset ??
                    "—"
                  }
                />

                <SpecRow
                  label="Evaluation Samples"
                  value={
                    classification
                      ? `${classification.samples}`
                      : "—"
                  }
                />

              </div>

            </Card>

          </section>
        )}

        {/* LEADERBOARD */}
        <section className="mt-6">

          <Card className="overflow-hidden">

            <div className="border-b border-[#1d3040] px-6 py-5">

              <div className="flex items-center gap-3">

                <Gauge
                  size={18}
                  className="text-[#37d5c3]"
                />

                <div>
                  <h2 className="text-sm font-medium text-white">
                    Foundation Model Efficiency
                  </h2>

                  <p className="mt-1 text-xs text-[#60788b]">
                    Ranked by measured tile throughput.
                  </p>
                </div>

              </div>

            </div>

            <div className="divide-y divide-[#1d3040]">

              {efficiencyRanking.map(
                ({ model, benchmark }, index) => {

                  if (!benchmark) {
                    return null;
                  }

                  return (
                    <div
                      key={`${model.model_id}-${benchmark.task}`}
                      className="grid grid-cols-[50px_minmax(180px,1fr)_140px_140px_140px] items-center gap-4 px-6 py-4"
                    >

                      <div className="text-sm font-semibold text-[#37d5c3]">
                        #{index + 1}
                      </div>

                      <div>
                        <p className="text-sm font-medium text-white">
                          {model.name}
                        </p>

                        <p className="text-xs text-[#60788b]">
                          {model.modality}
                        </p>
                      </div>

                      <div>
                        <p className="text-xs text-[#60788b]">
                          Latency
                        </p>

                        <p className="mt-1 text-sm text-white">
                          {formatMetric(
                            benchmark.latency_ms,
                            3
                          )} ms
                        </p>
                      </div>

                      <div>
                        <p className="text-xs text-[#60788b]">
                          Throughput
                        </p>

                        <p className="mt-1 text-sm text-white">
                          {formatMetric(
                            benchmark.throughput,
                            1
                          )} /s
                        </p>
                      </div>

                      <div>
                        <p className="text-xs text-[#60788b]">
                          GPU Memory
                        </p>

                        <p className="mt-1 text-sm text-white">
                          {formatMetric(
                            benchmark.gpu_memory_mb,
                            0
                          )} MB
                        </p>
                      </div>

                    </div>
                  );
                }
              )}

            </div>

          </Card>

        </section>

      </div>
    </AppShell>
  );
}

function MiniMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-[#1d3040] bg-[#09131d] p-3">

      <p className="text-[9px] uppercase tracking-wider text-[#60788b]">
        {label}
      </p>

      <p className="mt-1 text-xs font-medium text-white">
        {value}
      </p>

    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
}: {
  icon: any;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-[#1d3040] bg-[#09131d] p-4">

      <div className="flex items-center justify-between">

        <p className="text-xs text-[#60788b]">
          {label}
        </p>

        <Icon
          size={15}
          className="text-[#37d5c3]"
        />

      </div>

      <p className="mt-3 text-lg font-semibold text-white">
        {value}
      </p>

    </div>
  );
}

function SpecRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-[#1d3040] pb-3">

      <span className="text-xs text-[#60788b]">
        {label}
      </span>

      <span className="text-right text-xs font-medium text-[#d9e3ea]">
        {value}
      </span>

    </div>
  );
}

function EvaluationCard({
  title,
  icon: Icon,
  values,
  percentage = false,
}: {
  title: string;
  icon: any;
  values: [
    string,
    number | null | undefined
  ][];
  percentage?: boolean;
}) {
  return (
    <div className="rounded-lg border border-[#1d3040] bg-[#09131d] p-5">

      <div className="flex items-center gap-2">

        <Icon
          size={16}
          className="text-[#37d5c3]"
        />

        <h4 className="text-sm font-medium text-white">
          {title}
        </h4>

      </div>

      <div className="mt-5 space-y-3">

        {values.map(([label, value]) => (
          <div
            key={label}
            className="flex items-center justify-between"
          >

            <span className="text-xs text-[#60788b]">
              {label}
            </span>

            <span className="text-sm font-medium text-white">
              {value === null ||
              value === undefined
                ? "—"
                : percentage
                ? `${(
                    value * 100
                  ).toFixed(1)}%`
                : value.toFixed(4)}
            </span>

          </div>
        ))}

      </div>

    </div>
  );
}