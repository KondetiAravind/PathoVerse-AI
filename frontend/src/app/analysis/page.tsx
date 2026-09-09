"use client";

import { useEffect, useMemo, useState } from "react";

import {
  getHeatmapUrl,
  getModels,
  getSlides,
  getTileImageUrl,
  runClassification,
  runMIL,
} from "@/lib/analysis-api";

import type {
  ClassificationResponse,
  MILResponse,
  MILTopTile,
  ModelInfo,
  SlideInfo,
} from "@/types/analysis";


function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return `${(value * 100).toFixed(2)}%`;
}


function formatNumber(
  value: number | null | undefined,
  digits = 2
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return value.toFixed(digits);
}


function MetricCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <div className="text-[10px] uppercase tracking-[0.16em] text-slate-500">
        {label}
      </div>

      <div className="mt-2 text-2xl font-semibold text-white">
        {value}
      </div>

      {description && (
        <div className="mt-1 text-xs text-slate-500">
          {description}
        </div>
      )}
    </div>
  );
}


function SectionHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="mb-5">
      <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400">
        {eyebrow}
      </div>

      <h2 className="mt-1 text-xl font-semibold text-white">
        {title}
      </h2>

      <p className="mt-1 text-sm text-slate-400">
        {description}
      </p>
    </div>
  );
}


function SelectField({
  label,
  value,
  onChange,
  children,
  disabled = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className="block">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
        {label}
      </div>

      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-200 outline-none transition focus:border-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {children}
      </select>
    </label>
  );
}


function StatusMessage({
  type,
  children,
}: {
  type: "error" | "success" | "info";
  children: React.ReactNode;
}) {
  const classes = {
    error: "border-red-500/30 bg-red-500/10 text-red-300",
    success: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    info: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  };

  return (
    <div
      className={`mt-4 rounded-lg border px-4 py-3 text-sm ${classes[type]}`}
    >
      {children}
    </div>
  );
}


export default function AnalysisPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [slides, setSlides] = useState<SlideInfo[]>([]);

  const [selectedClassificationModel, setSelectedClassificationModel] =
    useState("gigapath-flash");

  const [selectedSlide, setSelectedSlide] = useState(
    "CMU-1-Small-Region"
  );

  const [selectedMILModel, setSelectedMILModel] =
    useState("gigapath-flash");

  const [classificationResult, setClassificationResult] =
    useState<ClassificationResponse | null>(null);

  const [milResult, setMilResult] =
    useState<MILResponse | null>(null);

  const [loadingInitial, setLoadingInitial] = useState(true);
  const [classificationLoading, setClassificationLoading] =
    useState(false);
  const [milLoading, setMilLoading] = useState(false);

  const [classificationError, setClassificationError] =
    useState<string | null>(null);

  const [milError, setMilError] =
    useState<string | null>(null);

  const [showHeatmap, setShowHeatmap] = useState(false);

  useEffect(() => {
    async function loadInitialData() {
      try {
        setLoadingInitial(true);

        const [modelData, slideData] = await Promise.all([
          getModels(),
          getSlides(),
        ]);

        setModels(modelData);
        setSlides(slideData);

        const availableModel = modelData.find(
          (model) => model.model_id === "gigapath-flash"
        );

        if (availableModel) {
          setSelectedClassificationModel(
            availableModel.model_id
          );

          setSelectedMILModel(availableModel.model_id);
        }

        if (slideData.length > 0) {
          setSelectedSlide(slideData[0].slide_id);
        }
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "Unable to connect to the PathoVerse API.";

        setClassificationError(message);
        setMilError(message);
      } finally {
        setLoadingInitial(false);
      }
    }

    loadInitialData();
  }, []);


  const availableModels = useMemo(
    () => models.filter((model) => model.status === "available"),
    [models]
  );


  async function handleClassification() {
    try {
      setClassificationLoading(true);
      setClassificationError(null);

      const result = await runClassification(
        selectedClassificationModel,
        "patchcamelyon"
      );

      setClassificationResult(result);
    } catch (error) {
      setClassificationError(
        error instanceof Error
          ? error.message
          : "Classification failed."
      );
    } finally {
      setClassificationLoading(false);
    }
  }


  async function handleMIL() {
    try {
      setMilLoading(true);
      setMilError(null);
      setShowHeatmap(false);

      const result = await runMIL(
        selectedSlide,
        selectedMILModel
      );

      setMilResult(result);
    } catch (error) {
      setMilError(
        error instanceof Error
          ? error.message
          : "MIL analysis failed."
      );
    } finally {
      setMilLoading(false);
    }
  }


  const topTiles: MILTopTile[] =
    milResult?.attention?.top_tiles ?? [];


  if (loadingInitial) {
    return (
      <main className="px-8 py-10">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-16 text-center">
          <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />

          <div className="mt-5 text-sm text-slate-300">
            Loading AI analysis workspace...
          </div>

          <div className="mt-1 text-xs text-slate-500">
            Connecting to PathoVerse analysis services
          </div>
        </div>
      </main>
    );
  }


  return (
    <main className="px-8 py-8">
      {/* PAGE HEADER */}
      <section className="mb-8">
        <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-cyan-400">
              Artificial Intelligence
            </div>

            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">
              AI Analysis Workbench
            </h1>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Execute foundation-model pathology analysis,
              whole-slide multiple-instance learning and
              attention-based explainability directly through
              the PathoVerse inference API.
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3 py-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />

            <span className="text-xs text-emerald-300">
              Analysis API Ready
            </span>
          </div>
        </div>
      </section>


      {/* CLASSIFICATION */}
      <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
        <SectionHeader
          eyebrow="01 · Classification"
          title="PatchCamelyon Evaluation"
          description="Run the registered foundation encoder against the PathoVerse prototype classification evaluation."
        />

        <div className="grid gap-4 md:grid-cols-[1fr_1fr_auto]">
          <SelectField
            label="Foundation Model"
            value={selectedClassificationModel}
            onChange={setSelectedClassificationModel}
          >
            {availableModels.map((model) => (
              <option
                key={`classification-model-${model.model_id}`}
                value={model.model_id}
              >
                {model.name}
              </option>
            ))}
          </SelectField>

          <div>
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
              Dataset
            </div>

            <div className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5">
              <div className="text-sm text-slate-200">
                PatchCamelyon
              </div>

              <div className="mt-0.5 text-xs text-slate-500">
                Prototype test evaluation
              </div>
            </div>
          </div>

          <div className="flex items-end">
            <button
              type="button"
              onClick={handleClassification}
              disabled={classificationLoading}
              className="w-full whitespace-nowrap rounded-lg border border-cyan-500/50 bg-cyan-500/10 px-5 py-2.5 text-sm font-medium text-cyan-300 transition hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-50 md:w-auto"
            >
              {classificationLoading
                ? "Running..."
                : "Run Classification"}
            </button>
          </div>
        </div>

        {classificationError && (
          <StatusMessage type="error">
            {classificationError}
          </StatusMessage>
        )}

        {classificationResult && (
          <div className="mt-6">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-white">
                  Evaluation Results
                </div>

                <div className="mt-1 text-xs text-slate-500">
                  {classificationResult.model} ·{" "}
                  {classificationResult.dataset}
                </div>
              </div>

              <div className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-[10px] uppercase tracking-wider text-emerald-300">
                Completed
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
              <MetricCard
                label="Accuracy"
                value={formatPercent(
                  classificationResult.accuracy
                )}
              />

              <MetricCard
                label="AUROC"
                value={formatPercent(
                  classificationResult.auroc
                )}
              />

              <MetricCard
                label="F1"
                value={formatPercent(
                  classificationResult.f1
                )}
              />

              <MetricCard
                label="Precision"
                value={formatPercent(
                  classificationResult.precision
                )}
              />

              <MetricCard
                label="Recall"
                value={formatPercent(
                  classificationResult.recall
                )}
              />

              <MetricCard
                label="Sensitivity"
                value={formatPercent(
                  classificationResult.sensitivity
                )}
              />

              <MetricCard
                label="Specificity"
                value={formatPercent(
                  classificationResult.specificity
                )}
              />
            </div>
          </div>
        )}
      </section>


      {/* MIL */}
      <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
        <SectionHeader
          eyebrow="02 · Whole-Slide Intelligence"
          title="Attention-based MIL Analysis"
          description="Aggregate tile-level foundation embeddings into a whole-slide representation using gated attention MIL."
        />

        <div className="grid gap-4 md:grid-cols-[1fr_1fr_auto]">
          <SelectField
            label="Whole-Slide Study"
            value={selectedSlide}
            onChange={setSelectedSlide}
            disabled={slides.length === 0}
          >
            {slides.map((slide) => (
              <option
                key={`slide-${slide.slide_id}`}
                value={slide.slide_id}
              >
                {slide.slide_id}
              </option>
            ))}
          </SelectField>

          <SelectField
            label="Foundation Model"
            value={selectedMILModel}
            onChange={setSelectedMILModel}
          >
            {availableModels.map((model) => (
              <option
                key={`mil-model-${model.model_id}`}
                value={model.model_id}
              >
                {model.name}
              </option>
            ))}
          </SelectField>

          <div className="flex items-end">
            <button
              type="button"
              onClick={handleMIL}
              disabled={milLoading || slides.length === 0}
              className="w-full whitespace-nowrap rounded-lg border border-cyan-500/50 bg-cyan-500/10 px-5 py-2.5 text-sm font-medium text-cyan-300 transition hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-50 md:w-auto"
            >
              {milLoading
                ? "Running..."
                : "Run MIL Analysis"}
            </button>
          </div>
        </div>

        {milError && (
          <StatusMessage type="error">
            {milError}
          </StatusMessage>
        )}

        {milResult && (
          <div className="mt-6">
            <div className="mb-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
              <div>
                <div className="text-sm font-semibold text-white">
                  Whole-Slide Result
                </div>

                <div className="mt-1 text-xs text-slate-500">
                  {milResult.foundation_model.name} ·{" "}
                  {milResult.mil.architecture}
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                {milResult.prototype && (
                  <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-[10px] uppercase tracking-wider text-amber-300">
                    Prototype
                  </span>
                )}

                {!milResult.trained && (
                  <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-[10px] uppercase tracking-wider text-amber-300">
                    Untrained MIL
                  </span>
                )}
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard
                label="Prediction"
                value={`Class ${milResult.prediction.class_id}`}
                description="MIL predicted class"
              />

              <MetricCard
                label="Probability"
                value={formatPercent(
                  milResult.prediction.probability
                )}
                description="Predicted class probability"
              />

              <MetricCard
                label="Tiles Analyzed"
                value={String(
                  milResult.attention.tile_count
                )}
                description="Tissue-aware WSI tiles"
              />

              <MetricCard
                label="Attention Sum"
                value={formatNumber(
                  milResult.attention.sum,
                  4
                )}
                description="Normalized attention"
              />
            </div>


            {/* MIL TECHNICAL DETAILS */}
            <div className="mt-5 grid gap-4 lg:grid-cols-3">
              <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <div className="text-[10px] uppercase tracking-[0.16em] text-slate-500">
                  Foundation Encoder
                </div>

                <div className="mt-2 text-sm font-medium text-white">
                  {milResult.foundation_model.name}
                </div>

                <div className="mt-1 text-xs text-slate-500">
                  {milResult.foundation_model.embedding_dimension}-D tile embeddings
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <div className="text-[10px] uppercase tracking-[0.16em] text-slate-500">
                  MIL Architecture
                </div>

                <div className="mt-2 text-sm font-medium text-white">
                  {milResult.mil.architecture}
                </div>

                <div className="mt-1 text-xs text-slate-500">
                  Hidden {milResult.mil.hidden_dimension} ·
                  Attention {milResult.mil.attention_dimension}
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <div className="text-[10px] uppercase tracking-[0.16em] text-slate-500">
                  Slide Representation
                </div>

                <div className="mt-2 text-sm font-medium text-white">
                  {milResult.slide_embedding_dimension}-D
                </div>

                <div className="mt-1 text-xs text-slate-500">
                  Projected whole-slide embedding
                </div>
              </div>
            </div>
          </div>
        )}
      </section>


      {/* ATTENTION REGIONS */}
      {milResult && topTiles.length > 0 && (
        <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
          <SectionHeader
            eyebrow="03 · Attention"
            title="Top Tissue Regions"
            description="Highest-attention tiles identified by the gated MIL aggregation module."
          />

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
            {topTiles.map((tile) => (
              <div
                key={`attention-tile-${tile.tile_id}`}
                className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950/60"
              >
                <div className="relative aspect-square overflow-hidden bg-slate-900">
                  <img
                    src={getTileImageUrl(
                      milResult.slide_id,
                      tile.tile_id
                    )}
                    alt={`Attention tile ${tile.tile_id}`}
                    className="h-full w-full object-cover"
                    loading="lazy"
                  />

                  <div className="absolute left-2 top-2 rounded-md border border-slate-700 bg-slate-950/90 px-2 py-1 text-[10px] font-semibold text-white">
                    #{tile.rank}
                  </div>
                </div>

                <div className="p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-white">
                      Tile {tile.tile_id}
                    </span>

                    <span className="text-xs font-medium text-cyan-300">
                      {formatNumber(
                        tile.attention,
                        4
                      )}
                    </span>
                  </div>

                  <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-cyan-400"
                      style={{
                        width: `${Math.min(
                          100,
                          (tile.attention /
                            Math.max(
                              topTiles[0]?.attention || 1,
                              0.000001
                            )) *
                            100
                        )}%`,
                      }}
                    />
                  </div>

                  <div className="mt-2 flex justify-between text-[10px] text-slate-500">
                    <span>
                      Tissue{" "}
                      {formatPercent(
                        tile.tissue_ratio
                      )}
                    </span>

                    <span>
                      ({tile.x}, {tile.y})
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}


      {/* EXPLAINABILITY */}
      {milResult && (
        <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
          <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
            <SectionHeader
              eyebrow="04 · Explainability"
              title="Attention Heatmap"
              description="Visualize the spatial distribution of MIL attention across the whole-slide image."
            />

            <button
              type="button"
              onClick={() =>
                setShowHeatmap((current) => !current)
              }
              className="rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-4 py-2 text-sm font-medium text-cyan-300 transition hover:bg-cyan-500/20"
            >
              {showHeatmap
                ? "Hide Heatmap"
                : "View Attention Heatmap"}
            </button>
          </div>

          {showHeatmap && (
            <div className="mt-2 overflow-hidden rounded-xl border border-slate-800 bg-black">
              <div className="border-b border-slate-800 bg-slate-950/80 px-4 py-3">
                <div className="text-sm font-medium text-white">
                  {milResult.slide_id}
                </div>

                <div className="mt-1 text-xs text-slate-500">
                  {milResult.model} · attention-weighted WSI visualization
                </div>
              </div>

              <div className="flex min-h-[420px] items-center justify-center p-4">
                <img
                  src={getHeatmapUrl(
                    milResult.slide_id,
                    milResult.model
                  )}
                  alt={`${milResult.slide_id} attention heatmap`}
                  className="max-h-[760px] w-auto max-w-full rounded-lg object-contain"
                />
              </div>

              <div className="border-t border-slate-800 bg-slate-950/80 px-4 py-3 text-xs text-amber-300">
                Prototype explainability output. Attention regions
                indicate model focus and should not be interpreted as
                clinical diagnostic evidence.
              </div>
            </div>
          )}
        </section>
      )}


      {/* EMPTY STATE */}
      {!classificationResult &&
        !milResult &&
        !classificationLoading &&
        !milLoading && (
          <section className="mt-6 rounded-2xl border border-dashed border-slate-800 bg-slate-950/30 p-10 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-500/20 bg-cyan-500/10 text-cyan-300">
              AI
            </div>

            <h3 className="mt-4 text-sm font-semibold text-white">
              Ready for analysis
            </h3>

            <p className="mx-auto mt-2 max-w-lg text-xs leading-5 text-slate-500">
              Select a foundation model or whole-slide study above
              and execute an analysis to populate real evaluation,
              MIL and explainability results.
            </p>
          </section>
        )}
    </main>
  );
}