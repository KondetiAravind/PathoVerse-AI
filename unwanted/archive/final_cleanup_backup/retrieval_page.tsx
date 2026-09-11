"use client";

import { apiFetch } from "@/lib/api";

import { useEffect, useMemo, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

const fetch = apiFetch;

type Model = {
  name: string;
  model_id: string;
  embedding_dimension: number;
  input_size: number;
  modality: string;
  description: string;
  status: string;
};

type Slide = {
  slide_id: string;
  filename: string;
  width: number;
  height: number;
  levels: number;
  mpp_x: number;
  mpp_y: number;
  tissue_ratio: number | null;
  tile_count: number | null;
};

type Tile = {
  tile_id: number;
  slide_id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  level: number;
  tissue_ratio: number;
};

type RetrievalResult = {
  tile_id: number;
  slide_id: string;
  score: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

type RetrievalResponse = {
  query_tile_id: number;
  model: string;
  top_k: number;
  results: RetrievalResult[];
};

type BenchmarkRecord = {
  model: string;
  model_id: string;
  task: string;
  dataset: string | null;
  split: string | null;
  samples: number;
  embedding_dimension: number;
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

function apiUrl(path: string) {
  return `${API_BASE}${path}`;
}

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(2)}%`;
}

function formatScore(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function tileImageUrl(slideId: string, tileId: number) {
  return apiUrl(
    `/api/slides/${encodeURIComponent(slideId)}/tiles/${tileId}/image`
  );
}

function modelLabel(modelId: string, models: Model[]) {
  return models.find((m) => m.model_id === modelId)?.name || modelId;
}

export default function RetrievalPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [slides, setSlides] = useState<Slide[]>([]);
  const [tiles, setTiles] = useState<Tile[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkRecord[]>([]);

  const [selectedSlide, setSelectedSlide] = useState("");
  const [selectedModel, setSelectedModel] = useState("gigapath-flash");
  const [selectedTile, setSelectedTile] = useState(0);
  const [topK, setTopK] = useState(5);

  const [results, setResults] = useState<RetrievalResponse | null>(null);

  const [loadingWorkspace, setLoadingWorkspace] = useState(true);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");

  const selectedTileData = useMemo(
    () => tiles.find((tile) => tile.tile_id === selectedTile),
    [tiles, selectedTile]
  );

  const selectedModelData = useMemo(
    () => models.find((model) => model.model_id === selectedModel),
    [models, selectedModel]
  );

  const retrievalBenchmark = useMemo(
    () =>
      benchmarks.find(
        (record) =>
          record.task === "image_retrieval" &&
          record.model_id === selectedModel
      ),
    [benchmarks, selectedModel]
  );

  const availableModels = useMemo(
    () => models.filter((model) => model.status === "available"),
    [models]
  );

  async function loadWorkspace() {
    try {
      setLoadingWorkspace(true);
      setError("");

      const [modelsResponse, slidesResponse, benchmarksResponse] =
        await Promise.all([
          fetch(apiUrl("/api/models")),
          fetch(apiUrl("/api/slides")),
          fetch(apiUrl("/api/benchmarks")),
        ]);

      if (!modelsResponse.ok) {
        throw new Error("Failed to load foundation models.");
      }

      if (!slidesResponse.ok) {
        throw new Error("Failed to load whole-slide studies.");
      }

      if (!benchmarksResponse.ok) {
        throw new Error("Failed to load benchmark records.");
      }

      const modelsData: Model[] = await modelsResponse.json();
      const slidesData: Slide[] = await slidesResponse.json();
      const benchmarkData: BenchmarkResponse =
        await benchmarksResponse.json();

      setModels(modelsData);
      setSlides(slidesData);
      setBenchmarks(benchmarkData.records);

      if (slidesData.length > 0) {
        const firstSlide = slidesData[0];
        setSelectedSlide(firstSlide.slide_id);

        const tileResponse = await fetch(
          apiUrl(
            `/api/slides/${encodeURIComponent(
              firstSlide.slide_id
            )}/tiles`
          )
        );

        if (!tileResponse.ok) {
          throw new Error("Failed to load pathology tiles.");
        }

        const tileData: { total_tiles: number; tiles: Tile[] } =
          await tileResponse.json();

        setTiles(tileData.tiles);

        if (tileData.tiles.length > 0) {
          setSelectedTile(tileData.tiles[0].tile_id);
        }
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load retrieval workspace."
      );
    } finally {
      setLoadingWorkspace(false);
    }
  }

  async function loadTiles(slideId: string) {
    try {
      setError("");

      const response = await fetch(
        apiUrl(`/api/slides/${encodeURIComponent(slideId)}/tiles`)
      );

      if (!response.ok) {
        throw new Error("Failed to load tiles for selected slide.");
      }

      const data: { total_tiles: number; tiles: Tile[] } =
        await response.json();

      setTiles(data.tiles);

      if (data.tiles.length > 0) {
        setSelectedTile(data.tiles[0].tile_id);
      } else {
        setSelectedTile(0);
      }

      setResults(null);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load slide tiles."
      );
    }
  }

  async function runRetrieval() {
    if (!selectedSlide || selectedTile === undefined) {
      setError("Select a slide and query tile first.");
      return;
    }

    try {
      setSearching(true);
      setError("");
      setResults(null);

      const response = await fetch(apiUrl("/api/retrieval/search"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          slide_id: selectedSlide,
          tile_id: selectedTile,
          model: selectedModel,
          top_k: topK,
        }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(
          detail || "Retrieval request failed."
        );
      }

      const data: RetrievalResponse = await response.json();
      setResults(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to execute retrieval."
      );
    } finally {
      setSearching(false);
    }
  }

  useEffect(() => {
    loadWorkspace();
  }, []);

  if (loadingWorkspace) {
    return (
      <main className="min-h-screen px-6 py-8 lg:px-10">
        <div className="mx-auto max-w-7xl">
          <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-12 text-center">
            <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />
            <p className="text-sm text-slate-300">
              Loading retrieval workspace...
            </p>
          </div>
        </div>
      </main>
    );
  }

  const selectedSlideData = slides.find(
    (slide) => slide.slide_id === selectedSlide
  );

  return (
    <main className="min-h-screen px-6 py-8 lg:px-10">
      <div className="mx-auto max-w-7xl space-y-6">

        {/* ===================================================== */}
        {/* HEADER */}
        {/* ===================================================== */}

        <section className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.25em] text-cyan-400">
              Multimodal Intelligence
            </p>

            <h1 className="text-3xl font-semibold tracking-tight text-white">
              Pathology Retrieval
            </h1>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Search visually similar tissue regions using foundation-model
              embeddings and vector similarity retrieval.
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3 py-2 text-xs text-emerald-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            Retrieval API Ready
          </div>
        </section>

        {/* ===================================================== */}
        {/* ERROR */}
        {/* ===================================================== */}

        {error && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* ===================================================== */}
        {/* CONFIGURATION */}
        {/* ===================================================== */}

        <section className="rounded-2xl border border-slate-800 bg-slate-950/60 p-5">
          <div className="mb-5">
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400">
              01 · Retrieval Configuration
            </p>

            <h2 className="mt-1 text-lg font-semibold text-white">
              Image-to-Image Search
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              Select a whole-slide study, query tissue tile, foundation model
              and retrieval depth.
            </p>
          </div>

          <div className="grid gap-4 lg:grid-cols-[1.3fr_1.3fr_1fr_0.7fr_auto]">

            {/* Slide */}
            <div>
              <label className="mb-2 block text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                Whole-Slide Study
              </label>

              <select
                value={selectedSlide}
                onChange={(event) => {
                  const slideId = event.target.value;
                  setSelectedSlide(slideId);
                  loadTiles(slideId);
                }}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white outline-none transition focus:border-cyan-400"
              >
                {slides.map((slide) => (
                  <option key={slide.slide_id} value={slide.slide_id}>
                    {slide.slide_id}
                  </option>
                ))}
              </select>
            </div>

            {/* Model */}
            <div>
              <label className="mb-2 block text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                Foundation Model
              </label>

              <select
                value={selectedModel}
                onChange={(event) => {
                  setSelectedModel(event.target.value);
                  setResults(null);
                }}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white outline-none transition focus:border-cyan-400"
              >
                {availableModels.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Tile */}
            <div>
              <label className="mb-2 block text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                Query Tile
              </label>

              <select
                value={selectedTile}
                onChange={(event) => {
                  setSelectedTile(Number(event.target.value));
                  setResults(null);
                }}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white outline-none transition focus:border-cyan-400"
              >
                {tiles.map((tile) => (
                  <option key={tile.tile_id} value={tile.tile_id}>
                    Tile {tile.tile_id} ·{" "}
                    {(tile.tissue_ratio * 100).toFixed(0)}% tissue
                  </option>
                ))}
              </select>
            </div>

            {/* Top K */}
            <div>
              <label className="mb-2 block text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                Top K
              </label>

              <select
                value={topK}
                onChange={(event) => {
                  setTopK(Number(event.target.value));
                  setResults(null);
                }}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-white outline-none transition focus:border-cyan-400"
              >
                {[5, 10, 15, 20].map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>

            {/* Button */}
            <div className="flex items-end">
              <button
                type="button"
                onClick={runRetrieval}
                disabled={searching}
                className="w-full whitespace-nowrap rounded-lg border border-cyan-500/60 bg-cyan-500/10 px-5 py-3 text-sm font-medium text-cyan-300 transition hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {searching ? "Searching..." : "Search Similar Tissue"}
              </button>
            </div>
          </div>
        </section>

        {/* ===================================================== */}
        {/* QUERY + MODEL INFORMATION */}
        {/* ===================================================== */}

        <section className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">

          {/* Query Tile */}
          <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-5">
            <div className="mb-4">
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400">
                02 · Query Region
              </p>

              <h2 className="mt-1 text-lg font-semibold text-white">
                Query Tile
              </h2>

              <p className="mt-1 text-xs text-slate-500">
                Tissue region used as the retrieval query.
              </p>
            </div>

            {selectedTileData ? (
              <div className="overflow-hidden rounded-xl border border-slate-800 bg-black">
                <img
                  src={tileImageUrl(
                    selectedTileData.slide_id,
                    selectedTileData.tile_id
                  )}
                  alt={`Query pathology tile ${selectedTileData.tile_id}`}
                  className="aspect-square w-full object-cover"
                />

                <div className="grid grid-cols-2 gap-px border-t border-slate-800 bg-slate-800">
                  <div className="bg-slate-950 p-3">
                    <p className="text-[9px] uppercase tracking-wider text-slate-500">
                      Tile ID
                    </p>
                    <p className="mt-1 text-sm font-semibold text-white">
                      #{selectedTileData.tile_id}
                    </p>
                  </div>

                  <div className="bg-slate-950 p-3">
                    <p className="text-[9px] uppercase tracking-wider text-slate-500">
                      Tissue
                    </p>
                    <p className="mt-1 text-sm font-semibold text-white">
                      {(
                        selectedTileData.tissue_ratio * 100
                      ).toFixed(1)}
                      %
                    </p>
                  </div>

                  <div className="bg-slate-950 p-3">
                    <p className="text-[9px] uppercase tracking-wider text-slate-500">
                      Coordinates
                    </p>
                    <p className="mt-1 text-xs text-slate-300">
                      ({selectedTileData.x},{" "}
                      {selectedTileData.y})
                    </p>
                  </div>

                  <div className="bg-slate-950 p-3">
                    <p className="text-[9px] uppercase tracking-wider text-slate-500">
                      Resolution
                    </p>
                    <p className="mt-1 text-xs text-slate-300">
                      {selectedTileData.width} ×{" "}
                      {selectedTileData.height}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex aspect-square items-center justify-center rounded-xl border border-dashed border-slate-800 text-sm text-slate-500">
                No query tile available.
              </div>
            )}
          </div>

          {/* Model + benchmark */}
          <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-5">
            <div className="mb-5">
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400">
                Retrieval Engine
              </p>

              <h2 className="mt-1 text-lg font-semibold text-white">
                {selectedModelData?.name || selectedModel}
              </h2>

              <p className="mt-1 text-xs text-slate-500">
                {selectedModelData?.description ||
                  "Foundation-model embedding retrieval."}
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
                <p className="text-[9px] uppercase tracking-wider text-slate-500">
                  Embedding
                </p>

                <p className="mt-2 text-xl font-semibold text-white">
                  {selectedModelData?.embedding_dimension || "—"}-D
                </p>

                <p className="mt-1 text-[10px] text-slate-500">
                  Feature representation
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
                <p className="text-[9px] uppercase tracking-wider text-slate-500">
                  Input
                </p>

                <p className="mt-2 text-xl font-semibold text-white">
                  {selectedModelData?.input_size || "—"}px
                </p>

                <p className="mt-1 text-[10px] text-slate-500">
                  Model input resolution
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
                <p className="text-[9px] uppercase tracking-wider text-slate-500">
                  R@1
                </p>

                <p className="mt-2 text-xl font-semibold text-white">
                  {formatPercent(
                    retrievalBenchmark?.recall_at_1
                  )}
                </p>

                <p className="mt-1 text-[10px] text-slate-500">
                  Retrieval benchmark
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
                <p className="text-[9px] uppercase tracking-wider text-slate-500">
                  R@5
                </p>

                <p className="mt-2 text-xl font-semibold text-white">
                  {formatPercent(
                    retrievalBenchmark?.recall_at_5
                  )}
                </p>

                <p className="mt-1 text-[10px] text-slate-500">
                  Retrieval benchmark
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 sm:col-span-2">
                <p className="text-[9px] uppercase tracking-wider text-slate-500">
                  R@10
                </p>

                <p className="mt-2 text-2xl font-semibold text-white">
                  {formatPercent(
                    retrievalBenchmark?.recall_at_10
                  )}
                </p>

                <p className="mt-1 text-[10px] text-slate-500">
                  Fraction of relevant regions retrieved in top 10
                  results
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ===================================================== */}
        {/* RESULTS */}
        {/* ===================================================== */}

        <section className="rounded-2xl border border-slate-800 bg-slate-950/60 p-5">

          <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400">
                03 · Similar Tissue Regions
              </p>

              <h2 className="mt-1 text-lg font-semibold text-white">
                Retrieval Results
              </h2>

              <p className="mt-1 text-xs text-slate-500">
                Top-{topK} regions ranked by foundation-model embedding
                similarity.
              </p>
            </div>

            {results && (
              <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3 py-2 text-xs text-emerald-300">
                {results.results.length} regions retrieved
              </div>
            )}
          </div>

          {!results && !searching && (
            <div className="flex min-h-[260px] items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40">
              <div className="max-w-md text-center">
                <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-500/20 bg-cyan-500/5 text-cyan-300">
                  <span className="text-lg">⌕</span>
                </div>

                <h3 className="text-sm font-semibold text-white">
                  Ready for retrieval
                </h3>

                <p className="mt-2 text-xs leading-5 text-slate-500">
                  Select a query tile and foundation model, then execute
                  the search to retrieve visually similar pathology
                  regions.
                </p>
              </div>
            </div>
          )}

          {searching && (
            <div className="flex min-h-[260px] items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40">
              <div className="text-center">
                <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />

                <p className="text-sm text-slate-300">
                  Searching tissue embeddings...
                </p>

                <p className="mt-1 text-xs text-slate-600">
                  Querying {selectedModelData?.name || selectedModel}
                </p>
              </div>
            </div>
          )}

          {results && !searching && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
              {results.results.map((result, index) => {
                const isQuery =
                  result.tile_id === results.query_tile_id;

                const tile = tiles.find(
                  (item) => item.tile_id === result.tile_id
                );

                return (
                  <article
                    key={`${result.slide_id}-${result.tile_id}-${index}`}
                    className={`overflow-hidden rounded-xl border ${
                      isQuery
                        ? "border-cyan-400/70"
                        : "border-slate-800"
                    } bg-slate-950 transition hover:border-slate-600`}
                  >
                    <div className="relative aspect-square overflow-hidden bg-black">
                      <img
                        src={tileImageUrl(
                          result.slide_id,
                          result.tile_id
                        )}
                        alt={`Retrieved pathology tile ${result.tile_id}`}
                        className="h-full w-full object-cover"
                      />

                      <div className="absolute left-2 top-2 rounded-md border border-slate-700 bg-slate-950/90 px-2 py-1 text-[10px] font-semibold text-white">
                        #{index + 1}
                      </div>

                      {isQuery && (
                        <div className="absolute right-2 top-2 rounded-md border border-cyan-400/50 bg-cyan-950/90 px-2 py-1 text-[9px] font-semibold uppercase tracking-wider text-cyan-300">
                          Query
                        </div>
                      )}
                    </div>

                    <div className="space-y-3 p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="text-[9px] uppercase tracking-wider text-slate-500">
                            Tile ID
                          </p>

                          <p className="mt-1 text-sm font-semibold text-white">
                            #{result.tile_id}
                          </p>
                        </div>

                        <div className="text-right">
                          <p className="text-[9px] uppercase tracking-wider text-slate-500">
                            Similarity
                          </p>

                          <p className="mt-1 text-sm font-semibold text-cyan-300">
                            {formatScore(result.score)}
                          </p>
                        </div>
                      </div>

                      <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
                        <div
                          className="h-full rounded-full bg-cyan-400"
                          style={{
                            width: `${Math.max(
                              0,
                              Math.min(100, result.score * 100)
                            )}%`,
                          }}
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-[10px]">
                        <div>
                          <span className="text-slate-600">
                            Coordinates
                          </span>

                          <p className="mt-1 text-slate-300">
                            ({result.x}, {result.y})
                          </p>
                        </div>

                        <div className="text-right">
                          <span className="text-slate-600">
                            Tissue
                          </span>

                          <p className="mt-1 text-slate-300">
                            {tile
                              ? `${(
                                  tile.tissue_ratio * 100
                                ).toFixed(1)}%`
                              : "—"}
                          </p>
                        </div>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        {/* ===================================================== */}
        {/* BENCHMARK SUMMARY */}
        {/* ===================================================== */}

        <section className="rounded-2xl border border-slate-800 bg-slate-950/60 p-5">

          <div className="mb-5">
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400">
              04 · Retrieval Benchmark
            </p>

            <h2 className="mt-1 text-lg font-semibold text-white">
              Foundation Model Comparison
            </h2>

            <p className="mt-1 text-xs text-slate-500">
              Measured retrieval quality from the PathoVerse evaluation
              pipeline.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[700px] border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-left">
                  <th className="px-4 py-3 text-[9px] uppercase tracking-wider text-slate-500">
                    Model
                  </th>

                  <th className="px-4 py-3 text-[9px] uppercase tracking-wider text-slate-500">
                    Embedding
                  </th>

                  <th className="px-4 py-3 text-[9px] uppercase tracking-wider text-slate-500">
                    R@1
                  </th>

                  <th className="px-4 py-3 text-[9px] uppercase tracking-wider text-slate-500">
                    R@5
                  </th>

                  <th className="px-4 py-3 text-[9px] uppercase tracking-wider text-slate-500">
                    R@10
                  </th>
                </tr>
              </thead>

              <tbody>
                {availableModels.map((model) => {
                  const benchmark = benchmarks.find(
                    (record) =>
                      record.task === "image_retrieval" &&
                      record.model_id === model.model_id
                  );

                  const active =
                    model.model_id === selectedModel;

                  return (
                    <tr
                      key={model.model_id}
                      className={`border-b border-slate-900 ${
                        active
                          ? "bg-cyan-500/[0.03]"
                          : ""
                      }`}
                    >
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-2">
                          <span
                            className={`h-2 w-2 rounded-full ${
                              active
                                ? "bg-cyan-400"
                                : "bg-slate-700"
                            }`}
                          />

                          <span className="text-sm font-medium text-white">
                            {model.name}
                          </span>
                        </div>
                      </td>

                      <td className="px-4 py-4 text-sm text-slate-300">
                        {model.embedding_dimension}-D
                      </td>

                      <td className="px-4 py-4 text-sm text-slate-300">
                        {formatPercent(
                          benchmark?.recall_at_1
                        )}
                      </td>

                      <td className="px-4 py-4 text-sm font-semibold text-cyan-300">
                        {formatPercent(
                          benchmark?.recall_at_5
                        )}
                      </td>

                      <td className="px-4 py-4 text-sm text-slate-300">
                        {formatPercent(
                          benchmark?.recall_at_10
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* ===================================================== */}
        {/* SYSTEM INFORMATION */}
        {/* ===================================================== */}

        <section className="grid gap-4 md:grid-cols-3">

          <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
            <p className="text-[9px] uppercase tracking-wider text-slate-500">
              Study
            </p>

            <p className="mt-2 text-sm font-semibold text-white">
              {selectedSlideData?.slide_id || "—"}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              {selectedSlideData
                ? `${selectedSlideData.width} × ${selectedSlideData.height}px`
                : ""}
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
            <p className="text-[9px] uppercase tracking-wider text-slate-500">
              Query
            </p>

            <p className="mt-2 text-sm font-semibold text-white">
              Tile #{selectedTile}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              {selectedTileData
                ? `(${selectedTileData.x}, ${selectedTileData.y})`
                : ""}
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
            <p className="text-[9px] uppercase tracking-wider text-slate-500">
              Retrieval Engine
            </p>

            <p className="mt-2 text-sm font-semibold text-white">
              {selectedModelData?.name || selectedModel}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Embedding similarity search
            </p>
          </div>
        </section>

      </div>
    </main>
  );
}