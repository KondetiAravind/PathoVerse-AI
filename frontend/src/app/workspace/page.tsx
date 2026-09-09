"use client";

import {
  Activity,
  BrainCircuit,
  ChevronDown,
  Crosshair,
  Database,
  Eye,
  FileImage,
  Layers3,
  Microscope,
  Move,
  RefreshCw,
  Search,
  Sparkles,
  Target,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

import {
  TransformComponent,
  TransformWrapper,
} from "react-zoom-pan-pinch";

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";


// ============================================================
// CONFIGURATION
// ============================================================

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://localhost:8000";


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


type TileInfo = {
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


type MILTopTile = {
  rank: number;
  tile_index: number;
  tile_id: number;
  attention: number;
  x: number;
  y: number;
  width: number;
  height: number;
  level: number;
  tissue_ratio: number;
};


type MILResponse = {
  schema_version: string;
  task: string;
  slide_id: string;
  model: string;

  foundation_model: {
    name: string;
    embedding_dimension: number;
  };

  mil: {
    architecture: string;
    hidden_dimension: number;
    attention_dimension: number;
    num_classes: number;
    trained: boolean;
    prototype: boolean;
    seed?: number | null;
  };

  prediction: {
    class_id: number;
    probability: number;
  };

  attention: {
    tile_count: number;
    sum: number;
    top_k: number;
    top_tiles: MILTopTile[];
  };

  slide_embedding_dimension: number;
  slide_embedding_path: string;

  trained: boolean;
  prototype: boolean;
};


// ============================================================
// HELPERS
// ============================================================

function percent(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${(value * 100).toFixed(2)}%`;
}


function probability(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}


function score(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}


function normalizeTiles(payload: any): TileInfo[] {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (Array.isArray(payload?.tiles)) {
    return payload.tiles;
  }

  return [];
}


function normalizeSlides(payload: any): SlideInfo[] {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (Array.isArray(payload?.slides)) {
    return payload.slides;
  }

  if (payload && typeof payload === "object") {
    return [payload];
  }

  return [];
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
    <div className="flex items-center gap-2">
      <span className="text-[9px] font-semibold tracking-[0.22em] text-cyan-400">
        {number}
      </span>

      <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-slate-600">
        {children}
      </span>
    </div>
  );
}


function Metric({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
      <div className="text-[8px] uppercase tracking-[0.17em] text-slate-600">
        {label}
      </div>

      <div className="mt-1 text-sm font-semibold text-white">
        {value}
      </div>

      {sub && (
        <div className="mt-0.5 text-[9px] text-slate-600">
          {sub}
        </div>
      )}
    </div>
  );
}


// ============================================================
// PAGE
// ============================================================

export default function WorkspacePage() {

  // ----------------------------------------------------------
  // DATA
  // ----------------------------------------------------------

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [slides, setSlides] = useState<SlideInfo[]>([]);
  const [tiles, setTiles] = useState<TileInfo[]>([]);

  const [selectedSlide, setSelectedSlide] =
    useState("CMU-1-Small-Region");

  const [selectedModel, setSelectedModel] =
    useState("gigapath-flash");

  const [selectedTile, setSelectedTile] =
    useState<number | null>(null);

  const [selectedTileData, setSelectedTileData] =
    useState<TileInfo | null>(null);

  const [retrieval, setRetrieval] =
    useState<RetrievalResponse | null>(null);

  const [mil, setMil] =
    useState<MILResponse | null>(null);

  // ----------------------------------------------------------
  // UI STATE
  // ----------------------------------------------------------

  const [showMask, setShowMask] =
    useState(false);

  const [showHeatmap, setShowHeatmap] =
    useState(false);

  const [runningMIL, setRunningMIL] =
    useState(false);

  const [runningRetrieval, setRunningRetrieval] =
    useState(false);

  const [loadingTiles, setLoadingTiles] =
    useState(true);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState<string | null>(null);


  // ==========================================================
  // DERIVED
  // ==========================================================

  const slide = useMemo(
    () =>
      slides.find(
        (item) =>
          item.slide_id === selectedSlide
      ) ?? null,
    [slides, selectedSlide]
  );


  const selectedModelInfo = useMemo(
    () =>
      models.find(
        (model) =>
          model.model_id === selectedModel
      ) ?? null,
    [models, selectedModel]
  );


  const tissueTiles = useMemo(
    () =>
      [...tiles].sort(
        (a, b) =>
          b.tissue_ratio - a.tissue_ratio
      ),
    [tiles]
  );


  const selectedTileImage = selectedTileData
    ? `${API_BASE}/api/slides/${selectedSlide}/tiles/${selectedTileData.tile_id}/image`
    : null;


  const thumbnailUrl =
    `${API_BASE}/api/slides/${selectedSlide}/thumbnail`;


  const tissueMaskUrl =
    `${API_BASE}/api/slides/${selectedSlide}/tissue-mask`;


  const heatmapUrl =
    `${API_BASE}/api/analysis/${selectedSlide}/heatmap?model=${selectedModel}`;


  // ==========================================================
  // LOAD BASE DATA
  // ==========================================================

  const loadWorkspace = useCallback(
    async () => {

      try {

        setLoading(true);
        setError(null);

        const [
          modelsResponse,
          slidesResponse,
        ] = await Promise.all([
          fetch(`${API_BASE}/api/models`, {
            cache: "no-store",
          }),

          fetch(`${API_BASE}/api/slides`, {
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


        const modelData =
          await modelsResponse.json();

        const slidePayload =
          await slidesResponse.json();


        const slideData =
          normalizeSlides(slidePayload);


        setModels(modelData);
        setSlides(slideData);


        if (slideData.length > 0) {

          const firstSlide =
            slideData[0].slide_id;

          setSelectedSlide(firstSlide);

        }


        const preferredModel =
          modelData.find(
            (model: ModelInfo) =>
              model.model_id === "gigapath-flash"
          );


        if (preferredModel) {
          setSelectedModel(
            preferredModel.model_id
          );
        }


      } catch (err) {

        console.error(err);

        setError(
          err instanceof Error
            ? err.message
            : "Unable to load workspace."
        );

      } finally {

        setLoading(false);

      }

    },
    []
  );


  useEffect(() => {
    loadWorkspace();
  }, [loadWorkspace]);


  // ==========================================================
  // LOAD TILES
  // ==========================================================

  const loadTiles = useCallback(
    async () => {

      if (!selectedSlide) {
        return;
      }


      try {

        setLoadingTiles(true);

        const response =
          await fetch(
            `${API_BASE}/api/slides/${selectedSlide}/tiles`,
            {
              cache: "no-store",
            }
          );


        if (!response.ok) {
          throw new Error(
            `Tiles API returned ${response.status}`
          );
        }


        const payload =
          await response.json();


        const tileData =
          normalizeTiles(payload);


        setTiles(tileData);


        if (tileData.length > 0) {

          setSelectedTile(
            tileData[0].tile_id
          );

          setSelectedTileData(
            tileData[0]
          );

        } else {

          setSelectedTile(null);
          setSelectedTileData(null);

        }


        setRetrieval(null);
        setMil(null);

      } catch (err) {

        console.error(err);

        setError(
          err instanceof Error
            ? err.message
            : "Unable to load tissue tiles."
        );

      } finally {

        setLoadingTiles(false);

      }

    },
    [selectedSlide]
  );


  useEffect(() => {
    loadTiles();
  }, [loadTiles]);


  // ==========================================================
  // SELECT TILE
  // ==========================================================

  function selectTile(tile: TileInfo) {

    setSelectedTile(tile.tile_id);
    setSelectedTileData(tile);

    setRetrieval(null);

  }


  // ==========================================================
  // RUN MIL
  // ==========================================================

  async function runMIL() {

    if (!selectedSlide) {
      return;
    }


    try {

      setRunningMIL(true);


      const response =
        await fetch(
          `${API_BASE}/api/analysis/mil`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({
              slide_id: selectedSlide,
              model: selectedModel,
            }),
          }
        );


      if (!response.ok) {
        throw new Error(
          `MIL API returned ${response.status}`
        );
      }


      const result =
        await response.json();


      setMil(result);

      setShowHeatmap(true);

    } catch (err) {

      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "MIL analysis failed."
      );

    } finally {

      setRunningMIL(false);

    }

  }


  // ==========================================================
  // RETRIEVAL
  // ==========================================================

  async function runRetrieval() {

    if (
      selectedTile === null ||
      !selectedSlide
    ) {
      return;
    }


    try {

      setRunningRetrieval(true);


      const response =
        await fetch(
          `${API_BASE}/api/retrieval/search`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({
              slide_id:
                selectedSlide,

              tile_id:
                selectedTile,

              model:
                selectedModel,

              top_k: 5,
            }),
          }
        );


      if (!response.ok) {
        throw new Error(
          `Retrieval API returned ${response.status}`
        );
      }


      const result =
        await response.json();


      setRetrieval(result);

    } catch (err) {

      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Retrieval failed."
      );

    } finally {

      setRunningRetrieval(false);

    }

  }


  // ==========================================================
  // LOADING
  // ==========================================================

  if (loading) {

    return (
      <main className="min-h-full px-6 py-8">

        <div className="mx-auto max-w-[1500px]">

          <SectionLabel number="07">
            DIGITAL PATHOLOGY WORKSPACE
          </SectionLabel>

          <h1 className="mt-2 text-3xl font-semibold text-white">
            WSI Analysis Workbench
          </h1>

          <div className="mt-8 flex min-h-[600px] items-center justify-center rounded-2xl border border-slate-800 bg-slate-950/50">

            <div className="flex items-center gap-3 text-sm text-slate-500">

              <RefreshCw
                size={16}
                className="animate-spin text-cyan-400"
              />

              Initializing pathology workspace...

            </div>

          </div>

        </div>

      </main>
    );

  }


  // ==========================================================
  // ERROR
  // ==========================================================

  if (error && !slide) {

    return (
      <main className="min-h-full px-6 py-8">

        <div className="mx-auto max-w-7xl">

          <SectionLabel number="07">
            DIGITAL PATHOLOGY WORKSPACE
          </SectionLabel>

          <h1 className="mt-2 text-3xl font-semibold text-white">
            WSI Analysis Workbench
          </h1>


          <div className="mt-8 rounded-2xl border border-red-500/20 bg-red-500/[0.04] p-8">

            <div className="text-sm font-semibold text-white">
              Workspace unavailable
            </div>

            <div className="mt-2 text-sm text-slate-500">
              {error}
            </div>


            <button
              onClick={loadWorkspace}
              className="mt-5 inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/[0.06] px-4 py-2 text-sm text-cyan-300"
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

    <main className="min-h-full bg-[#050b12]">

      <div className="mx-auto max-w-[1600px] px-4 py-5 lg:px-6">


        {/* ================================================== */}
        {/* HEADER */}
        {/* ================================================== */}

        <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">

          <div>

            <SectionLabel number="07">
              DIGITAL PATHOLOGY WORKSPACE
            </SectionLabel>

            <div className="mt-2 flex items-center gap-3">

              <Microscope
                size={21}
                className="text-cyan-400"
              />

              <h1 className="text-2xl font-semibold tracking-tight text-white">
                WSI Analysis Workbench
              </h1>

            </div>

            <p className="mt-1 text-xs text-slate-500">
              Interactive whole-slide visualization, foundation-model
              inference, attention analysis and tissue-region retrieval.
            </p>

          </div>


          <div className="flex flex-wrap items-center gap-2">

            {/* Slide selector */}

            <div className="relative">

              <select
                value={selectedSlide}
                onChange={(event) =>
                  setSelectedSlide(
                    event.target.value
                  )
                }
                className="appearance-none rounded-lg border border-slate-700 bg-slate-950 px-4 py-2.5 pr-9 text-xs font-medium text-slate-200 outline-none transition focus:border-cyan-500/50"
              >

                {slides.map((item) => (

                  <option
                    key={item.slide_id}
                    value={item.slide_id}
                  >
                    {item.slide_id}
                  </option>

                ))}

              </select>

              <ChevronDown
                size={13}
                className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-500"
              />

            </div>


            {/* Model selector */}

            <div className="relative">

              <select
                value={selectedModel}
                onChange={(event) =>
                  setSelectedModel(
                    event.target.value
                  )
                }
                className="appearance-none rounded-lg border border-cyan-500/30 bg-slate-950 px-4 py-2.5 pr-9 text-xs font-medium text-cyan-200 outline-none transition focus:border-cyan-400"
              >

                {models.map((model) => (

                  <option
                    key={model.model_id}
                    value={model.model_id}
                  >
                    {model.name}
                  </option>

                ))}

              </select>

              <ChevronDown
                size={13}
                className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-500"
              />

            </div>


            {/* Refresh */}

            <button
              onClick={loadTiles}
              className="rounded-lg border border-slate-700 bg-slate-950 p-2.5 text-slate-400 transition hover:border-cyan-500/30 hover:text-cyan-300"
              title="Refresh tiles"
            >

              <RefreshCw
                size={15}
              />

            </button>

          </div>

        </div>


        {/* ================================================== */}
        {/* ERROR BANNER */}
        {/* ================================================== */}

        {error && (

          <div className="mb-4 flex items-center justify-between rounded-lg border border-amber-500/20 bg-amber-500/[0.04] px-4 py-3">

            <div className="text-xs text-amber-300">
              {error}
            </div>

            <button
              onClick={() => setError(null)}
              className="text-[10px] text-slate-500 hover:text-white"
            >
              dismiss
            </button>

          </div>

        )}


        {/* ================================================== */}
        {/* MAIN WORKSPACE */}
        {/* ================================================== */}

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_370px]">


          {/* ================================================= */}
          {/* LEFT VIEWER */}
          {/* ================================================= */}

          <section className="overflow-hidden rounded-2xl border border-slate-800 bg-[#03080e]">


            {/* Viewer toolbar */}

            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 bg-slate-950/70 px-4 py-3">

              <div className="flex items-center gap-3">

                <div className="flex items-center gap-2">

                  <span className="h-2 w-2 rounded-full bg-emerald-400" />

                  <span className="text-xs font-medium text-slate-300">
                    {slide?.slide_id}
                  </span>

                </div>

                <span className="text-[10px] text-slate-700">
                  |
                </span>

                <span className="text-[10px] text-slate-500">
                  {slide?.width} × {slide?.height}px
                </span>

              </div>


              <div className="flex items-center gap-2">

                <button
                  onClick={() =>
                    setShowMask(
                      !showMask
                    )
                  }
                  className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-[10px] transition ${
                    showMask
                      ? "border-cyan-500/30 bg-cyan-500/[0.08] text-cyan-300"
                      : "border-slate-800 bg-slate-950 text-slate-500 hover:text-slate-300"
                  }`}
                >

                  <Eye size={12} />

                  Tissue Mask

                </button>


                <button
                  onClick={() =>
                    setShowHeatmap(
                      !showHeatmap
                    )
                  }
                  className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-[10px] transition ${
                    showHeatmap
                      ? "border-amber-500/30 bg-amber-500/[0.08] text-amber-300"
                      : "border-slate-800 bg-slate-950 text-slate-500 hover:text-slate-300"
                  }`}
                >

                  <Activity size={12} />

                  Attention Heatmap

                </button>

              </div>

            </div>


            {/* Viewer */}

            <div className="relative h-[620px] overflow-hidden bg-[#02060a]">


              <TransformWrapper
                initialScale={0.45}
                minScale={0.2}
                maxScale={8}
                centerOnInit
                wheel={{
                  step: 0.12,
                }}
                doubleClick={{
                  mode: "zoomIn",
                }}
              >

                {({
                  zoomIn,
                  zoomOut,
                  resetTransform,
                }) => (

                  <>

                    {/* Viewer controls */}

                    <div className="absolute left-4 top-4 z-30 flex flex-col overflow-hidden rounded-lg border border-slate-700 bg-slate-950/90 shadow-2xl">

                      <button
                        onClick={() =>
                          zoomIn()
                        }
                        className="border-b border-slate-800 p-2.5 text-slate-400 transition hover:bg-slate-900 hover:text-cyan-300"
                        title="Zoom in"
                      >
                        <ZoomIn size={15} />
                      </button>


                      <button
                        onClick={() =>
                          zoomOut()
                        }
                        className="border-b border-slate-800 p-2.5 text-slate-400 transition hover:bg-slate-900 hover:text-cyan-300"
                        title="Zoom out"
                      >
                        <ZoomOut size={15} />
                      </button>


                      <button
                        onClick={() =>
                          resetTransform()
                        }
                        className="p-2.5 text-slate-400 transition hover:bg-slate-900 hover:text-cyan-300"
                        title="Reset"
                      >
                        <Crosshair size={15} />
                      </button>

                    </div>


                    {/* Pan hint */}

                    <div className="absolute bottom-4 left-4 z-30 flex items-center gap-2 rounded-md border border-slate-800 bg-slate-950/85 px-3 py-2 text-[9px] text-slate-500">

                      <Move size={11} />

                      Scroll to zoom · drag to pan · double-click to zoom

                    </div>


                    <TransformComponent
                      wrapperStyle={{
                        width: "100%",
                        height: "100%",
                      }}

                      contentStyle={{
                        width: "100%",
                        height: "100%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >

                      <div
                        className="relative"
                        style={{
                          width:
                            slide?.width ??
                            2220,

                          height:
                            slide?.height ??
                            2967,

                          maxWidth:
                            "none",
                        }}
                      >

                        {/* Whole slide */}

                        <img
                          src={thumbnailUrl}
                          alt={selectedSlide}
                          draggable={false}
                          className="absolute inset-0 h-full w-full object-fill"
                        />


                        {/* Tissue mask */}

                        {showMask && (

                          <img
                            src={tissueMaskUrl}
                            alt="Tissue mask"
                            draggable={false}
                            className="absolute inset-0 h-full w-full object-fill opacity-35 mix-blend-screen"
                          />

                        )}


                        {/* Heatmap */}

                        {showHeatmap && (

                          <img
                            src={heatmapUrl}
                            alt="Attention heatmap"
                            draggable={false}
                            className="absolute inset-0 h-full w-full object-fill opacity-55 mix-blend-screen"
                          />

                        )}


                        {/* Tile regions */}

                        {tiles.map((tile) => {

                          const left =
                            ((tile.x /
                              (slide?.width ||
                                2220)) *
                              100);

                          const top =
                            ((tile.y /
                              (slide?.height ||
                                2967)) *
                              100);

                          const width =
                            ((tile.width /
                              (slide?.width ||
                                2220)) *
                              100);

                          const height =
                            ((tile.height /
                              (slide?.height ||
                                2967)) *
                              100);

                          const active =
                            tile.tile_id ===
                            selectedTile;


                          return (

                            <button
                              key={tile.tile_id}
                              onClick={() =>
                                selectTile(tile)
                              }
                              title={`Tile ${tile.tile_id} · Tissue ${percent(tile.tissue_ratio)}`}
                              className={`absolute transition ${
                                active
                                  ? "z-20 border-2 border-cyan-300 bg-cyan-300/10 shadow-[0_0_12px_rgba(34,211,238,0.5)]"
                                  : "border border-cyan-400/0 hover:border-cyan-300/60 hover:bg-cyan-300/[0.04]"
                              }`}
                              style={{
                                left: `${left}%`,
                                top: `${top}%`,
                                width: `${width}%`,
                                height: `${height}%`,
                              }}
                            />

                          );

                        })}

                      </div>

                    </TransformComponent>

                  </>

                )}

              </TransformWrapper>

            </div>


            {/* Viewer status bar */}

            <div className="grid grid-cols-2 border-t border-slate-800 bg-slate-950/70 sm:grid-cols-4">

              <div className="border-r border-slate-800 px-4 py-3">

                <div className="text-[8px] uppercase tracking-wider text-slate-600">
                  Objective
                </div>

                <div className="mt-1 text-xs font-semibold text-slate-300">
                  {slide?.objective_power
                    ? `${slide.objective_power}×`
                    : "20×"}
                </div>

              </div>


              <div className="border-r border-slate-800 px-4 py-3">

                <div className="text-[8px] uppercase tracking-wider text-slate-600">
                  Resolution
                </div>

                <div className="mt-1 text-xs font-semibold text-slate-300">
                  {slide?.mpp_x
                    ? `${slide.mpp_x.toFixed(3)} µm/px`
                    : "0.499 µm/px"}
                </div>

              </div>


              <div className="border-r border-slate-800 px-4 py-3">

                <div className="text-[8px] uppercase tracking-wider text-slate-600">
                  Tissue Tiles
                </div>

                <div className="mt-1 text-xs font-semibold text-cyan-300">
                  {loadingTiles
                    ? "..."
                    : tiles.length}
                </div>

              </div>


              <div className="px-4 py-3">

                <div className="text-[8px] uppercase tracking-wider text-slate-600">
                  Viewer Mode
                </div>

                <div className="mt-1 text-xs font-semibold text-emerald-300">
                  Interactive
                </div>

              </div>

            </div>

          </section>


          {/* ================================================= */}
          {/* RIGHT CONTROL PANEL */}
          {/* ================================================= */}

          <aside className="space-y-4">


            {/* Slide metadata */}

            <section className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5">

              <div className="flex items-center gap-2">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2">

                  <FileImage
                    size={15}
                    className="text-cyan-400"
                  />

                </div>

                <div>

                  <div className="text-sm font-semibold text-white">
                    Slide Information
                  </div>

                  <div className="text-[9px] text-slate-600">
                    Registered WSI metadata
                  </div>

                </div>

              </div>


              <div className="mt-5 grid grid-cols-2 gap-2">

                <Metric
                  label="Dimensions"
                  value={
                    slide
                      ? `${slide.width} × ${slide.height}`
                      : "—"
                  }
                />

                <Metric
                  label="Levels"
                  value={
                    slide
                      ? String(slide.levels)
                      : "—"
                  }
                />

                <Metric
                  label="MPP"
                  value={
                    slide?.mpp_x
                      ? slide.mpp_x.toFixed(3)
                      : "—"
                  }
                  sub="µm / pixel"
                />

                <Metric
                  label="Objective"
                  value={
                    slide?.objective_power
                      ? `${slide.objective_power}×`
                      : "—"
                  }
                />

              </div>

            </section>


            {/* Model */}

            <section className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5">

              <div className="flex items-center gap-2">

                <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/[0.07] p-2">

                  <BrainCircuit
                    size={15}
                    className="text-cyan-400"
                  />

                </div>

                <div>

                  <div className="text-sm font-semibold text-white">
                    Foundation Encoder
                  </div>

                  <div className="text-[9px] text-slate-600">
                    Active inference model
                  </div>

                </div>

              </div>


              <div className="mt-5 rounded-xl border border-cyan-500/20 bg-cyan-500/[0.025] p-4">

                <div className="text-base font-semibold text-white">
                  {selectedModelInfo?.name ??
                    selectedModel}
                </div>

                <div className="mt-1 text-[10px] text-slate-500">
                  {selectedModelInfo?.modality}
                </div>


                <div className="mt-4 grid grid-cols-2 gap-2">

                  <Metric
                    label="Embedding"
                    value={
                      selectedModelInfo
                        ? `${selectedModelInfo.embedding_dimension}-D`
                        : "—"
                    }
                  />

                  <Metric
                    label="Input"
                    value={
                      selectedModelInfo
                        ? `${selectedModelInfo.input_size}px`
                        : "—"
                    }
                  />

                </div>

              </div>

            </section>


            {/* MIL */}

            <section className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5">

              <div className="flex items-center justify-between">

                <div className="flex items-center gap-2">

                  <div className="rounded-lg border border-amber-500/20 bg-amber-500/[0.06] p-2">

                    <Sparkles
                      size={15}
                      className="text-amber-400"
                    />

                  </div>

                  <div>

                    <div className="text-sm font-semibold text-white">
                      WSI Attention MIL
                    </div>

                    <div className="text-[9px] text-slate-600">
                      Whole-slide inference
                    </div>

                  </div>

                </div>


                {mil?.prototype && (

                  <span className="rounded-full border border-amber-500/20 bg-amber-500/[0.05] px-2 py-1 text-[8px] font-semibold uppercase tracking-wider text-amber-400">
                    Prototype
                  </span>

                )}

              </div>


              <button
                onClick={runMIL}
                disabled={runningMIL}
                className="mt-5 flex w-full items-center justify-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/[0.07] px-4 py-2.5 text-xs font-semibold text-cyan-300 transition hover:bg-cyan-500/[0.13] disabled:cursor-not-allowed disabled:opacity-50"
              >

                {runningMIL ? (

                  <>
                    <RefreshCw
                      size={13}
                      className="animate-spin"
                    />

                    Running WSI MIL...

                  </>

                ) : (

                  <>
                    <Activity size={13} />

                    Run Attention MIL

                  </>

                )}

              </button>


              {mil && (

                <div className="mt-4 space-y-3">

                  <div className="grid grid-cols-2 gap-2">

                    <Metric
                      label="Prediction"
                      value={`Class ${mil.prediction.class_id}`}
                    />

                    <Metric
                      label="Probability"
                      value={probability(
                        mil.prediction.probability
                      )}
                    />

                    <Metric
                      label="Tiles"
                      value={String(
                        mil.attention.tile_count
                      )}
                    />

                    <Metric
                      label="Attention Sum"
                      value={
                        mil.attention.sum.toFixed(4)
                      }
                    />

                  </div>


                  <div className="rounded-lg border border-amber-500/15 bg-amber-500/[0.025] p-3">

                    <div className="text-[8px] uppercase tracking-wider text-slate-600">
                      MIL Architecture
                    </div>

                    <div className="mt-1 text-xs font-semibold text-slate-300">
                      {mil.mil.architecture}
                    </div>

                    <div className="mt-1 text-[9px] text-slate-600">
                      Hidden {mil.mil.hidden_dimension}
                      {" · "}
                      Attention {mil.mil.attention_dimension}
                      {" · "}
                      {mil.mil.trained
                        ? "trained"
                        : "untrained"}
                    </div>

                  </div>

                </div>

              )}

            </section>

          </aside>

        </div>


        {/* ================================================== */}
        {/* TILE ANALYSIS */}
        {/* ================================================== */}

        <section className="mt-4 grid gap-4 xl:grid-cols-[360px_minmax(0,1fr)]">


          {/* ================================================= */}
          {/* SELECTED TILE */}
          {/* ================================================= */}

          <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5">

            <SectionLabel number="01">
              REGION INSPECTION
            </SectionLabel>

            <h2 className="mt-2 text-lg font-semibold text-white">
              Selected Tissue Region
            </h2>


            {selectedTileData ? (

              <>

                <div className="mt-4 overflow-hidden rounded-xl border border-slate-800 bg-black">

                  {selectedTileImage && (

                    <img
                      src={selectedTileImage}
                      alt={`Tile ${selectedTileData.tile_id}`}
                      className="aspect-square w-full object-cover"
                    />

                  )}

                </div>


                <div className="mt-4 grid grid-cols-2 gap-2">

                  <Metric
                    label="Tile ID"
                    value={`#${selectedTileData.tile_id}`}
                  />

                  <Metric
                    label="Tissue"
                    value={percent(
                      selectedTileData.tissue_ratio
                    )}
                  />

                  <Metric
                    label="X"
                    value={
                      selectedTileData.x.toLocaleString()
                    }
                  />

                  <Metric
                    label="Y"
                    value={
                      selectedTileData.y.toLocaleString()
                    }
                  />

                </div>


                <button
                  onClick={runRetrieval}
                  disabled={runningRetrieval}
                  className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/[0.07] px-4 py-2.5 text-xs font-semibold text-cyan-300 transition hover:bg-cyan-500/[0.13] disabled:opacity-50"
                >

                  {runningRetrieval ? (

                    <>
                      <RefreshCw
                        size={13}
                        className="animate-spin"
                      />

                      Searching Similar Regions...

                    </>

                  ) : (

                    <>
                      <Search size={13} />

                      Find Similar Tissue

                    </>

                  )}

                </button>

              </>

            ) : (

              <div className="mt-4 flex aspect-square items-center justify-center rounded-xl border border-dashed border-slate-800 text-xs text-slate-600">

                Select a tissue tile from the viewer.

              </div>

            )}

          </div>


          {/* ================================================= */}
          {/* RETRIEVAL / ATTENTION */}
          {/* ================================================= */}

          <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5">

            <div className="flex items-center justify-between">

              <div>

                <SectionLabel number="02">
                  TISSUE INTELLIGENCE
                </SectionLabel>

                <h2 className="mt-2 text-lg font-semibold text-white">
                  Analysis Regions
                </h2>

              </div>


              <div className="flex items-center gap-2">

                {retrieval && (

                  <span className="rounded-full border border-cyan-500/20 bg-cyan-500/[0.06] px-2 py-1 text-[8px] font-semibold uppercase tracking-wider text-cyan-300">
                    {retrieval.results.length} Similar
                  </span>

                )}

                {mil && (

                  <span className="rounded-full border border-amber-500/20 bg-amber-500/[0.06] px-2 py-1 text-[8px] font-semibold uppercase tracking-wider text-amber-300">
                    {mil.attention.top_k} Attention
                  </span>

                )}

              </div>

            </div>


            {/* Retrieval */}

            {retrieval ? (

              <div className="mt-5">

                <div className="mb-3 flex items-center gap-2">

                  <Search
                    size={14}
                    className="text-cyan-400"
                  />

                  <span className="text-xs font-semibold text-slate-300">
                    Similar Tissue Regions
                  </span>

                  <span className="text-[9px] text-slate-600">
                    {retrieval.model}
                  </span>

                </div>


                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">

                  {retrieval.results.map(
                    (result, index) => {

                      const imageUrl =
                        `${API_BASE}/api/slides/${selectedSlide}/tiles/${result.tile_id}/image`;

                      return (

                        <button
                          key={`${result.tile_id}-${index}`}
                          onClick={() => {

                            const tile =
                              tiles.find(
                                (item) =>
                                  item.tile_id ===
                                  result.tile_id
                              );

                            if (tile) {
                              selectTile(tile);
                            }

                          }}
                          className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950/50 text-left transition hover:border-cyan-500/40"
                        >

                          <div className="relative">

                            <img
                              src={imageUrl}
                              alt={`Retrieved tile ${result.tile_id}`}
                              className="aspect-square w-full object-cover"
                            />

                            <div className="absolute left-2 top-2 rounded-md border border-slate-700 bg-slate-950/90 px-1.5 py-1 text-[8px] font-semibold text-white">
                              #{index + 1}
                            </div>

                          </div>


                          <div className="p-3">

                            <div className="flex items-center justify-between">

                              <span className="text-[9px] uppercase tracking-wider text-slate-600">
                                Tile
                              </span>

                              <span className="text-xs font-semibold text-white">
                                #{result.tile_id}
                              </span>

                            </div>


                            <div className="mt-2">

                              <div className="mb-1 flex justify-between">

                                <span className="text-[8px] uppercase tracking-wider text-slate-600">
                                  Similarity
                                </span>

                                <span className="text-[9px] font-semibold text-cyan-300">
                                  {score(result.score)}
                                </span>

                              </div>


                              <div className="h-1 overflow-hidden rounded-full bg-slate-800">

                                <div
                                  className="h-full rounded-full bg-cyan-400"
                                  style={{
                                    width: `${Math.min(
                                      result.score * 100,
                                      100
                                    )}%`,
                                  }}
                                />

                              </div>

                            </div>


                            <div className="mt-2 text-[8px] text-slate-600">
                              ({result.x}, {result.y})
                            </div>

                          </div>

                        </button>

                      );

                    }
                  )}

                </div>

              </div>

            ) : mil ? (

              /* Attention */

              <div className="mt-5">

                <div className="mb-3 flex items-center gap-2">

                  <Target
                    size={14}
                    className="text-amber-400"
                  />

                  <span className="text-xs font-semibold text-slate-300">
                    Highest Attention Regions
                  </span>

                </div>


                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">

                  {mil.attention.top_tiles
                    .slice(0, 10)
                    .map((tile) => {

                      const imageUrl =
                        `${API_BASE}/api/slides/${selectedSlide}/tiles/${tile.tile_id}/image`;

                      return (

                        <button
                          key={tile.tile_id}
                          onClick={() => {

                            const localTile =
                              tiles.find(
                                (item) =>
                                  item.tile_id ===
                                  tile.tile_id
                              );

                            if (localTile) {
                              selectTile(localTile);
                            }

                          }}
                          className="overflow-hidden rounded-lg border border-slate-800 bg-slate-950/50 text-left transition hover:border-amber-500/40"
                        >

                          <div className="relative">

                            <img
                              src={imageUrl}
                              alt={`Attention tile ${tile.tile_id}`}
                              className="aspect-square w-full object-cover"
                            />

                            <div className="absolute left-1.5 top-1.5 rounded bg-slate-950/90 px-1.5 py-1 text-[8px] font-semibold text-amber-300">
                              #{tile.rank}
                            </div>

                          </div>


                          <div className="p-2">

                            <div className="text-[9px] font-semibold text-white">
                              Tile {tile.tile_id}
                            </div>

                            <div className="mt-1 text-[8px] text-slate-600">
                              Attention{" "}
                              {tile.attention.toFixed(4)}
                            </div>

                          </div>

                        </button>

                      );

                    })}

                </div>

              </div>

            ) : (

              <div className="mt-5 flex min-h-[280px] items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/30">

                <div className="max-w-sm text-center">

                  <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-lg border border-cyan-500/20 bg-cyan-500/[0.05]">

                    <Sparkles
                      size={16}
                      className="text-cyan-400"
                    />

                  </div>

                  <div className="mt-3 text-sm font-semibold text-slate-300">
                    Ready for tissue intelligence
                  </div>

                  <div className="mt-1 text-[10px] leading-5 text-slate-600">
                    Select a region and run similarity search,
                    or execute whole-slide MIL to identify
                    high-attention tissue regions.
                  </div>

                </div>

              </div>

            )}

          </div>

        </section>


        {/* ================================================== */}
        {/* TILE INVENTORY */}
        {/* ================================================== */}

        <section className="mt-4 rounded-2xl border border-slate-800 bg-slate-950/50 p-5">

          <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">

            <div>

              <SectionLabel number="03">
                TISSUE TILE INVENTORY
              </SectionLabel>

              <h2 className="mt-2 text-lg font-semibold text-white">
                Tissue-Aware Regions
              </h2>

              <p className="mt-1 text-[10px] text-slate-600">
                {tiles.length} extracted regions ordered by tissue
                occupancy.
              </p>

            </div>


            <div className="flex items-center gap-4 text-[9px] text-slate-600">

              <div className="flex items-center gap-1.5">

                <span className="h-2 w-2 rounded-full bg-cyan-400" />

                Selected

              </div>

              <div className="flex items-center gap-1.5">

                <span className="h-2 w-2 rounded-full bg-slate-700" />

                Available

              </div>

            </div>

          </div>


          {loadingTiles ? (

            <div className="mt-5 flex h-32 items-center justify-center text-xs text-slate-600">

              <RefreshCw
                size={14}
                className="mr-2 animate-spin"
              />

              Loading tissue inventory...

            </div>

          ) : (

            <div className="mt-5 grid grid-cols-3 gap-2 sm:grid-cols-5 md:grid-cols-7 lg:grid-cols-9 xl:grid-cols-12">

              {tissueTiles.map((tile) => {

                const active =
                  tile.tile_id ===
                  selectedTile;


                return (

                  <button
                    key={tile.tile_id}
                    onClick={() =>
                      selectTile(tile)
                    }
                    className={`group overflow-hidden rounded-lg border transition ${
                      active
                        ? "border-cyan-400 bg-cyan-400/[0.06]"
                        : "border-slate-800 bg-slate-950/40 hover:border-slate-600"
                    }`}
                  >

                    <img
                      src={`${API_BASE}/api/slides/${selectedSlide}/tiles/${tile.tile_id}/image`}
                      alt={`Tile ${tile.tile_id}`}
                      loading="lazy"
                      className="aspect-square w-full object-cover"
                    />


                    <div className="px-2 py-1.5">

                      <div className="flex items-center justify-between">

                        <span
                          className={`text-[8px] font-semibold ${
                            active
                              ? "text-cyan-300"
                              : "text-slate-500"
                          }`}
                        >
                          #{tile.tile_id}
                        </span>

                        <span className="text-[7px] text-slate-700">
                          {(
                            tile.tissue_ratio *
                            100
                          ).toFixed(0)}
                          %
                        </span>

                      </div>

                    </div>

                  </button>

                );

              })}

            </div>

          )}

        </section>


        {/* ================================================== */}
        {/* FOOTER */}
        {/* ================================================== */}

        <div className="flex flex-col items-center justify-between gap-2 py-7 text-[9px] text-slate-700 sm:flex-row">

          <div className="flex items-center gap-2">

            <Database size={11} />

            PathoVerse AI · Whole-slide analysis engine

          </div>


          <div className="flex items-center gap-2">

            <Layers3 size={11} />

            OpenSlide · Foundation Models · MIL · FAISS

          </div>

        </div>

      </div>

    </main>

  );
}