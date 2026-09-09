export interface HealthResponse {
  status: string;
  service: string;
  project: string;
  project_root_exists: boolean;
  timestamp: string;
}

export interface ModelInfo {
  name: string;
  model_id: string;
  embedding_dimension: number;
  input_size: number;
  modality: string;
  description: string;
  status: string;
}

export interface SlideSummary {
  slide_id: string;
  width: number;
  height: number;
  levels: number;
  mpp_x: number | null;
  mpp_y: number | null;
  tissue_ratio: number | null;
  tile_count: number | null;
}

export interface TileInfo {
  tile_id: number;
  slide_id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  level: number;
  tissue_ratio: number;
}

export interface RetrievalResult {
  tile_id: number;
  score: number;
}

export interface RetrievalResponse {
  slide_id: string;
  model_id: string;
  query_tile_id: number;
  top_k: number;
  results: RetrievalResult[];
}

export interface ClassificationMetrics {
  accuracy: number | null;
  auroc: number | null;
  f1: number | null;
  precision: number | null;
  recall: number | null;
  sensitivity?: number | null;
  specificity?: number | null;
}

export interface ClassificationResponse {
  model_id: string;
  dataset: string;
  samples: number;
  metrics: ClassificationMetrics;
}

export interface MILResponse {
  slide_id: string;
  model_id: string;
  prediction: number;
  probability: number;
  attention_sum: number;
  top_tiles: {
    tile_id: number;
    attention: number;
  }[];
}

export interface BenchmarkRecord {
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
}