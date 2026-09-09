export interface ModelInfo {
  name: string;
  model_id: string;
  embedding_dimension: number;
  input_size: number;
  modality: string;
  description: string;
  status: string;
}

export interface SlideInfo {
  slide_id: string;
  filename: string;
  width: number;
  height: number;
  levels: number;
  mpp_x: number;
  mpp_y: number;
  tissue_ratio: number | null;
  tile_count: number | null;
}

export interface ClassificationResponse {
  model: string;
  dataset: string;
  accuracy: number | null;
  auroc: number | null;
  f1: number | null;
  precision: number | null;
  recall: number | null;
  sensitivity: number | null;
  specificity: number | null;
}

export interface MILFoundationModel {
  name: string;
  embedding_dimension: number;
}

export interface MILMetadata {
  architecture: string;
  hidden_dimension: number;
  attention_dimension: number;
  num_classes: number;
  trained: boolean;
  prototype: boolean;
  seed: number | null;
}

export interface MILPrediction {
  class_id: number;
  probability: number;
}

export interface MILTopTile {
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
  image_path?: string;
}

export interface MILAttention {
  tile_count: number;
  sum: number;
  top_k: number;
  top_tiles: MILTopTile[];
}

export interface MILResponse {
  schema_version: string;
  task: string;
  slide_id: string;
  model: string;
  foundation_model: MILFoundationModel;
  mil: MILMetadata;
  prediction: MILPrediction;
  attention: MILAttention;
  slide_embedding_dimension: number;
  slide_embedding_path: string;
  trained: boolean;
  prototype: boolean;
}