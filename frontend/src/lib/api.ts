import axios from "axios";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export async function getHealth() {
  const response = await api.get("/api/health");
  return response.data;
}

export async function getModels() {
  const response = await api.get("/api/models");
  return response.data;
}

export async function getModel(modelId: string) {
  const response = await api.get(`/api/models/${modelId}`);
  return response.data;
}

export async function getSlides() {
  const response = await api.get("/api/slides");
  return response.data;
}

export async function getSlide(slideId: string) {
  const response = await api.get(`/api/slides/${slideId}`);
  return response.data;
}

export async function getTiles(slideId: string) {
  const response = await api.get(`/api/slides/${slideId}/tiles`);
  return response.data;
}

export function getTileImageUrl(
  slideId: string,
  tileId: number
): string {
  return `${API_BASE_URL}/api/slides/${slideId}/tiles/${tileId}/image`;
}

export function getThumbnailUrl(slideId: string): string {
  return `${API_BASE_URL}/api/slides/${slideId}/thumbnail`;
}

export function getTissueMaskUrl(slideId: string): string {
  return `${API_BASE_URL}/api/slides/${slideId}/tissue-mask`;
}

export function getHeatmapUrl(slideId: string): string {
  return `${API_BASE_URL}/api/analysis/${slideId}/heatmap`;
}

export async function searchRetrieval(payload: {
  slide_id: string;
  model_id: string;
  query_tile_id: number;
  top_k: number;
}) {
  const response = await api.post(
    "/api/retrieval/search",
    payload
  );

  return response.data;
}

export async function runClassification(payload: {
  model_id: string;
  dataset: string;
}) {
  const response = await api.post(
    "/api/analysis/classification",
    payload
  );

  return response.data;
}

export async function runMIL(payload: {
  slide_id: string;
  model_id: string;
}) {
  const response = await api.post(
    "/api/analysis/mil",
    payload
  );

  return response.data;
}

export async function getBenchmarks() {
  const response = await api.get("/api/benchmarks");
  return response.data;
}

export async function getBenchmark(task: string) {
  const response = await api.get(
    `/api/benchmarks/${task}`
  );

  return response.data;
}