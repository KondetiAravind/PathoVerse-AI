import axios from "axios";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

const API_KEY = process.env.NEXT_PUBLIC_PATHOVERSE_API_KEY || "";

export async function apiFetch(
  input: RequestInfo | URL,
  init: RequestInit = {}
): Promise<Response> {
  const headers = new Headers(init.headers);

  if (!headers.has("Content-Type") && init.body) {
    headers.set("Content-Type", "application/json");
  }

  if (API_KEY) {
    headers.set("Authorization", `Bearer ${API_KEY}`);
  }

  return fetch(input, {
    ...init,
    headers,
  });
}

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.request.use((config) => {
  if (API_KEY) {
    config.headers.Authorization = `Bearer ${API_KEY}`;
  }
  return config;
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
  const response = await api.get(`/api/models/${encodeURIComponent(modelId)}`);
  return response.data;
}

export async function getSlides() {
  const response = await api.get("/api/slides");
  return response.data;
}

export async function getSlide(slideId: string) {
  const response = await api.get(
    `/api/slides/${encodeURIComponent(slideId)}`
  );
  return response.data;
}

export async function getTiles(slideId: string) {
  const response = await api.get(
    `/api/slides/${encodeURIComponent(slideId)}/tiles`
  );
  return response.data;
}

export function getTileImageUrl(slideId: string, tileId: number): string {
  return `${API_BASE_URL}/api/slides/${encodeURIComponent(
    slideId
  )}/tiles/${tileId}/image`;
}

export function getThumbnailUrl(slideId: string): string {
  return `${API_BASE_URL}/api/slides/${encodeURIComponent(slideId)}/thumbnail`;
}

export function getTissueMaskUrl(slideId: string): string {
  return `${API_BASE_URL}/api/slides/${encodeURIComponent(
    slideId
  )}/tissue-mask`;
}

export function getHeatmapUrl(
  slideId: string,
  model = "gigapath-flash"
): string {
  return `${API_BASE_URL}/api/analysis/${encodeURIComponent(
    slideId
  )}/heatmap?model=${encodeURIComponent(model)}`;
}

/**
 * Frontend-friendly retrieval adapter.
 *
 * The backend contract is:
 *   slide_id, tile_id, model, top_k
 *
 * Keep model_id/query_tile_id here for compatibility with existing pages,
 * but translate them before making the actual API request.
 */
export async function searchRetrieval(payload: {
  slide_id: string;
  model_id: string;
  query_tile_id: number;
  top_k: number;
}) {
  const response = await api.post("/api/retrieval/search", {
    slide_id: payload.slide_id,
    tile_id: payload.query_tile_id,
    model: payload.model_id,
    top_k: payload.top_k,
  });

  return response.data;
}

export async function runClassification(payload: {
  model_id: string;
  dataset: string;
}) {
  const response = await api.post("/api/analysis/classification", {
    model: payload.model_id,
    dataset: payload.dataset,
  });

  return response.data;
}

export async function runMIL(payload: {
  slide_id: string;
  model_id: string;
}) {
  const response = await api.post("/api/analysis/mil", {
    slide_id: payload.slide_id,
    model: payload.model_id,
  });

  return response.data;
}

export async function getBenchmarks() {
  const response = await api.get("/api/benchmarks");
  return response.data;
}

export async function getBenchmark(task: string) {
  const response = await api.get(
    `/api/benchmarks/${encodeURIComponent(task)}`
  );
  return response.data;
}
