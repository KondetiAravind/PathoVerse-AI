import type {
  ClassificationResponse,
  MILResponse,
  ModelInfo,
  SlideInfo,
} from "@/types/analysis";
import { apiFetch } from "@/lib/api";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await apiFetch(`${API_BASE}${path}`, {
    ...options,
    cache: "no-store",
  });

  if (!response.ok) {
    let message = `Request failed with HTTP ${response.status}`;
    try {
      const error = await response.json();
      if (typeof error?.detail === "string") message = error.detail;
      else if (typeof error?.error?.message === "string") message = error.error.message;
      else if (Array.isArray(error?.detail)) {
        message = error.detail.map((item: { msg?: string }) => item.msg || "Validation error").join(", ");
      }
    } catch {
      // Keep default message.
    }
    throw new Error(message);
  }

  return response.json();
}

export async function getModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>("/api/models");
}

export async function getSlides(): Promise<SlideInfo[]> {
  return request<SlideInfo[]>("/api/slides");
}

export async function runClassification(model: string, dataset = "patchcamelyon"): Promise<ClassificationResponse> {
  return request<ClassificationResponse>("/api/analysis/classification", {
    method: "POST",
    body: JSON.stringify({ model, dataset }),
  });
}

export async function runMIL(slideId: string, model = "gigapath-flash"): Promise<MILResponse> {
  return request<MILResponse>("/api/analysis/mil", {
    method: "POST",
    body: JSON.stringify({ slide_id: slideId, model }),
  });
}

export function getHeatmapUrl(slideId: string, model: string): string {
  return `${API_BASE}/api/analysis/${encodeURIComponent(slideId)}/heatmap?model=${encodeURIComponent(model)}`;
}

export function getTileImageUrl(slideId: string, tileId: number): string {
  return `${API_BASE}/api/slides/${encodeURIComponent(slideId)}/tiles/${tileId}/image`;
}
