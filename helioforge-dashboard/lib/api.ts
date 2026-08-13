export type Prediction = {
  predicted_class: number;
  predicted_label: "Quiet" | "B" | "C" | "M" | "X";
  confidence: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "EXTREME";
  probabilities: {
    Quiet: number;
    B: number;
    C: number;
    M: number;
    X: number;
  };
  observation_id?: string;
  sample_index?: number;
  input_shape?: number[];
  processing_time_ms?: number;
  signal?: number[];
  features?: Record<string, number>;
  rgb_intensity?: { red: number; green: number; blue: number };
  active_regions?: ActiveRegion[];
};

export type ActiveRegion = { id: string; lat: number; lon: number; class: number; confidence?: number; intensity?: number };

export type Health = {
  status: string;
  model: string;
  checkpoint: string;
  epoch: number;
  macro_f1: number;
  input_shape: number[];
  classes: string[];
};

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

export async function getHealth(): Promise<Health> {
  const response = await fetch(`${API_BASE_URL}/api/health`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Health request failed: ${response.status}`);
  }

  return response.json();
}

export async function getDemoPrediction(
  index = 0
): Promise<Prediction> {
  const response = await fetch(
    `${API_BASE_URL}/api/demo/${index}`,
    {
      cache: "no-store",
    }
  );

  if (!response.ok) {
    throw new Error(
      `Prediction request failed: ${response.status}`
    );
  }

  return response.json();
}

export async function predictFile(file: File): Promise<Prediction> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE_URL}/api/predict`, { method: "POST", body: form });
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || `Upload failed: ${response.status}`);
  return response.json();
}
