"use client";
import { create } from "zustand";
import {
  getDemoPrediction,
  getHealth,
  predictFile,
  generateFallbackPrediction,
  type Prediction,
  type Health,
  type ActiveRegion,
} from "@/lib/api";

export type WavelengthMode = "171" | "304" | "131" | "hmi" | "rgb";

export type ActiveSection =
  | "control"
  | "sun"
  | "evolution"
  | "prediction"
  | "intensity"
  | "signals"
  | "features"
  | "upload"
  | "performance"
  | "animation"
  | "forecast";

interface PredictionStoreState {
  prediction: Prediction;
  health: Health | null;
  isLoading: boolean;
  error: string | null;
  activeSection: ActiveSection;
  wavelength: WavelengthMode;
  selectedRegion: ActiveRegion | null;
  demoIndex: number;

  // Actions
  setPrediction: (prediction: Prediction) => void;
  setActiveSection: (section: ActiveSection) => void;
  setWavelength: (mode: WavelengthMode) => void;
  setSelectedRegion: (region: ActiveRegion | null) => void;
  fetchDemo: (index?: number) => Promise<void>;
  predict: (file: File) => Promise<void>;
  fetchHealth: () => Promise<void>;
}

const initialPrediction = generateFallbackPrediction(0);

export const usePredictionStore = create<PredictionStoreState>((set) => ({
  prediction: initialPrediction,
  health: null,
  isLoading: false,
  error: null,
  activeSection: "control",
  wavelength: "171",
  selectedRegion: initialPrediction.active_regions?.[0] || null,
  demoIndex: 0,

  setPrediction: (prediction) =>
    set({
      prediction,
      selectedRegion: prediction.active_regions?.[0] || null,
      error: null,
    }),

  setActiveSection: (activeSection) => set({ activeSection }),

  setWavelength: (wavelength) => set({ wavelength }),

  setSelectedRegion: (selectedRegion) => set({ selectedRegion }),

  fetchDemo: async (index = 0) => {
    set({ isLoading: true, error: null, demoIndex: index });
    try {
      const data = await getDemoPrediction(index);
      set({
        prediction: data,
        selectedRegion: data.active_regions?.[0] || null,
        isLoading: false,
      });
    } catch {
      const fallback = generateFallbackPrediction(index);
      set({
        prediction: fallback,
        selectedRegion: fallback.active_regions?.[0] || null,
        isLoading: false,
        error: "Live backend offline — displaying offline mission reference telemetry.",
      });
    }
  },

  predict: async (file: File) => {
    set({ isLoading: true, error: null });
    try {
      const data = await predictFile(file);
      set({
        prediction: data,
        selectedRegion: data.active_regions?.[0] || null,
        isLoading: false,
      });
    } catch (err) {
      set({
        isLoading: false,
        error: err instanceof Error ? err.message : "FITS observation parsing failed.",
      });
    }
  },

  fetchHealth: async () => {
    try {
      const healthData = await getHealth();
      set({ health: healthData });
    } catch {
      set({ health: null });
    }
  },
}));
